"""Fine-tune LumiNet on the leak-free LOL train split.

  python scripts/train.py --variant hybrid   # DCE + RefineNet conditioned on classical priors
  python scripts/train.py --variant dlonly   # identical network without the priors

Both start from the official Zero-DCE weights. Losses: L1 + SSIM + edge (Sobel)
+ colour-angle on the final output (trains RefineNet); low-frequency L1 + colour
angle + the Zero-DCE illumination-smoothness (TV) regulariser on the curve stage
(trains DCE-Net; its output is detached before the refiner).
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lumifuse.models import init_luminet  # noqa: E402


# ------------------------------------------------------------------ losses
def _gauss(ws=11, sigma=1.5, ch=3, device="cpu"):
    g = torch.exp(-((torch.arange(ws) - ws // 2) ** 2).float() / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(ch, 1, ws, ws).contiguous().to(device)


def ssim(x, y, win):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ch = x.shape[1]
    mx, my = F.conv2d(x, win, groups=ch), F.conv2d(y, win, groups=ch)
    sxx = F.conv2d(x * x, win, groups=ch) - mx ** 2
    syy = F.conv2d(y * y, win, groups=ch) - my ** 2
    sxy = F.conv2d(x * y, win, groups=ch) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def sobel(x):
    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    kx = k.expand(3, 1, 3, 3)
    ky = k.t().expand(3, 1, 3, 3)
    return torch.cat([F.conv2d(x, kx, padding=1, groups=3), F.conv2d(x, ky, padding=1, groups=3)], 1)


def color_angle(x, y):
    return (1 - F.cosine_similarity(x + 1e-4, y + 1e-4, dim=1)).mean()


def tv(a):
    return ((a[..., 1:, :] - a[..., :-1, :]) ** 2).mean() + ((a[..., :, 1:] - a[..., :, :-1]) ** 2).mean()


# ------------------------------------------------------------------ data
def load_pairs(items):
    out = []
    for low, high in items:
        lo = cv2.imread(os.path.join(ROOT, low))[:, :, ::-1].copy()
        hi = cv2.imread(os.path.join(ROOT, high))[:, :, ::-1].copy()
        if lo.shape != hi.shape:
            hi = cv2.resize(hi, (lo.shape[1], lo.shape[0]))
        out.append((lo, hi))
    return out


def smooth_field(h, w, rng):
    """Random smooth illumination field in [lo, 1] (simulates non-uniform light)."""
    g = rng.uniform(0, 1, (3, 3)).astype(np.float32)
    f = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    lo = rng.uniform(0.25, 0.7)
    return lo + (1 - lo) * f


def batch(pairs, bs, crop, rng):
    xs, ys = [], []
    for _ in range(bs):
        lo, hi = pairs[rng.integers(len(pairs))]
        h, w = lo.shape[:2]
        i, j = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
        x = lo[i:i + crop, j:j + crop].astype(np.float32) / 255
        y = hi[i:i + crop, j:j + crop].astype(np.float32) / 255
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        k = rng.integers(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        # Illumination augmentation -> generalisation across lighting conditions.
        r = rng.random()
        if r < 0.3:
            x = x * rng.uniform(0.3, 1.0)                   # darker
        elif r < 0.45:
            x = np.clip(x * rng.uniform(1.0, 2.0), 0, 1)    # brighter
        elif r < 0.65:
            x = x * smooth_field(crop, crop, rng)[..., None]  # non-uniform
        x = np.round(np.clip(x, 0, 1) * 255) / 255           # keep 8-bit quantisation
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
    t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float()
    return t(xs), t(ys)


@torch.no_grad()
def validate(net, pairs, device):
    net.eval()
    ps = []
    for lo, hi in pairs:
        x = torch.from_numpy(lo.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)
        y = torch.from_numpy(hi.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)
        o = net(x)["out"]
        mse = F.mse_loss(o, y).item()
        ps.append(10 * math.log10(1 / max(mse, 1e-10)))
    net.train()
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["hybrid", "dlonly"], default="hybrid")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--val_every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    train, val = load_pairs(man["train"]), load_pairs(man["val"])
    print(f"device={device} train={len(train)} val={len(val)}", flush=True)

    net = init_luminet(args.variant == "hybrid", os.path.join(ROOT, "weights", "zerodce_official.pth")).to(device)
    opt = torch.optim.AdamW([
        {"params": net.dce.parameters(), "lr": args.lr * 0.5},
        {"params": net.refine.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.5, args.lr], total_steps=args.iters,
                                                pct_start=0.05, final_div_factor=100)
    win = _gauss(device=device)
    out_path = os.path.join(ROOT, "weights", f"luminet_{args.variant}.pth")
    log = open(os.path.join(ROOT, "results", f"train_{args.variant}.log"), "w")
    best, t0 = -1.0, time.time()
    for it in range(1, args.iters + 1):
        x, y = batch(train, args.bs, args.crop, rng)
        x, y = x.to(device), y.to(device)
        o = net(x)
        out, dce = o["out"], o["dce"]
        loss_main = F.l1_loss(out, y) + 0.2 * (1 - ssim(out, y, win)) + 0.1 * F.l1_loss(sobel(out), sobel(y)) \
            + 0.1 * color_angle(out, y)
        dp, yp = F.avg_pool2d(dce, 4), F.avg_pool2d(y, 4)  # low-pass: curves cannot remove noise
        loss_dce = F.l1_loss(dp, yp) + 0.5 * color_angle(dp, yp) + 10 * tv(o["curves"])
        loss = loss_main + loss_dce  # refiner input is detached -> curve stage learns only from loss_dce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 50 == 0:
            msg = f"it {it} loss {loss.item():.4f} main {loss_main.item():.4f} dce {loss_dce.item():.4f} " \
                  f"{(time.time() - t0) / it:.3f}s/it"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if it % args.val_every == 0 or it == args.iters:
            p = validate(net, val, device)
            msg = f"VAL it {it} psnr {p:.3f} (best {max(best, p):.3f})"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            if p > best:
                best = p
                torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out_path)
    print("done best val psnr", best, flush=True)


if __name__ == "__main__":
    main()
