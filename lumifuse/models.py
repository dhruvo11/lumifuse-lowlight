"""Neural components.

* DCENet      - Zero-DCE curve estimator (Guo et al., CVPR 2020), ported from the
                official code. Optional extra input channel for the classical
                illumination prior.
* RefineNet   - small U-Net that removes noise / colour error left after the
                curve stage (curves can brighten but cannot denoise).
* LumiNet     - DCENet + RefineNet. With use_prior=True (hybrid) both stages are
                conditioned on classical priors: the guided-filter illumination
                map L and the per-pixel amplification (gain) map G.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)


# ----------------------------------------------------------------- classical priors (torch)
def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    mi, mp = box_filter(guide, r), box_filter(src, r)
    cov = box_filter(guide * src, r) - mi * mp
    var = box_filter(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box_filter(a, r) * guide + box_filter(b, r)


def illumination_prior(x: torch.Tensor, r: int = 8, eps: float = 1e-2) -> torch.Tensor:
    """Retinex illumination estimate: max-RGB refined by an edge-preserving guided filter."""
    l0 = x.max(1, keepdim=True)[0]
    return guided_filter(l0, l0, r, eps).clamp(1e-3, 1.0)


def luma(x: torch.Tensor) -> torch.Tensor:
    return LUMA[0] * x[:, 0:1] + LUMA[1] * x[:, 1:2] + LUMA[2] * x[:, 2:3]


# ----------------------------------------------------------------- Zero-DCE
class DCENet(nn.Module):
    def __init__(self, in_ch: int = 3, nf: int = 32, n_iter: int = 8):
        super().__init__()
        self.n_iter = n_iter
        self.e_conv1 = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 3 * n_iter, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def curves(self, inp: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(inp))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    @staticmethod
    def apply_curves(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Iterative light-enhancement curve LE(x) = x + a*x*(1-x), applied n times."""
        for r in torch.split(a, 3, dim=1):
            x = x + r * (x * x - x)
        return x

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None, scale: int = 1):
        inp = x if extra is None else torch.cat([x, extra], 1)
        if scale > 1:  # Zero-DCE++ trick: estimate smooth curves at low res, upsample.
            small = F.interpolate(inp, scale_factor=1 / scale, mode="bilinear", align_corners=False)
            a = F.interpolate(self.curves(small), size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            a = self.curves(inp)
        return self.apply_curves(x, a), a


def load_official_zerodce(path: str, in_ch: int = 3) -> DCENet:
    """Load the authors' Epoch99.pth; extra input channels start at zero so the
    hybrid model is initialised to behave exactly like official Zero-DCE."""
    net = DCENet(in_ch=in_ch)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if in_ch != 3:
        w = sd["e_conv1.weight"]
        sd["e_conv1.weight"] = torch.cat([w, torch.zeros(w.shape[0], in_ch - 3, 3, 3)], 1)
    net.load_state_dict(sd)
    return net


# ----------------------------------------------------------------- Refinement U-Net
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
    )


class RefineNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = _block(in_ch, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.mid = _block(4 * b, 4 * b)
        self.u3 = nn.ConvTranspose2d(4 * b, 4 * b, 2, 2)
        self.d3 = _block(8 * b, 2 * b)
        self.u2 = nn.ConvTranspose2d(2 * b, 2 * b, 2, 2)
        self.d2 = _block(4 * b, b)
        self.u1 = nn.ConvTranspose2d(b, b, 2, 2)
        self.d1 = _block(2 * b, b)
        self.out = nn.Conv2d(b, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ----------------------------------------------------------------- full network
class LumiNet(nn.Module):
    def __init__(self, use_prior: bool = True, base: int = 32):
        super().__init__()
        self.use_prior = use_prior
        self.dce = DCENet(in_ch=4 if use_prior else 3)
        self.refine = RefineNet(in_ch=5 if use_prior else 3, base=base)

    def forward(self, x: torch.Tensor, dce_scale: int = 2, use_refine: bool = True) -> dict:
        out = {}
        if self.use_prior:
            L = illumination_prior(x)
            enh, a = self.dce(x, extra=L, scale=dce_scale)
            enh = enh.clamp(0, 1)
            # Stage separation: the refiner may not reshape the curve stage, so the
            # curve stage stays a genuine enhancement (trained by its own loss).
            gain = luma(enh.detach()).clamp_min(1e-3) / (luma(x) + 0.02)
            g_feat = torch.log(gain.clamp_min(1.0)) / 4.0
            ref_in = torch.cat([enh.detach(), L, g_feat], 1)
            out.update(illum=L, gain=gain)
        else:
            enh, a = self.dce(x, scale=dce_scale)
            enh = enh.clamp(0, 1)
            ref_in = enh.detach()
        out.update(dce=enh, curves=a)
        if use_refine:
            h, w = x.shape[-2:]
            ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
            ri = F.pad(ref_in, (0, pw, 0, ph), mode="reflect") if (ph or pw) else ref_in
            res = self.refine(ri)[..., :h, :w]
            out["out"] = (enh.detach() + res).clamp(0, 1)
        else:
            out["out"] = enh
        return out


def init_luminet(use_prior: bool, official_path: str) -> LumiNet:
    net = LumiNet(use_prior=use_prior)
    net.dce.load_state_dict(load_official_zerodce(official_path, in_ch=4 if use_prior else 3).state_dict())
    return net
