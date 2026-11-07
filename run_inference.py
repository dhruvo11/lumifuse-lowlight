"""Batch inference CLI.

  python run_inference.py --input_dir ./samples --output_dir ./results/inference [--method hybrid] [--save_maps]
"""
import argparse
import glob
import json
import os

from lumifuse.pipeline import LumiFuse
from lumifuse.utils import load_image, save_image

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def main():
    ap = argparse.ArgumentParser(description="LumiFuse low-light restoration")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--method", default="hybrid", choices=["hybrid", "classical", "dlonly", "zerodce"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save_maps", action="store_true", help="also save intermediate maps")
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.input_dir, "*"))) if f.lower().endswith(EXTS)]
    if not files:
        raise SystemExit(f"No images found in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    lf = LumiFuse(device=args.device)
    log = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            out, vis, info = lf.enhance(load_image(f), args.method)
        except Exception as e:  # keep going on bad files
            print(f"skip {f}: {e}")
            continue
        save_image(os.path.join(args.output_dir, f"{name}_{args.method}.png"), out)
        if args.save_maps:
            for k, v in vis.items():
                slug = "".join(c if c.isalnum() else "_" for c in k).strip("_").lower()
                save_image(os.path.join(args.output_dir, f"{name}_{slug}.png"), v)
        log[name] = {k: v for k, v in info.items() if k in ("stats", "latency_ms", "quality_gates", "input_gain")}
        print(f"{name}: {info['latency_ms']['total']:.0f} ms")
    json.dump(log, open(os.path.join(args.output_dir, "run_log.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
