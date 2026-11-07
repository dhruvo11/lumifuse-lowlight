"""Deploy the Gradio app to a Hugging Face Space (needs `hf auth login` first).

  python scripts/deploy_hf.py --space <hf-username>/lumifuse
"""
import argparse
import json
import os
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPACE_README = """---
title: LumiFuse Low-Light Restoration
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Adaptive hybrid (Retinex + Zero-DCE) low-light restoration
---

# LumiFuse — illumination-adaptive low-light restoration

Classical Retinex / CLAHE / guided filtering + an illumination-guided Zero-DCE network, fused adaptively.
Source code, design document and full evaluation: {github}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. username/lumifuse")
    args = ap.parse_args()
    team = json.load(open(os.path.join(ROOT, "team.json")))
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "README.md")
        open(p, "w").write(SPACE_README.format(github=team.get("github_url", "")))
        api.upload_file(path_or_fileobj=p, path_in_repo="README.md", repo_id=args.space, repo_type="space")
    api.upload_folder(
        folder_path=ROOT, repo_id=args.space, repo_type="space",
        allow_patterns=["app.py", "requirements.txt", "team.json", "results/summary.json", "lumifuse/*.py",
                        "lumifuse/*.npz", "weights/*.pth", "samples/*"],
        commit_message="Deploy LumiFuse",
    )
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
