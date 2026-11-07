#!/usr/bin/env bash
# Download the public datasets used for training and evaluation into ./data
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data
# LOL-v1 (official Google Drive mirror re-hosted on Hugging Face)
curl -L -o lol_dataset.zip "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip"
unzip -q -o lol_dataset.zip -d lol
# Zero-DCE official repo: pretrained weights + DICM/LIME test photos
curl -L -o zerodce.zip "https://github.com/Li-Chongyi/Zero-DCE/archive/refs/heads/master.zip"
unzip -q -o zerodce.zip -d zdce
# LOL-v2-real (Train + Test)
python3 - <<'PY'
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
api = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/okhater/lolv2-real"))
files = [s["rfilename"] for s in api["siblings"] if s["rfilename"].startswith(("Train/", "Test/"))]
def get(f):
    dst = os.path.join("lolv2_real", f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/okhater/lolv2-real/resolve/main/{f}", dst)
with ThreadPoolExecutor(16) as ex:
    list(ex.map(get, files))
print("LOL-v2-real files:", len(files))
PY
echo "done"
