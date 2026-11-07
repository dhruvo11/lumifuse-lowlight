#!/usr/bin/env bash
# Download the public datasets used for training and evaluation into ./data
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data
# LOL-v1 (official Google Drive mirror re-hosted on Hugging Face)
[ -s lol_dataset.zip ] || curl -L --retry 5 -o lol_dataset.zip "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip"
unzip -q -o lol_dataset.zip -d lol
# Zero-DCE official repo: pretrained weights + DICM/LIME test photos
[ -s zerodce.zip ] || curl -L --retry 5 -o zerodce.zip "https://github.com/Li-Chongyi/Zero-DCE/archive/refs/heads/master.zip"
unzip -q -o zerodce.zip -d zdce
# LOL-v2-real (Train + Test)
python3 - <<'PY'
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
api = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/okhater/lolv2-real"))
files = [s["rfilename"] for s in api["siblings"] if s["rfilename"].startswith(("Train/", "Test/"))]
import time
def get(f):
    dst = os.path.join("lolv2_real", f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    url = f"https://huggingface.co/datasets/okhater/lolv2-real/resolve/main/{f}"
    for attempt in range(5):  # retry transient network errors / timeouts
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst + ".part", "wb") as out:
                out.write(r.read())
            os.replace(dst + ".part", dst)
            return True
        except Exception as e:
            print(f"retry {attempt + 1}/5 {f}: {e}")
            time.sleep(2 * (attempt + 1))
    return False
with ThreadPoolExecutor(24) as ex:
    ok = list(ex.map(get, files))
print("LOL-v2-real files:", sum(ok), "/", len(files))
if not all(ok):
    raise SystemExit("some files failed to download - re-run the script (it resumes)")
PY
echo "done"
