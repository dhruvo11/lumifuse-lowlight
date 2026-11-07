"""Build leak-free train/val manifests.

Train pool = LOL-v1 our485 + LOL-v2-real Train. Any image whose low-light input
also appears in a test set (LOL-v1 eval15, LOL-v2-real Test) is removed, and the
pool is de-duplicated (LOL-v2-real re-uses most LOL-v1 images). 5% is held out
for checkpoint selection so the test sets are never used for model selection.
"""
import glob
import hashlib
import json
import os
import random
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def key(path):
    im = cv2.imread(path)
    return hashlib.md5(cv2.resize(im, (60, 40), interpolation=cv2.INTER_AREA).tobytes()).hexdigest()


def pairs(low_glob, high_dir, rename=lambda n: n):
    out = []
    for p in sorted(glob.glob(low_glob)):
        h = os.path.join(high_dir, rename(os.path.basename(p)))
        if os.path.exists(h):
            out.append((p, h))
    return out


def main():
    v1_train = pairs(f"{D}/lol/lol_dataset/our485/low/*.png", f"{D}/lol/lol_dataset/our485/high")
    v1_test = pairs(f"{D}/lol/lol_dataset/eval15/low/*.png", f"{D}/lol/lol_dataset/eval15/high")
    v2_train = pairs(f"{D}/lolv2_real/Train/Input/*.png", f"{D}/lolv2_real/Train/GT",
                     lambda n: n.replace("low", "normal"))
    v2_test = pairs(f"{D}/lolv2_real/Test/Input/*.png", f"{D}/lolv2_real/Test/GT",
                    lambda n: n.replace("low", "normal"))
    print("found", len(v1_train), len(v1_test), len(v2_train), len(v2_test))
    test_keys = {key(p) for p, _ in v1_test + v2_test}
    seen, pool, dropped_leak, dropped_dup = set(), [], 0, 0
    for low, high in v1_train + v2_train:
        k = key(low)
        if k in test_keys:
            dropped_leak += 1
            continue
        if k in seen:
            dropped_dup += 1
            continue
        seen.add(k)
        pool.append((os.path.relpath(low, ROOT), os.path.relpath(high, ROOT)))
    random.Random(0).shuffle(pool)
    n_val = max(20, len(pool) // 20)
    man = {
        "train": pool[n_val:], "val": pool[:n_val],
        "test_lolv1": [(os.path.relpath(a, ROOT), os.path.relpath(b, ROOT)) for a, b in v1_test],
        "test_lolv2": [(os.path.relpath(a, ROOT), os.path.relpath(b, ROOT)) for a, b in v2_test],
        "stats": {"dropped_test_leak": dropped_leak, "dropped_duplicates": dropped_dup,
                  "train": len(pool) - n_val, "val": n_val},
    }
    json.dump(man, open(os.path.join(D, "manifest.json"), "w"), indent=1)
    print(json.dumps(man["stats"]))


if __name__ == "__main__":
    sys.exit(main())
