"""Robustness tests: odd sizes, gray/RGBA/16-bit inputs, black/white frames, NaN-free outputs.
Run: python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumifuse.pipeline import LumiFuse  # noqa: E402
from lumifuse.quality import analyze  # noqa: E402
from lumifuse.utils import to_float_rgb  # noqa: E402

LF = LumiFuse(device="cpu")
METHODS = ["classical", "zerodce"] + list(LF.nets)
rng = np.random.default_rng(0)


def _check(out, shape):
    assert out.shape == shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("hw", [(64, 64), (101, 77), (240, 321)])
def test_odd_sizes(method, hw):
    img = (rng.random((*hw, 3)) * 0.1).astype(np.float32)
    out, vis, info = LF.enhance(img, method)
    _check(out, img.shape)
    for v in vis.values():
        assert v.shape[:2] == img.shape[:2]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_frames(method, value):
    img = np.full((96, 128, 3), value, np.float32)
    out, _, _ = LF.enhance(img, method)
    _check(out, img.shape)


def test_input_formats():
    gray = (rng.random((50, 60)) * 255).astype(np.uint8)
    rgba = (rng.random((50, 60, 4)) * 255).astype(np.uint8)
    u16 = (rng.random((50, 60, 3)) * 65535).astype(np.uint16)
    for a in (gray, rgba, u16):
        f = to_float_rgb(a)
        assert f.shape == (50, 60, 3) and f.dtype == np.float32 and 0 <= f.min() and f.max() <= 1


def test_upload_decoding(tmp_path):
    import io

    from PIL import Image

    from lumifuse.utils import decode_image_bytes
    # A phone portrait photo: landscape pixels + EXIF "rotate 90" tag. Left half white.
    a = np.zeros((60, 100, 3), np.uint8)
    a[:, :50] = 255
    im = Image.fromarray(a)
    ex = im.getexif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=ex)
    up = decode_image_bytes(buf.getvalue())
    assert up.shape == (100, 60, 3)                     # shown upright, as on the phone
    assert up[:40].mean() > 0.8 and up[60:].mean() < 0.2  # white half rotated to the top
    for bad in (b"", b"this is not an image"):
        with pytest.raises(ValueError):
            decode_image_bytes(bad)


def test_condition_labels():
    dark = np.full((128, 128, 3), 0.02, np.float32)
    bright = np.full((128, 128, 3), 0.6, np.float32)
    assert analyze(dark)["condition"] == "extremely dark"
    assert analyze(bright)["condition"] == "well-lit"


def test_hybrid_does_not_overprocess_well_lit():
    img = np.clip(rng.normal(0.5, 0.1, (128, 128, 3)), 0, 1).astype(np.float32)
    out, _, info = LF.enhance(img, "hybrid") if "hybrid" in LF.nets else LF.enhance(img, "classical")
    assert abs(out.mean() - img.mean()) < 0.2
