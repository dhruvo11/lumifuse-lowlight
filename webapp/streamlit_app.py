"""LumiFuse web app — Streamlit version (deployed on Streamlit Community Cloud).
Local run:  streamlit run webapp/streamlit_app.py"""
import base64
import glob
import html
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
import torch  # noqa: E402

from lumifuse.niqe import niqe  # noqa: E402
from lumifuse.pipeline import LumiFuse  # noqa: E402
from lumifuse.utils import decode_image_bytes, load_image, resize_max_side, to_uint8  # noqa: E402

MAX_SIDE = 768
MAX_SIDE_COMPARE = 512
METHODS = {"Hybrid": "hybrid", "Classical": "classical", "Deep learning": "dlonly", "Zero-DCE": "zerodce"}
COMPARE = "Compare all"
LONG = {"Hybrid": "Hybrid · LumiFuse", "Classical": "Classical only", "Deep learning": "Deep learning only",
        "Zero-DCE": "Zero-DCE (official weights)"}
EXAMPLE_NAMES = {
    "lol_1.png": "Indoor shelf (LOL)", "lol_22.png": "Toys (LOL)", "lol_111.png": "Cupboard (LOL)",
    "dicm_27_very_dark.jpg": "Monument at night — very dark", "dicm_07_backlit.jpg": "Backlit building",
    "lime_10.png": "Night street", "lime_1.png": "Dim interior",
    "synthetic_extreme_dark.png": "Synthetic — extremely dark", "synthetic_nonuniform.png": "Synthetic — spotlight",
}

st.set_page_config(page_title="LumiFuse — See in the dark", page_icon="🌙", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--ap-bg:#f5f5f7;--ap-card:#ffffff;--ap-text:#1d1d1f;--ap-sub:#6e6e73;--ap-line:#d2d2d7;--ap-blue:#0071e3;--ap-tile:#f5f5f7}
.stApp,.stApp p,.stApp label,.stApp button,.stApp h1,.stApp h2,.stApp h3,.stApp input,.stApp li,
.stApp [data-testid="stMarkdownContainer"] *:not([data-testid="stIconMaterial"]){
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Inter","Helvetica Neue",Arial,sans-serif!important}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:var(--ap-bg)!important;
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Inter","Helvetica Neue",Arial,sans-serif!important;color:var(--ap-text)}
[data-testid="stHeader"]{background:transparent!important}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;height:0}
.block-container{max-width:1180px!important;padding:0 16px 40px!important}
#hero{text-align:center;padding:40px 8px 22px}
#hero .eyebrow{font-size:15px;font-weight:600;color:#bf4800;margin-bottom:10px}
#hero h1{font-size:clamp(40px,7vw,76px);line-height:1.04;font-weight:700;letter-spacing:-.035em;margin:0;padding:0;color:var(--ap-text)}
#hero p{font-size:clamp(17px,2.2vw,22px);line-height:1.4;color:var(--ap-sub);max-width:720px;margin:18px auto 0}
#hero .chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:24px}
#hero .chip{background:var(--ap-card);border:1px solid var(--ap-line);border-radius:980px;padding:8px 16px;font-size:14px;color:var(--ap-text)}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--ap-card);border-radius:22px!important;border:none!important;
 box-shadow:0 2px 6px rgba(0,0,0,.04),0 12px 40px rgba(0,0,0,.06);padding:8px 10px}
.ap-h{font-size:24px;font-weight:600;letter-spacing:-.02em;margin:0 0 2px;color:var(--ap-text)}
.ap-p{font-size:15px;color:var(--ap-sub);margin:0 0 10px}
.stButton>button[kind="primary"]{background:var(--ap-blue)!important;border:none!important;border-radius:980px!important;
 font-size:17px!important;font-weight:500!important;min-height:48px;width:100%}
.stDownloadButton>button{border-radius:980px!important}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:12px 0 4px}
.tile{background:var(--ap-tile);border-radius:16px;padding:14px}
.tile .k{font-size:12px;font-weight:600;color:var(--ap-sub);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:21px;font-weight:600;letter-spacing:-.02em;margin-top:4px;color:var(--ap-text);line-height:1.2}
.tile .s{font-size:12.5px;color:var(--ap-sub);margin-top:2px}
.notes{font-size:14px;color:var(--ap-sub);line-height:1.55}
.notes b{color:var(--ap-text)}
.placeholder{display:flex;align-items:center;justify-content:center;height:360px;border-radius:18px;background:var(--ap-tile);
 color:var(--ap-sub);font-size:15px;text-align:center;padding:20px}
#how{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:30px 0 8px}
@media (max-width:820px){#how{grid-template-columns:1fr}}
#how .f{background:var(--ap-card);border-radius:22px;padding:26px;box-shadow:0 2px 6px rgba(0,0,0,.04),0 12px 40px rgba(0,0,0,.06)}
#how .n{font-size:13px;font-weight:600;color:var(--ap-blue)}
#how h3{font-size:21px;font-weight:600;letter-spacing:-.02em;margin:6px 0 8px;color:var(--ap-text);padding:0}
#how p{font-size:15px;line-height:1.5;color:var(--ap-sub);margin:0}
#foot{text-align:center;font-size:13px;color:var(--ap-sub);padding:26px 0 10px;line-height:1.7}
#foot a{color:var(--ap-blue);text-decoration:none}
</style>
"""


@st.cache_resource(show_spinner="Loading models…")
def load_model():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return LumiFuse(device="cpu")


def _json(name):
    try:
        return json.load(open(os.path.join(ROOT, name)))
    except Exception:
        return {}


TEAM = _json("team.json")
SUMMARY = _json(os.path.join("results", "summary.json"))


def b64jpg(img, q=90):
    ok, buf = cv2.imencode(".jpg", to_uint8(img)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def png_bytes(img):
    ok, buf = cv2.imencode(".png", to_uint8(img)[:, :, ::-1])
    return buf.tobytes()


def slider(before, after, height=440):
    b, a = b64jpg(before), b64jpg(after)
    components.html(f"""
<div id="c" style="position:relative;width:100%;height:{height}px;border-radius:18px;overflow:hidden;background:#111;user-select:none">
 <img src="data:image/jpeg;base64,{a}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain">
 <img id="b" src="data:image/jpeg;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;clip-path:inset(0 50% 0 0)">
 <div id="l" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.5)"></div>
 <div id="h" style="position:absolute;top:50%;left:50%;width:38px;height:38px;margin:-19px 0 0 -19px;border-radius:50%;background:#fff;
   box-shadow:0 2px 10px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;font:600 14px -apple-system,sans-serif;color:#1d1d1f">⇆</div>
 <span style="position:absolute;left:12px;top:12px;background:rgba(0,0,0,.55);color:#fff;border-radius:980px;padding:4px 10px;font:500 12px -apple-system,sans-serif">Before</span>
 <span style="position:absolute;right:12px;top:12px;background:rgba(0,0,0,.55);color:#fff;border-radius:980px;padding:4px 10px;font:500 12px -apple-system,sans-serif">After</span>
 <input id="r" type="range" min="0" max="100" value="50" style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0">
</div>
<script>
const r=document.getElementById('r'),b=document.getElementById('b'),l=document.getElementById('l'),h=document.getElementById('h');
r.addEventListener('input',()=>{{const v=r.value;b.style.clipPath='inset(0 '+(100-v)+'% 0 0)';l.style.left=v+'%';h.style.left=v+'%';}});
</script>""", height=height + 4)


def niqe_fast(img):
    try:
        return niqe(resize_max_side(img, 512))
    except Exception:
        return float("nan")


def tile(k, v, s=""):
    return f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div><div class="s">{html.escape(s)}</div></div>'


def chips():
    out = []
    h = SUMMARY.get("main/LOL-v1 eval15", {}).get("Hybrid (LumiFuse, proposed)")
    if h:
        out += [f"<b>{h['psnr']:.2f} dB</b> PSNR on LOL", f"<b>{h['ssim']:.3f}</b> SSIM"]
    out += ["<b>1.0 M</b> parameters", "Runs on <b>CPU</b>", "Adaptive · no fixed constants"]
    return "".join(f'<span class="chip">{c}</span>' for c in out)


def run(img, label):
    lf = load_model()
    if label == COMPARE:
        outs, tiles, hyb = [], [], None
        t0 = time.perf_counter()
        for lab, m in METHODS.items():
            o, vis, info = lf.enhance(img, m)
            outs.append((o, LONG[lab]))
            tiles.append(tile(lab, f"{niqe_fast(o):.1f}", f"NIQE · {info['latency_ms']['total'] / 1000:.2f} s"))
            if m == "hybrid":
                hyb = (o, vis, info)
        body = (f'<div class="tiles">{"".join(tiles)}</div><div class="notes"><b>All four methods</b> in '
                f'{time.perf_counter() - t0:.1f} s. The slider shows the hybrid result; every output is under <b>Outputs</b>.</div>')
        return hyb[0], body, outs, hyb[1], hyb[2]
    o, vis, info = lf.enhance(img, METHODS[label])
    s = info.get("stats", {})
    lat = info.get("latency_ms", {})
    tiles = [
        tile("Condition", (s.get("condition") or "—").split(" /")[0].capitalize(),
             f"mean brightness {s['mean_luma']:.3f}" if s else LONG[label]),
        tile("Latency", f"{lat.get('total', 0) / 1000:.2f} s", "on CPU"),
        tile("NIQE", f"{niqe_fast(img):.1f} → {niqe_fast(o):.1f}", "naturalness · lower is better"),
        tile("Noise σ", f"{s['noise_sigma']:.4f}" if s else "—", "measured on input"),
    ]
    notes = [f"<b>{html.escape(LONG[label])}</b>"]
    if info.get("input_gain", 1.0) != 1.0:
        notes.append(f"Input exposure rescaled ×{info['input_gain']} into the network's training range.")
    if info.get("global_classical_trust", 0) > 0:
        notes.append(f"Only moderately dark, so <b>{100 * info['global_classical_trust']:.0f}%</b> weight goes to the classical branch.")
    if "fusion_classical_share" in info:
        notes.append(f"Reliability fusion: on average <b>{100 * info['fusion_classical_share']:.1f}%</b> of each pixel comes from the classical branch.")
    g = info.get("quality_gates", {})
    if g:
        fired = [k.replace("_", " ") for k in g if k != "residual_noise"]
        notes.append(f"Quality gates: residual noise σ {g.get('residual_noise', 0):.4f}"
                     + (" · applied " + ", ".join(fired) if fired else " · nothing to correct"))
    parts = {k: v for k, v in lat.items() if k != "total"}
    if parts:
        notes.append("Stage timing: " + " · ".join(f"{k.replace('_', ' ')} {v:.0f} ms" for k, v in parts.items()))
    body = f'<div class="tiles">{"".join(tiles)}</div><div class="notes">' + "<br>".join(notes) + "</div>"
    return o, body, [(o, LONG[label])], vis, info


# ----------------------------------------------------------------------------- page
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"""<div id="hero"><div class="eyebrow">LumiFuse · Low-light restoration</div><h1>See in the dark.</h1>
<p>Classical Retinex meets a Zero-DCE neural network. Every photo is measured first, restored twice, then fused
pixel by pixel — brighter, cleaner, true to colour.</p><div class="chips">{chips()}</div></div>""", unsafe_allow_html=True)

examples = {EXAMPLE_NAMES.get(os.path.basename(p), os.path.basename(p)): p
            for p in sorted(glob.glob(os.path.join(ROOT, "samples", "*")))}
left, right = st.columns([5, 7], gap="medium")
with left:
    with st.container(border=True):
        st.markdown('<div class="ap-h">Your photo</div><div class="ap-p">Upload one — dark and noisy is the point — or pick an example.</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
                              label_visibility="collapsed")
        ex = st.selectbox("Example", ["Choose an example…"] + list(examples), label_visibility="collapsed")
        img, err = None, None
        try:
            if up is not None:
                img = decode_image_bytes(up.getvalue())  # also rotates phone photos upright
            elif ex in examples:
                img = load_image(examples[ex])
        except Exception:
            err = "That file couldn't be read as an image. Try a JPG or PNG."
        if img is not None and min(img.shape[:2]) < 32:
            img, err = None, "That image is too small. Use at least 32 × 32 pixels."
        if img is not None:
            st.image(to_uint8(img), use_container_width=True)
        if err:
            st.error(err)
        method = st.segmented_control("Method", list(METHODS) + [COMPARE], default="Hybrid",
                                      label_visibility="collapsed") or "Hybrid"
        go = st.button("Enhance", type="primary", disabled=img is None, use_container_width=True)

if go and img is not None:
    orig = img.shape[:2]
    work = resize_max_side(img, MAX_SIDE_COMPARE if method == COMPARE else MAX_SIDE)
    try:
        with st.spinner("Measuring the light, restoring twice, fusing…"):
            out, body, outs, vis, info = run(work, method)
        if work.shape[:2] != orig:
            body += f'<div class="notes">Processed at {work.shape[1]}×{work.shape[0]} (input {orig[1]}×{orig[0]}).</div>'
        st.session_state.result = dict(before=work, out=out, body=body, outs=outs, vis=vis, info=info)
    except Exception as e:  # never show a stack trace
        st.session_state.result = None
        st.session_state.error = f"Something went wrong while enhancing ({type(e).__name__}). Try another or smaller image."

with right:
    with st.container(border=True):
        st.markdown('<div class="ap-h">Result</div><div class="ap-p">Drag across the image to compare before and after.</div>', unsafe_allow_html=True)
        res = st.session_state.get("result")
        if st.session_state.pop("error", None):
            st.error("Something went wrong while enhancing. Try another or smaller image.")
        if res:
            slider(res["before"], res["out"])
            st.markdown(res["body"], unsafe_allow_html=True)
            st.download_button("Download result (PNG)", png_bytes(res["out"]), "lumifuse.png", "image/png")
        else:
            st.markdown('<div class="placeholder">Your restored photo and the decisions LumiFuse made will appear here.</div>',
                        unsafe_allow_html=True)

res = st.session_state.get("result")
if res:
    with st.container(border=True):
        t1, t2, t3 = st.tabs(["Pipeline stages", "Outputs", "Decisions"])
        with t1:
            items = list(res["vis"].items())
            for i in range(0, len(items), 4):
                cols = st.columns(4)
                for c, (k, v) in zip(cols, items[i:i + 4]):
                    c.image(to_uint8(v), caption=k, use_container_width=True)
        with t2:
            cols = st.columns(min(4, len(res["outs"])))
            for c, (o, k) in zip(cols, res["outs"]):
                c.image(to_uint8(o), caption=k, use_container_width=True)
        with t3:
            st.json({k: v for k, v in res["info"].items() if k in ("stats", "classical_params", "quality_gates", "input_gain",
                     "global_classical_trust", "fusion_classical_share", "latency_ms", "params")})

st.markdown("""<div id="how">
<div class="f"><div class="n">01 · Measure</div><h3>Reads the light.</h3><p>Brightness, sensor noise, colour cast and how
unevenly the scene is lit are measured first. Those numbers set every parameter that follows.</p></div>
<div class="f"><div class="n">02 · Restore twice</div><h3>Two experts, one photo.</h3><p>A Retinex + CLAHE + guided-filter
pipeline and a Zero-DCE network guided by the same illumination map each restore the image.</p></div>
<div class="f"><div class="n">03 · Fuse &amp; verify</div><h3>Best of both, checked.</h3><p>A reliability map picks the
better branch per pixel. No-reference quality gates then fix exposure, residual noise and colour cast.</p></div>
</div>""", unsafe_allow_html=True)
members = " · ".join(m["name"] for m in TEAM.get("members", []))
gh = TEAM.get("github_url", "")
st.markdown(f"""<div id="foot">Problem P1 · Robust Low-Light Image Restoration<br>{html.escape(members)}
{'<br><a href="' + html.escape(gh) + '" target="_blank">Source code, design document &amp; evaluation ›</a>' if gh and '<user>' not in gh else ''}
<br>Photos are processed in memory and never stored.</div>""", unsafe_allow_html=True)
