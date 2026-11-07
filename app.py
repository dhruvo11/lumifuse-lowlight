"""LumiFuse web app (Gradio). Run: python app.py  ->  http://localhost:7860"""
import glob
import html
import json
import os
import time
import traceback

import gradio as gr
import numpy as np
import torch

from lumifuse.niqe import niqe
from lumifuse.pipeline import LumiFuse
from lumifuse.utils import resize_max_side, to_float_rgb, to_uint8

torch.set_num_threads(max(1, os.cpu_count() or 1))
ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 768           # processing cap for one method (keeps CPU latency to a few seconds)
MAX_SIDE_COMPARE = 512   # cap when running all four methods
LF = LumiFuse(device="cpu")


def _load_json(name):
    try:
        return json.load(open(os.path.join(ROOT, name)))
    except Exception:
        return {}


TEAM = _load_json("team.json")
SUMMARY = _load_json(os.path.join("results", "summary.json"))

METHODS = {
    "Hybrid": "hybrid",
    "Classical": "classical",
    "Deep learning": "dlonly",
    "Zero-DCE": "zerodce",
}
COMPARE = "Compare all"
LONG = {"Hybrid": "Hybrid · LumiFuse", "Classical": "Classical only", "Deep learning": "Deep learning only",
        "Zero-DCE": "Zero-DCE (official weights)"}

# ----------------------------------------------------------------------------- styling
CSS = """
:root{--ap-bg:#f5f5f7;--ap-card:#ffffff;--ap-text:#1d1d1f;--ap-sub:#6e6e73;--ap-line:#d2d2d7;--ap-blue:#0071e3;
--ap-blue-h:#0077ed;--ap-tile:#f5f5f7;--ap-shadow:0 2px 6px rgba(0,0,0,.04),0 12px 40px rgba(0,0,0,.06)}
.dark{--ap-bg:#000000;--ap-card:#1c1c1e;--ap-text:#f5f5f7;--ap-sub:#a1a1a6;--ap-line:#3a3a3c;--ap-blue:#2997ff;
--ap-blue-h:#47a6ff;--ap-tile:#2c2c2e;--ap-shadow:none}
body,.gradio-container{background:var(--ap-bg)!important}
.gradio-container{max-width:1180px!important;margin:0 auto!important;padding:0 16px!important;
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Inter","Helvetica Neue",Arial,sans-serif!important;
 -webkit-font-smoothing:antialiased;color:var(--ap-text)}
footer{display:none!important}
#hero{text-align:center;padding:56px 8px 24px}
#hero .eyebrow{font-size:15px;font-weight:600;color:#bf4800;letter-spacing:.01em;margin-bottom:10px}
.dark #hero .eyebrow{color:#ff9f0a}
#hero h1{font-size:clamp(40px,7vw,76px)!important;line-height:1.04!important;font-weight:700!important;letter-spacing:-.035em;
 margin:0!important;color:var(--ap-text)!important;opacity:1!important}
#hero p.sub{font-size:clamp(17px,2.2vw,22px);line-height:1.4;color:var(--ap-sub);max-width:720px;margin:18px auto 0;font-weight:400;letter-spacing:-.01em}
#hero .chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:26px}
#hero .chip{background:var(--ap-card);border-radius:980px;padding:9px 16px;font-size:14px;color:var(--ap-text);box-shadow:var(--ap-shadow);border:1px solid var(--ap-line)}
#hero .chip b{font-weight:600}
.ap-card{background:var(--ap-card)!important;border-radius:22px!important;box-shadow:var(--ap-shadow)!important;border:none!important;padding:22px!important}
.ap-card .block,.ap-card .form{border:none!important;box-shadow:none!important;background:transparent!important}
.ap-h{font-size:24px;font-weight:600;letter-spacing:-.02em;margin:2px 0 4px;color:var(--ap-text)}
.ap-p{font-size:15px;color:var(--ap-sub);margin:0 0 6px}
#seg .wrap{display:flex!important;flex-wrap:wrap;gap:4px;background:var(--ap-tile);border-radius:12px;padding:4px}
#seg label{flex:1 1 auto;justify-content:center;border:none!important;background:transparent!important;border-radius:9px!important;
 padding:7px 9px!important;font-size:13.5px!important;white-space:nowrap;font-weight:500;color:var(--ap-sub)!important;box-shadow:none!important;cursor:pointer;transition:all .2s}
#seg label.selected{background:var(--ap-card)!important;color:var(--ap-text)!important;box-shadow:0 1px 4px rgba(0,0,0,.12)!important}
#seg input[type=radio]{display:none!important}
#go{background:var(--ap-blue)!important;color:#fff!important;border:none!important;border-radius:980px!important;font-size:17px!important;
 font-weight:500!important;padding:13px 24px!important;min-height:48px;transition:background .2s,transform .1s}
#go:hover{background:var(--ap-blue-h)!important}
#go:active{transform:scale(.98)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:4px}
.tile{background:var(--ap-tile);border-radius:16px;padding:14px 14px 12px;min-width:0}
.tile .k{font-size:12px;font-weight:600;color:var(--ap-sub);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:21px;font-weight:600;letter-spacing:-.02em;margin-top:4px;color:var(--ap-text);line-height:1.2}
.tile .s{font-size:12.5px;color:var(--ap-sub);margin-top:2px}
.notes{font-size:14px;color:var(--ap-sub);margin-top:12px;line-height:1.55}
.notes b{color:var(--ap-text);font-weight:600}
.placeholder{font-size:15px;color:var(--ap-sub);padding:8px 2px}
.err{background:#fff1f0;color:#b3261e;border-radius:12px;padding:12px 14px;font-size:14px}
.dark .err{background:#3a1d1b;color:#ffb4ab}
#how{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:28px 0 8px}
@media (max-width:820px){#how{grid-template-columns:1fr}}
#how .f{background:var(--ap-card);border-radius:22px;padding:26px;box-shadow:var(--ap-shadow)}
#how .n{font-size:13px;font-weight:600;color:var(--ap-blue)}
#how h3{font-size:21px;font-weight:600;letter-spacing:-.02em;margin:6px 0 8px;color:var(--ap-text)}
#how p{font-size:15px;line-height:1.5;color:var(--ap-sub);margin:0}
#foot{text-align:center;font-size:13px;color:var(--ap-sub);padding:26px 0 40px;line-height:1.7}
#foot a{color:var(--ap-blue);text-decoration:none}
.tabs button{font-size:15px!important}
"""


def _theme():
    font = ["-apple-system", "BlinkMacSystemFont", gr.themes.GoogleFont("Inter"), "Helvetica Neue", "sans-serif"]
    return gr.themes.Base(primary_hue="blue", neutral_hue="gray", radius_size="lg", font=font).set(
        body_background_fill="#f5f5f7", body_background_fill_dark="#000000",
        block_background_fill="#ffffff", block_background_fill_dark="#1c1c1e",
        block_border_width="0px", block_shadow="none", block_radius="18px",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color="#6e6e73", block_label_border_width="0px",
        button_primary_background_fill="#0071e3", button_primary_background_fill_hover="#0077ed",
        button_primary_text_color="#ffffff", input_background_fill="#f5f5f7", input_background_fill_dark="#2c2c2e",
        color_accent_soft="#e8f2fd", color_accent_soft_dark="#0a2a4a",
    )


# ----------------------------------------------------------------------------- helpers
def _niqe_fast(img):
    try:
        return niqe(resize_max_side(img, 512))
    except Exception:
        return float("nan")


def _tile(k, v, s=""):
    return f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>' \
           f'<div class="s">{html.escape(s)}</div></div>'


def _hero_chips():
    chips = []
    lol = SUMMARY.get("main/LOL-v1 eval15", {})
    h = lol.get("Hybrid (LumiFuse, proposed)")
    if h:
        chips.append(f"<b>{h['psnr']:.2f} dB</b> PSNR on LOL")
        chips.append(f"<b>{h['ssim']:.3f}</b> SSIM")
    chips += ["<b>1.0 M</b> parameters", "Runs on <b>CPU</b>", "Adaptive · no fixed constants"]
    return "".join(f'<span class="chip">{c}</span>' for c in chips)


def _result_html(label, img, out, info, resized_from):
    s = info.get("stats", {})
    lat = info.get("latency_ms", {}).get("total", 0)
    n0, n1 = _niqe_fast(img), _niqe_fast(out)
    tiles = [
        _tile("Condition", (s.get("condition") or "—").split(" /")[0].capitalize(),
              f"mean brightness {s['mean_luma']:.3f}" if s else LONG.get(label, label)),
        _tile("Latency", f"{lat / 1000:.2f} s", "on CPU"),
        _tile("NIQE", f"{n0:.1f} → {n1:.1f}", "naturalness · lower is better"),
        _tile("Noise σ", f"{s['noise_sigma']:.4f}" if s else "—", "measured on input"),
    ]
    notes = [f"<b>{html.escape(LONG.get(label, label))}</b>"]
    if info.get("input_gain", 1.0) != 1.0:
        notes.append(f"Extremely dark input — pre-scaled ×{info['input_gain']} into the network's training range.")
    if info.get("global_classical_trust", 0) > 0:
        notes.append(f"Input is only moderately dark, so <b>{100 * info['global_classical_trust']:.0f}%</b> weight "
                     "goes to the classical branch (the network is strongest on very dark, noisy photos).")
    if "fusion_classical_share" in info:
        notes.append(f"Reliability fusion: on average <b>{100 * info['fusion_classical_share']:.1f}%</b> of each pixel "
                     "comes from the classical branch.")
    g = info.get("quality_gates", {})
    fired = [k for k in g if k != "residual_noise"]
    if g:
        notes.append("Quality gates: residual noise σ " + f"{g.get('residual_noise', 0):.4f}"
                     + (" · applied " + ", ".join(fired).replace("_", " ") if fired else " · nothing to correct"))
    if "well_lit_blend" in info:
        notes.append("Image is already well lit, so it was only lightly adjusted.")
    if resized_from:
        notes.append(f"Processed at {img.shape[1]}×{img.shape[0]} (input was {resized_from[1]}×{resized_from[0]}).")
    lat_parts = {k: v for k, v in info.get("latency_ms", {}).items() if k != "total"}
    if lat_parts:
        notes.append("Stage timing: " + " · ".join(f"{k.replace('_', ' ')} {v:.0f} ms" for k, v in lat_parts.items()))
    return f'<div class="tiles">{"".join(tiles)}</div><div class="notes">' + "<br>".join(notes) + "</div>"


def _details(info):
    keep = {k: v for k, v in info.items() if k in ("stats", "classical_params", "quality_gates", "input_gain",
                                                    "global_classical_trust", "fusion_classical_share",
                                                    "latency_ms", "params")}
    return keep or {"note": "No adaptive decisions for this method."}


# ----------------------------------------------------------------------------- inference
def enhance(image, method_label):
    if image is None:
        raise gr.Error("Add a photo first — upload one, paste it, or tap an example.")
    try:
        img = to_float_rgb(np.asarray(image))
    except Exception:
        raise gr.Error("That file couldn't be read as an image. Try a JPG or PNG.")
    if min(img.shape[:2]) < 32:
        raise gr.Error("That image is too small. Use at least 32 × 32 pixels.")
    compare = method_label == COMPARE
    orig = img.shape[:2]
    img = resize_max_side(img, MAX_SIDE_COMPARE if compare else MAX_SIDE)
    resized_from = orig if img.shape[:2] != orig else None
    try:
        if compare:
            t0 = time.perf_counter()
            gallery, tiles, hyb = [], [], None
            for label, m in METHODS.items():
                out, vis, info = LF.enhance(img, m)
                gallery.append((to_uint8(out), LONG[label]))
                tiles.append(_tile(label, f"{_niqe_fast(out):.1f}", f"NIQE · {info['latency_ms']['total'] / 1000:.2f} s"))
                if m == "hybrid":
                    hyb = (out, vis, info)
            body = (f'<div class="tiles">{"".join(tiles)}</div><div class="notes"><b>All four methods</b> in '
                    f'{time.perf_counter() - t0:.1f} s on CPU. The slider shows the hybrid result; '
                    "every output is under <b>Outputs</b>.</div>")
            inter = [(to_uint8(v), k) for k, v in hyb[1].items()]
            return (to_uint8(img), to_uint8(hyb[0])), body, gallery, inter, _details(hyb[2])
        out, vis, info = LF.enhance(img, METHODS[method_label])
        inter = [(to_uint8(v), k) for k, v in vis.items()]
        return ((to_uint8(img), to_uint8(out)), _result_html(method_label, img, out, info, resized_from),
                [(to_uint8(out), LONG[method_label])], inter, _details(info))
    except gr.Error:
        raise
    except Exception as e:  # never surface a stack trace
        traceback.print_exc()
        raise gr.Error(f"Something went wrong while enhancing ({type(e).__name__}). Try another or smaller image.")


EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "samples", "*")))
MEMBERS = " · ".join(m["name"] for m in TEAM.get("members", []) if not m["name"].startswith("MEMBER"))
GITHUB = TEAM.get("github_url", "")

HERO = f"""<div id="hero"><div class="eyebrow">LumiFuse · Low-light restoration</div>
<h1>See in the dark.</h1>
<p class="sub">Classical Retinex meets a Zero-DCE neural network. Every photo is measured first, restored twice,
then fused pixel by pixel — brighter, cleaner, true to colour.</p>
<div class="chips">{_hero_chips()}</div></div>"""

HOW = """<div id="how">
<div class="f"><div class="n">01 · Measure</div><h3>Reads the light.</h3><p>Brightness, sensor noise, colour cast and how
unevenly the scene is lit are measured first. Those numbers set every parameter that follows.</p></div>
<div class="f"><div class="n">02 · Restore twice</div><h3>Two experts, one photo.</h3><p>A Retinex + CLAHE + guided-filter
pipeline and a Zero-DCE network guided by the same illumination map each restore the image.</p></div>
<div class="f"><div class="n">03 · Fuse &amp; verify</div><h3>Best of both, checked.</h3><p>A reliability map picks the
better branch per pixel. No-reference quality gates then fix exposure, residual noise and colour cast.</p></div>
</div>"""

FOOT = (f'<div id="foot">Problem P1 · Robust Low-Light Image Restoration'
        + (f"<br>{html.escape(MEMBERS)}" if MEMBERS else "")
        + (f'<br><a href="{html.escape(GITHUB)}" target="_blank">Source code, design document &amp; evaluation ›</a>'
           if GITHUB and "<user>" not in GITHUB else "")
        + "<br>Photos are processed in memory and never stored.</div>")

with gr.Blocks(title="LumiFuse — See in the dark") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320, elem_classes="ap-card"):
            gr.HTML('<div class="ap-h">Your photo</div><div class="ap-p">Upload, paste or take one. Dark and noisy is the point.</div>')
            inp = gr.Image(show_label=False, type="numpy", sources=["upload", "clipboard", "webcam"], height=340)
            method = gr.Radio(list(METHODS) + [COMPARE], value="Hybrid", show_label=False, elem_id="seg")
            btn = gr.Button("Enhance", elem_id="go")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="Try an example", examples_per_page=12)
        with gr.Column(scale=7, min_width=320, elem_classes="ap-card"):
            gr.HTML('<div class="ap-h">Result</div><div class="ap-p">Drag the handle to compare before and after.</div>')
            slider = gr.ImageSlider(show_label=False, type="numpy", format="png", height=430)
            info_html = gr.HTML('<div class="placeholder">Your restored photo and the decisions LumiFuse made will appear here.</div>')
    with gr.Column(elem_classes="ap-card"):
        with gr.Tabs():
            with gr.Tab("Pipeline stages"):
                inter = gr.Gallery(show_label=False, columns=4, height=360, format="png", object_fit="contain")
            with gr.Tab("Outputs"):
                outs = gr.Gallery(show_label=False, columns=4, height=360, format="png", object_fit="contain")
            with gr.Tab("Decisions"):
                details = gr.JSON(show_label=False)
    gr.HTML(HOW)
    gr.HTML(FOOT)
    btn.click(enhance, [inp, method], [slider, info_html, outs, inter, details],
              concurrency_limit=2, show_progress="full")

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   theme=_theme(), css=CSS, show_error=True)
