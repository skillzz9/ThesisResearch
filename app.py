"""
app.py -- simple web GUI for per-axis loop compatibility.
Drag/drop two audio loops, get a per-axis compatibility read in the browser.

    python app.py            # then open the printed http://127.0.0.1:7860 URL
"""
import numpy as np
import torch
import librosa
import gradio as gr
from train_model import CompatModel, AXES, TARGET_T
from left_eye.feature_extractor import VisualFeatureExtractor
import midi_utils as M

NICE = {"key": "key", "vertical": "consonance", "tempo": "tempo", "timing": "timing"}
THRESHOLDS = {"key": 0.47, "vertical": 0.60, "tempo": 0.40, "timing": 0.40}
COLOR = {"key": "#9b2f7a", "vertical": "#b0472f", "tempo": "#0d7d8c", "timing": "#1f6f54"}
MODEL_PATH = "models/compat_model.pth"

_vfe = VisualFeatureExtractor(sample_rate=M.RENDER_SR)
_model = CompatModel()
_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
_model.eval()


def _to_tensor(path):
    y, _ = librosa.load(path, sr=M.RENDER_SR, mono=True)
    feat = _vfe.extract_and_stack(y.astype(np.float32))
    t = torch.nan_to_num(torch.tensor(feat, dtype=torch.float32), nan=0.0)
    T = t.shape[2]
    t = torch.cat([t, torch.zeros(3, 84, TARGET_T - T)], dim=2) if T < TARGET_T else t[:, :, :TARGET_T]
    return t.unsqueeze(0)


def compare(a_path, b_path):
    if not a_path or not b_path:
        return "<p style='color:#888'>Upload two loops, then click Check.</p>"
    with torch.no_grad():
        p = torch.sigmoid(_model(_to_tensor(a_path), _to_tensor(b_path)))[0].numpy()
    rows, clashes = "", []
    for j, ax in enumerate(AXES):
        prob = float(p[j]); thr = THRESHOLDS[ax]
        clash = prob <= thr
        if clash:
            clashes.append(NICE[ax])
        pct = int(prob * 100)
        label = "CLASH" if clash else "compatible"
        lc = "#c23a31" if clash else "#1f8a5b"
        rows += f"""
        <div style="margin:14px 0">
          <div style="display:flex;justify-content:space-between;font-size:15px;margin-bottom:4px">
            <b style="color:{COLOR[ax]}">{NICE[ax]}</b>
            <span style="color:{lc};font-weight:700">{label} &nbsp;({prob:.2f})</span>
          </div>
          <div style="background:#eee;border-radius:6px;height:16px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{COLOR[ax]};opacity:{0.4 if clash else 1}"></div>
          </div>
        </div>"""
    verdict = ("<div style='font-size:17px;font-weight:700;color:#1f8a5b;margin-top:10px'>"
               "&#10003; Compatible on all four axes</div>") if not clashes else (
               f"<div style='font-size:17px;font-weight:700;color:#c23a31;margin-top:10px'>"
               f"&#9888; Clash on: {', '.join(clashes)}</div>")
    return (f"<div style='font-family:system-ui;max-width:520px'>"
            f"<div style='font-size:13px;color:#888;margin-bottom:6px'>bar = P(compatible); "
            f"faded + labelled CLASH = below the decision threshold</div>{rows}{verdict}</div>")


with gr.Blocks(title="Loop Compatibility") as demo:
    gr.Markdown("# 🎼 Per-Axis Loop Compatibility\nUpload two audio loops — the model reports "
                "whether they clash on **key, consonance, tempo, or timing**.")
    with gr.Row():
        a = gr.Audio(label="Loop A", type="filepath")
        b = gr.Audio(label="Loop B", type="filepath")
    btn = gr.Button("Check compatibility", variant="primary")
    out = gr.HTML()
    btn.click(compare, inputs=[a, b], outputs=out)
    gr.Markdown("Tip: generate test loops with `python make_test_loops.py` "
                "(compatible + one clash per axis in `test_loops/`).")

if __name__ == "__main__":
    demo.launch()
