"""
predict_from_audio.py
---------------------
The "play with it yourself" tool: feed it TWO audio loops, get a per-axis
compatibility score (key, consonance, tempo, timing). Works on any wav/mp3/flac.

    python predict_from_audio.py loopA.wav loopB.wav
    python predict_from_audio.py A.wav B.wav --model models/compat_model.pth

Prints, per axis, P(compatible) 0-1 and a compatible/CLASH verdict using the
calibrated decision thresholds from the 100-song run.
"""
import sys, argparse
import numpy as np
import torch
import librosa
from train_model import CompatModel, AXES, TARGET_T
from left_eye.feature_extractor import VisualFeatureExtractor
import midi_utils as M

NICE = {"key": "key", "vertical": "consonance", "tempo": "tempo", "timing": "timing"}
# calibrated thresholds from the 100-song k-fold run (train-side balanced-accuracy optimum)
DEFAULT_THRESHOLDS = {"key": 0.47, "vertical": 0.60, "tempo": 0.40, "timing": 0.40}


def audio_to_tensor(path, vfe, target_t=TARGET_T):
    """Load any audio file -> [3,84,T] feature tensor the model expects."""
    y, _ = librosa.load(path, sr=M.RENDER_SR, mono=True)
    if y.size == 0:
        raise ValueError(f"{path} is empty/unreadable")
    feat = vfe.extract_and_stack(y.astype(np.float32))          # [3,84,T]
    t = torch.nan_to_num(torch.tensor(feat, dtype=torch.float32), nan=0.0)
    # pad/crop time to target_t (same as load_img in training)
    T = t.shape[2]
    if T < target_t:
        t = torch.cat([t, torch.zeros(3, 84, target_t - T)], dim=2)
    else:
        t = t[:, :, :target_t]
    return t.unsqueeze(0)                                        # [1,3,84,T]


def score(model, a, b, thr):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(a, b))[0].cpu().numpy()          # P(compatible) per axis
    out = {}
    for j, ax in enumerate(AXES):
        prob = float(p[j])
        out[ax] = (prob, "compatible" if prob > thr[ax] else "CLASH")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loopA"); ap.add_argument("loopB")
    ap.add_argument("--model", default="models/compat_model.pth")
    a = ap.parse_args()

    vfe = VisualFeatureExtractor(sample_rate=M.RENDER_SR)
    model = CompatModel()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    ta = audio_to_tensor(a.loopA, vfe)
    tb = audio_to_tensor(a.loopB, vfe)
    verdicts = score(model, ta, tb, DEFAULT_THRESHOLDS)

    print(f"\n  Loop A: {a.loopA}")
    print(f"  Loop B: {a.loopB}")
    print(f"  model : {a.model}\n")
    print(f"  {'axis':12s} {'P(compatible)':>14s}   verdict")
    print("  " + "-" * 40)
    n_clash = 0
    for ax in AXES:
        prob, v = verdicts[ax]
        bar = "#" * int(prob * 20)
        flag = "  <-- CLASH" if v == "CLASH" else ""
        if v == "CLASH":
            n_clash += 1
        print(f"  {NICE[ax]:12s} {prob:>10.2f}    {bar}{flag}")
    print("  " + "-" * 40)
    if n_clash == 0:
        print("  => these loops are COMPATIBLE on all four axes\n")
    else:
        clashed = ", ".join(NICE[ax] for ax in AXES if verdicts[ax][1] == "CLASH")
        print(f"  => CLASH detected on: {clashed}\n")


if __name__ == "__main__":
    main()
