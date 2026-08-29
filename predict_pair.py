"""
predict_pair.py
---------------
The deployment story in one script: load the trained model, hand it two loops, get
per-axis compatibility verdicts. Run without args to demo on held-out pairs the model
never trained on (one positive + one negative per axis), or pass two feature paths:

    ./.venv/bin/python predict_pair.py                       # demo on unseen pairs
    ./.venv/bin/python predict_pair.py a.pt b.pt             # judge a specific pair
"""
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_model import (CompatModel, PairDataset, split_by_track, AXES, load_img,
                         recalibrate_bn, predict_probs, best_thresholds)

NICE = {"key": "key", "vertical": "consonance", "tempo": "tempo", "timing": "timing"}


def load_calibrated(manifest="dataset.csv", model_path="models/compat_model.pth", device="cpu"):
    """Load the model, recalibrate BN, and derive per-axis thresholds from TRAIN data."""
    model = CompatModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    train_df, val_df, val_tracks = split_by_track(manifest)
    cl = DataLoader(PairDataset(train_df), batch_size=32)
    recalibrate_bn(model, cl, device)
    Ptr, Ytr = predict_probs(model, cl, device)
    thr = best_thresholds(Ptr, Ytr)
    return model, thr, val_df, val_tracks


def judge(model, thr, fa, fb, device="cpu"):
    """-> {axis: (P(compatible), verdict)} for one pair of feature files."""
    a = load_img(fa).unsqueeze(0).to(device)
    b = load_img(fb).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(a, b))[0].cpu()
    return {ax: (float(p[j]), "compatible" if p[j] > thr[ax] else "CLASH")
            for j, ax in enumerate(AXES)}


def pretty(verdicts, truth=None):
    cells = []
    for ax in AXES:
        prob, v = verdicts[ax]
        mark = ""
        if truth is not None:
            mark = " ✓" if (v == "compatible") == (truth[ax] == 1) else " ✗"
        cells.append(f"{NICE[ax]}={v}({prob:.2f}){mark}")
    return "  ".join(cells)


if __name__ == "__main__":
    device = "cpu"
    model, thr, val_df, val_tracks = load_calibrated(device=device)
    print(f"thresholds: " + " ".join(f"{NICE[a]}={t:.2f}" for a, t in thr.items()))

    if len(sys.argv) == 3:
        print(pretty(judge(model, thr, sys.argv[1], sys.argv[2], device)))
        sys.exit(0)

    # demo: one held-out example of each pair type (songs the model never saw)
    print(f"demo on UNSEEN songs: {val_tracks}\n")
    rng = np.random.RandomState(0)
    for typ in ["positive", "key", "vertical", "tempo", "timing"]:
        sub = val_df[val_df.type == typ]
        if len(sub) == 0:
            continue
        r = sub.iloc[rng.randint(len(sub))]
        truth = {ax: int(r[ax]) for ax in AXES}
        expected = "all compatible" if typ == "positive" else f"{NICE.get(typ, typ)} clash"
        print(f"[{typ:9s}] expected: {expected}")
        print(f"            {pretty(judge(model, thr, r['file_A'], r['file_B'], device), truth)}\n")
