"""
diagnose_model.py — look at what the trained model ACTUALLY outputs.
Answers: does it emit 6 distinct numbers? are they collapsed to ~1 (compatible)?
does it learn clashes on TRAIN data (proving signal exists) vs VAL (generalization)?
"""
import torch, pandas as pd, numpy as np
from train_model import CompatModel, PairDataset, split_by_track, AXES
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CompatModel().to(device)
model.load_state_dict(torch.load("models/compat_model.pth", map_location=device))
model.eval()

train_df, val_df, val_tracks = split_by_track("dataset.csv")
print(f"val tracks held out: {val_tracks}\n")

def raw_outputs(df, tag, n=1):
    """Print raw sigmoid outputs for one example of each pair-type."""
    print(f"===== RAW OUTPUTS on {tag} =====")
    print(f"{'pair type':16s} " + " ".join(f"{a[:5]:>6s}" for a in AXES) + "   <- label")
    # positive (all 1) + one negative per axis
    shown = set()
    for _, r in df.iterrows():
        label = tuple(int(r[a]) for a in AXES)
        # identify which axis is the clash (the 0), or 'positive'
        zeros = [AXES[i] for i, v in enumerate(label) if v == 0]
        tag2 = "positive" if not zeros else f"neg:{zeros[0]}"
        if tag2 in shown:
            continue
        shown.add(tag2)
        a = torch.load(r["file_A"], weights_only=True)["image_tensor"]
        b = torch.load(r["file_B"], weights_only=True)["image_tensor"]
        from train_model import load_img
        a = load_img(r["file_A"]).unsqueeze(0).to(device)
        b = load_img(r["file_B"]).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(a, b)).cpu().numpy()[0]
        print(f"{tag2:16s} " + " ".join(f"{x:6.2f}" for x in p) + "   " + str(label))
    print()

def stats(df, tag):
    dl = DataLoader(PairDataset(df), batch_size=32)
    allp, ally = [], []
    with torch.no_grad():
        for a, b, y in dl:
            allp.append(torch.sigmoid(model(a.to(device), b.to(device))).cpu())
            ally.append(y)
    P = torch.cat(allp); Y = torch.cat(ally)
    print(f"===== {tag}: per-axis behaviour =====")
    print(f"{'axis':14s} {'mean_p':>7s} {'min_p':>7s} {'max_p':>7s} {'clash_recall':>13s}")
    for j, ax in enumerate(AXES):
        pj, yj = P[:, j], Y[:, j]
        pred = (pj > 0.5).float()
        neg = yj == 0
        cr = ((pred == yj) & neg).sum().item() / max(1, neg.sum().item())
        print(f"{ax:14s} {pj.mean():7.3f} {pj.min():7.3f} {pj.max():7.3f} {cr:13.2f}")
    print()

raw_outputs(train_df, "TRAIN")
raw_outputs(val_df, "VAL")
stats(train_df, "TRAIN (seen instruments)")
stats(val_df, "VAL (unseen instruments)")
