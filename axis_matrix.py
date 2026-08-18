"""
axis_matrix.py — the explainability cross-talk test.

For each pair type (positive, and each single-axis negative) print the model's
AVERAGE 6-number output. The IDEAL is a clean diagonal:
  - positive row       -> all axes HIGH (all compatible)
  - neg:<axis> row      -> ONLY <axis> LOW, every other axis HIGH
If breaking tempo also drops 'key', that's cross-talk = a broken explanation.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from train_model import CompatModel, PairDataset, split_by_track, AXES

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CompatModel().to(device)
model.load_state_dict(torch.load("models/compat_model.pth", map_location=device))
model.eval()

train_df, val_df, val_tracks = split_by_track("dataset.csv")

def pair_type(row):
    zeros = [AXES[i] for i in range(len(AXES)) if int(row[AXES[i]]) == 0]
    return "positive" if not zeros else f"neg:{zeros[0]}"

def matrix(df, tag):
    df = df.copy()
    df["ptype"] = df.apply(pair_type, axis=1)
    print(f"\n===== {tag} : average P(compatible) per axis =====")
    print(f"{'pair type':16s} " + " ".join(f"{a[:5]:>6s}" for a in AXES))
    order = ["positive"] + [f"neg:{a}" for a in AXES]
    for pt in order:
        sub = df[df["ptype"] == pt]
        if len(sub) == 0:
            continue
        ds = PairDataset(sub)
        dl = DataLoader(ds, batch_size=32)
        outs = []
        with torch.no_grad():
            for a, b, y in dl:
                outs.append(torch.sigmoid(model(a.to(device), b.to(device))).cpu().numpy())
        m = np.concatenate(outs).mean(axis=0)
        # mark the axis that SHOULD be low for this negative
        target = None if pt == "positive" else pt.split(":")[1]
        cells = []
        for j, ax in enumerate(AXES):
            s = f"{m[j]:6.2f}"
            cells.append(("[" + s.strip() + "]").rjust(6) if ax == target else s)
        print(f"{pt:16s} " + " ".join(cells) + (f"   (n={len(sub)})"))
    print("  [x] = the axis that SHOULD be low.  Everything else should be ~high (>0.5).")

matrix(val_df, "UNSEEN (val)")
matrix(train_df, "SEEN (train)")
