"""
scaling_curve.py
----------------
The decisive PoC experiment: does generalisation IMPROVE as we add training songs?
Hold out a FIXED validation set, then train on 2, 4, 6, ... of the remaining songs and
measure per-axis UNSEEN AUC each time. A rising curve = "more data helps" = evidence the
full Slakh2100 run will work. A flat curve = data won't fix it (find out before the pod).
"""
import os, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import CompatModel, PairDataset, AXES, _auc


def train_eval(train_df, val_df, device, epochs, lr=3e-4, batch=32):
    model = CompatModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = [max(1, int((train_df[ax] == 0).sum())) / max(1, int((train_df[ax] == 1).sum())) for ax in AXES]
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))
    dl = DataLoader(PairDataset(train_df), batch_size=batch, shuffle=True, num_workers=2)
    for _ in range(epochs):
        model.train()
        for a, b, y in dl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad(); crit(model(a, b), y).backward(); opt.step()
        sched.step()
    # pooled AUC over the held-out val set
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for a, b, y in DataLoader(PairDataset(val_df), batch_size=32):
            P.append(torch.sigmoid(model(a.to(device), b.to(device))).cpu()); Y.append(y)
    P = torch.cat(P); Y = torch.cat(Y)
    return {ax: _auc(P[:, j], Y[:, j]) for j, ax in enumerate(AXES)}


def run(manifest="dataset.csv", epochs=25, n_val=3, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(manifest)
    songs = sorted(df["track"].unique())
    rng = random.Random(seed); rng.shuffle(songs)
    val_songs = songs[:n_val]
    train_pool = songs[n_val:]
    val_df = df[df["track"].isin(val_songs)].reset_index(drop=True)
    print(f"{device.type} | fixed val songs={val_songs} ({len(val_df)} pairs) | train pool={len(train_pool)} songs")

    sizes = list(range(2, len(train_pool) + 1, 2))
    if train_pool and sizes[-1] != len(train_pool):
        sizes.append(len(train_pool))
    print(f"\n{'#songs':>7s} {'#pairs':>7s} " + " ".join(f"{ax[:5]:>6s}" for ax in AXES) + f" {'mean':>6s}")
    results = []
    for n in sizes:
        sub = train_pool[:n]
        tr = df[df["track"].isin(sub)].reset_index(drop=True)
        auc = train_eval(tr, val_df, device, epochs)
        mean = float(np.nanmean([auc[ax] for ax in AXES]))
        results.append((n, mean, auc))
        print(f"{n:7d} {len(tr):7d} " + " ".join(f"{auc[ax]:6.2f}" for ax in AXES) + f" {mean:6.2f}")

    print("\n=== TREND (unseen AUC vs #training songs) ===")
    print("mean:", " -> ".join(f"{n}s:{m:.2f}" for n, m, _ in results))
    for ax in AXES:
        vals = [a[ax] for _, _, a in results]
        arrow = "RISING ↑" if vals[-1] - vals[0] > 0.05 else ("flat ~" if abs(vals[-1]-vals[0]) <= 0.05 else "FALLING ↓")
        print(f"  {ax:11s} " + " -> ".join(f"{v:.2f}" for v in vals) + f"   {arrow}")
    print("\nRising curve => more data helps => the Slakh2100 run is expected to work.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--manifest", default="dataset.csv")
    a = ap.parse_args()
    run(a.manifest, a.epochs)
