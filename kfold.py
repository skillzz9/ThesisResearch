"""
kfold.py
--------
The honest generalisation test. Instead of judging on one tiny 2-song validation
split (where clash-recall flips 0<->1 every epoch), we do K folds: each fold holds
out a different group of songs, trains from scratch, predicts the held-out songs,
and we POOL every held-out prediction across all folds. Then per-axis clash-recall
is computed over the WHOLE dataset's worth of held-out examples -> a real number,
not a 5-example coin flip.

Every pair is evaluated exactly once, on a model that never saw its song.
"""
import os, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import CompatModel, PairDataset, AXES


def folds_by_track(df, k, seed=42):
    tracks = sorted(df["track"].unique())
    random.Random(seed).shuffle(tracks)
    return [tracks[i::k] for i in range(k)]   # round-robin split of songs into k groups


def train_one(train_df, device, epochs, lr, batch):
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
    return model


def predict(model, df, device):
    model.eval()
    dl = DataLoader(PairDataset(df), batch_size=32)
    P, Y = [], []
    with torch.no_grad():
        for a, b, y in dl:
            P.append((torch.sigmoid(model(a.to(device), b.to(device))).cpu() > 0.5).float())
            Y.append(y)
    return torch.cat(P), torch.cat(Y)


def run(manifest="dataset.csv", k=5, epochs=30, lr=3e-4, batch=32):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(manifest)
    groups = folds_by_track(df, k)
    print(f"{k}-fold by song on {device.type} | {len(df)} pairs | "
          f"fold sizes (songs): {[len(g) for g in groups]}")

    all_P, all_Y = [], []
    for fi, val_tracks in enumerate(groups, 1):
        tr = df[~df["track"].isin(val_tracks)].reset_index(drop=True)
        va = df[df["track"].isin(val_tracks)].reset_index(drop=True)
        if len(va) == 0:
            continue
        model = train_one(tr, device, epochs, lr, batch)
        P, Y = predict(model, va, device)
        all_P.append(P); all_Y.append(Y)
        # per-fold clash-recall (for visibility)
        rec = {}
        for j, ax in enumerate(AXES):
            neg = Y[:, j] == 0
            rec[ax] = ((P[:, j] == Y[:, j]) & neg).sum().item() / max(1, int(neg.sum()))
        print(f"  fold {fi}/{k}  val songs={val_tracks}  n={len(va)}  "
              + " ".join(f"{ax[:4]}={rec[ax]:.2f}" for ax in AXES))

    P = torch.cat(all_P); Y = torch.cat(all_Y)
    print("\n" + "=" * 60)
    print(" POOLED HELD-OUT RESULT (every pair scored by a model that never saw its song)")
    print("=" * 60)
    print(f"{'axis':12s} {'clash-recall':>12s} {'#clash':>7s} {'precision':>10s} {'#pairs':>7s}")
    for j, ax in enumerate(AXES):
        yj, pj = Y[:, j], P[:, j]
        neg = yj == 0
        recall = ((pj == yj) & neg).sum().item() / max(1, int(neg.sum()))
        flagged = pj == 0
        prec = ((pj == yj) & neg).sum().item() / max(1, int(flagged.sum()))
        print(f"{ax:12s} {recall:12.2f} {int(neg.sum()):7d} {prec:10.2f} {len(yj):7d}")
    print("=" * 60)
    print("clash-recall = of the truly-clashing held-out pairs, fraction caught.")
    print("This is the honest generalisation number (pooled over all songs).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--manifest", default="dataset.csv")
    a = ap.parse_args()
    run(a.manifest, a.k, a.epochs)
