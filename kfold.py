"""
kfold.py
--------
The honest generalisation test. Each fold holds out a different group of songs, trains
from scratch, and predicts the held-out songs. We POOL every held-out prediction across
all folds, then compute per-axis AUC + balanced accuracy over the WHOLE dataset's worth
of examples -> trustworthy numbers, not a 5-example coin flip. Also tells us whether a
single-run result (e.g. key=0.84) was real or lucky.
"""
import os, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import CompatModel, PairDataset, AXES, _auc


def folds_by_track(df, k, seed=42):
    tracks = sorted(df["track"].unique())
    random.Random(seed).shuffle(tracks)
    return [tracks[i::k] for i in range(k)]


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
    P, Y = [], []
    with torch.no_grad():
        for a, b, y in DataLoader(PairDataset(df), batch_size=32):
            P.append(torch.sigmoid(model(a.to(device), b.to(device))).cpu()); Y.append(y)
    return torch.cat(P), torch.cat(Y)


def run(manifest="dataset.csv", k=5, epochs=25, lr=3e-4, batch=32):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(manifest)
    groups = folds_by_track(df, k)
    print(f"{k}-fold by song on {device.type} | {len(df)} pairs | fold sizes: {[len(g) for g in groups]}")

    all_P, all_Y = [], []
    for fi, val_tracks in enumerate(groups, 1):
        tr = df[~df["track"].isin(val_tracks)].reset_index(drop=True)
        va = df[df["track"].isin(val_tracks)].reset_index(drop=True)
        if len(va) == 0:
            continue
        P, Y = predict(train_one(tr, device, epochs, lr, batch), va, device)
        all_P.append(P); all_Y.append(Y)
        aucs = {ax: _auc(P[:, j], Y[:, j]) for j, ax in enumerate(AXES)}
        print(f"  fold {fi}/{k} val={val_tracks} n={len(va)}  AUC " + " ".join(f"{ax[:4]}={aucs[ax]:.2f}" for ax in AXES))

    P = torch.cat(all_P); Y = torch.cat(all_Y)
    print("\n" + "=" * 66)
    print(" POOLED HELD-OUT (every pair scored by a model that never saw its song)")
    print("=" * 66)
    print(f"{'axis':12s} {'AUC':>6s} {'bal-acc':>8s} {'clash-rec':>10s} {'#clash':>7s} {'#pairs':>7s}")
    for j, ax in enumerate(AXES):
        p, y = P[:, j], Y[:, j]
        auc = _auc(p, y)
        pred = (p > 0.5).float(); neg = y == 0; pos = y == 1
        rc = ((pred == y) & neg).sum().item() / max(1, int(neg.sum()))
        rk = ((pred == y) & pos).sum().item() / max(1, int(pos.sum()))
        print(f"{ax:12s} {auc:6.2f} {0.5*(rc+rk):8.2f} {rc:10.2f} {int(neg.sum()):7d} {len(y):7d}")
    print("=" * 66)
    print(f"MEAN pooled AUC: {np.nanmean([_auc(P[:,j],Y[:,j]) for j in range(len(AXES))]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--manifest", default="dataset.csv")
    a = ap.parse_args()
    run(a.manifest, a.k, a.epochs)
