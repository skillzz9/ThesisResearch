"""
kfold.py
--------
The honest generalisation test = THE PoC criteria measurement. Each fold holds out a
group of songs, trains from scratch, recalibrates BN stats, picks per-axis decision
thresholds ON ITS TRAINING FOLD, then predicts the held-out songs. All held-out
predictions are pooled -> per-axis AUC + calibrated balanced accuracy over the whole
dataset's worth of examples.

PoC criteria this measures:  C1 pooled AUC >= 0.70/axis   C2 bal-acc >= 0.60/axis
"""
import os, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import (CompatModel, PairDataset, AXES, _auc,
                         recalibrate_bn, predict_probs, best_thresholds, metrics)


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
    dl = DataLoader(PairDataset(train_df, train=True), batch_size=batch, shuffle=True, num_workers=4)
    for _ in range(epochs):
        model.train()
        for a, b, y in dl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad(); crit(model(a, b), y).backward(); opt.step()
        sched.step()
    return model


def run(manifest="dataset.csv", k=5, epochs=15, lr=3e-4, batch=32):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(manifest)
    groups = folds_by_track(df, k)
    print(f"{k}-fold by song on {device.type} | {len(df)} pairs | fold sizes: {[len(g) for g in groups]}")

    all_P, all_Y, all_pred = [], [], []
    for fi, val_tracks in enumerate(groups, 1):
        tr = df[~df["track"].isin(val_tracks)].reset_index(drop=True)
        va = df[df["track"].isin(val_tracks)].reset_index(drop=True)
        if len(va) == 0:
            continue
        model = train_one(tr, device, epochs, lr, batch)
        cl = DataLoader(PairDataset(tr), batch_size=batch, shuffle=False, num_workers=4)
        recalibrate_bn(model, cl, device)
        Ptr, Ytr = predict_probs(model, cl, device)
        thr = best_thresholds(Ptr, Ytr)                       # thresholds from TRAIN fold only
        P, Y = predict_probs(model, DataLoader(PairDataset(va), batch_size=batch, num_workers=4), device)
        pred = torch.stack([(P[:, j] > thr[ax]).float() for j, ax in enumerate(AXES)], dim=1)
        all_P.append(P); all_Y.append(Y); all_pred.append(pred)
        aucs, bals = metrics(P, Y, thr)
        print(f"  fold {fi}/{k} val={val_tracks} n={len(va)}"
              f"  AUC " + " ".join(f"{ax[:4]}={aucs[ax]:.2f}" for ax in AXES)
              + "  bal " + " ".join(f"{ax[:4]}={bals[ax]:.2f}" for ax in AXES))

    P = torch.cat(all_P); Y = torch.cat(all_Y); PR = torch.cat(all_pred)
    print("\n" + "=" * 70)
    print(" POOLED HELD-OUT (every pair scored by a model that never saw its song)")
    print("=" * 70)
    print(f"{'axis':12s} {'AUC':>6s} {'bal-acc':>8s} {'clash-rec':>10s} {'compat-rec':>11s} {'#clash':>7s}")
    means = {"auc": [], "bal": []}
    for j, ax in enumerate(AXES):
        p, y, pr = P[:, j], Y[:, j], PR[:, j]
        auc = _auc(p, y)
        neg, pos = y == 0, y == 1
        rc = ((pr == y) & neg).sum().item() / max(1, int(neg.sum()))
        rk = ((pr == y) & pos).sum().item() / max(1, int(pos.sum()))
        bal = 0.5 * (rc + rk)
        means["auc"].append(auc); means["bal"].append(bal)
        print(f"{ax:12s} {auc:6.2f} {bal:8.2f} {rc:10.2f} {rk:11.2f} {int(neg.sum()):7d}")
    print("=" * 70)
    print(f"MEAN pooled: AUC={np.nanmean(means['auc']):.3f}  bal-acc={np.nanmean(means['bal']):.3f}")
    print("CRITERIA: C1 AUC>=0.70/axis, mean>=0.75 | C2 bal-acc>=0.60/axis, mean>=0.65")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--manifest", default="dataset.csv")
    a = ap.parse_args()
    run(a.manifest, a.k, a.epochs)
