"""
scaling_curve.py
----------------
The "data closes the gap" proof, run with the FINAL (WIN) architecture. Holds out a
FIXED validation set of songs, trains on 2, 4, 6, ... of the remaining songs, and
reports UNSEEN balanced accuracy (calibrated thresholds) + AUC at each size.
A rising curve = generalisation improves with song count = the remaining per-axis
gaps are data-limited and the scale run is justified.
"""
import os, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import (CompatModel, PairDataset, AXES, recalibrate_bn,
                         predict_probs, best_thresholds, metrics)


def train_eval(train_df, val_df, device, epochs, lr=3e-4, batch=32):
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
    cl = DataLoader(PairDataset(train_df), batch_size=batch, shuffle=False, num_workers=4)
    recalibrate_bn(model, cl, device)
    Ptr, Ytr = predict_probs(model, cl, device)
    thr = best_thresholds(Ptr, Ytr)
    Pva, Yva = predict_probs(model, DataLoader(PairDataset(val_df), batch_size=batch, num_workers=4), device)
    return metrics(Pva, Yva, thr)          # (auc, bal-acc) dicts on UNSEEN


def run(manifest="dataset.csv", epochs=15, n_val=3, seed=42, sizes=None, n_seeds=1):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(manifest)
    songs = sorted(df["track"].unique())
    rng = random.Random(seed); rng.shuffle(songs)
    # scale n_val with corpus size (>=3, ~15% of songs) so the fixed val set is meaningful
    n_val = max(n_val, int(0.15 * len(songs)))
    val_songs = songs[:n_val]
    pool = songs[n_val:]
    val_df = df[df["track"].isin(val_songs)].reset_index(drop=True)
    print(f"{device.type} | fixed val={len(val_songs)} songs ({len(val_df)} pairs) | pool={len(pool)} songs")

    if sizes is None:
        sizes = list(range(2, len(pool), 2)) + [len(pool)]
    sizes = sorted(set(min(s, len(pool)) for s in sizes))
    print(f"\n{'#songs':>7s} {'#pairs':>7s} {'seed':>5s} | bal-acc: "
          + " ".join(f"{ax[:5]:>6s}" for ax in AXES) + f" {'MEAN':>6s}")
    per_size = {n: [] for n in sizes}                     # list of bal dicts per seed
    for n in sizes:
        tr = df[df["track"].isin(pool[:n])].reset_index(drop=True)
        for s in range(n_seeds):
            torch.manual_seed(1000 + s); random.seed(1000 + s); np.random.seed(1000 + s)
            auc, bal = train_eval(tr, val_df, device, epochs)
            per_size[n].append(bal)
            mb = float(np.nanmean([bal[ax] for ax in AXES]))
            print(f"{n:7d} {len(tr):7d} {s:5d} |          "
                  + " ".join(f"{bal[ax]:6.2f}" for ax in AXES) + f" {mb:6.2f}")

    mean = {ax: [float(np.mean([b[ax] for b in per_size[n]])) for n in sizes] for ax in AXES}
    std = {ax: [float(np.std([b[ax] for b in per_size[n]])) for n in sizes] for ax in AXES}
    print("\n=== TREND: unseen balanced accuracy vs #training songs (mean over seeds) ===")
    for ax in AXES:
        vals = mean[ax]
        d = vals[-1] - vals[0]
        tag = "RISING ↑" if d > 0.04 else ("flat ~" if abs(d) <= 0.04 else "FALLING ↓")
        print(f"  {ax:11s} " + " -> ".join(f"{v:.2f}" for v in vals) + f"   {tag}")
    mm = [float(np.mean([mean[ax][i] for ax in AXES])) for i in range(len(sizes))]
    print(f"  {'MEAN':11s} " + " -> ".join(f"{m:.2f}" for m in mm))
    print("\nRising = the per-axis gaps are DATA-limited -> scale run justified.")

    # evidence artifacts: raw numbers + error-bar figure
    import plots
    os.makedirs("runs", exist_ok=True)
    recs = [{"songs": n, "seed": s, **per_size[n][s]} for n in sizes for s in range(len(per_size[n]))]
    pd.DataFrame(recs).to_csv("runs/scaling_log.csv", index=False)
    print("graph ->", plots.scaling_curves(sizes, mean, std))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--manifest", default="dataset.csv")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated training-song counts, e.g. '10,25,50,85' (default: 2,4,6,...)")
    ap.add_argument("--seeds", type=int, default=1, help="training runs per size (mean +/- std)")
    a = ap.parse_args()
    sz = [int(x) for x in a.sizes.split(",")] if a.sizes else None
    run(a.manifest, a.epochs, sizes=sz, n_seeds=a.seeds)
