"""
train_model.py
--------------
Two-tier per-axis compatibility model.

  TIER 1 (foundational gates, binary, BCE):
        harmony: key, vertical, pitch_jitter
        rhythm : tempo, timing, jitter
  TIER 2 (quality of match, continuous [0,1], regression/MSE):
        register  (frequency-band masking; 1=muddy/masked, 0=clean separation)

Both tiers share ONE encoder and train jointly in a single backward pass
(multi-task): loss = BCE(gates) + LAMBDA_REG * MSE(quality).

  * 80/20 train/val split BY TRACK (no leakage)
  * reports Tier-1 clash-recall AND Tier-2 register MAE, SEEN vs UNSEEN
  * cosine LR decay + save-best (on mean unseen clash-recall) -> no noisy last-epoch snapshot

Run:  ./.venv/bin/python train_model.py --epochs 60
"""

import os
import argparse
import random
import math

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

AXES = ["key", "vertical", "pitch_jitter", "tempo", "timing", "jitter"]   # Tier-1 binary gates
QUALITY = ["register"]                                                    # Tier-2 continuous
TARGET_T = 350
LAMBDA_REG = 10.0        # weight on the Tier-2 MSE so it matters next to the 6 BCE gates


# ------------------------------------------------------------------ split
def split_by_track(manifest, val_frac=0.2, seed=42):
    df = pd.read_csv(manifest)
    tracks = sorted(df["track"].unique())
    rng = random.Random(seed)
    rng.shuffle(tracks)
    n_val = max(1, round(len(tracks) * val_frac))
    val_tracks = set(tracks[:n_val])
    train_df = df[~df["track"].isin(val_tracks)].reset_index(drop=True)
    val_df = df[df["track"].isin(val_tracks)].reset_index(drop=True)
    return train_df, val_df, sorted(val_tracks)


# ------------------------------------------------------------------ data
def load_img(path):
    t = torch.load(path, weights_only=True)["image_tensor"]     # [3, 84, T]
    T = t.shape[2]
    if T < TARGET_T:
        t = F.pad(t, (0, TARGET_T - T))
    elif T > TARGET_T:
        t = t[:, :, :TARGET_T]
    return torch.nan_to_num(t, nan=0.0)


class PairDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = load_img(r["file_A"])
        b = load_img(r["file_B"])
        y = torch.tensor([float(r[ax]) for ax in AXES], dtype=torch.float32)
        yq = torch.tensor([float(r[q]) for q in QUALITY], dtype=torch.float32)
        return a, b, y, yq


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    """Shared CNN. Pools FREQUENCY, keeps TIME. Returns (global [B,C], seq [B,C,T'])."""
    def __init__(self, cin=3, C=128):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d((2, 1)),          # halve frequency, keep time
            )
        self.net = nn.Sequential(blk(cin, 32), blk(32, 64), blk(64, C), blk(C, C))

    def forward(self, x):            # x [B,3,84,T]
        f = self.net(x)              # [B,C,~5,T]
        f = f.mean(dim=2)            # pool remaining frequency -> [B,C,T]
        g = f.mean(dim=2)            # global (pool time) -> [B,C]
        return g, f


def sym(a, b):
    """Symmetric (order-invariant) combine, so score(A,B) == score(B,A)."""
    return torch.cat([a + b, (a - b).abs(), a * b], dim=1)


class CompatModel(nn.Module):
    def __init__(self, C=128):
        super().__init__()
        self.enc = Encoder(C=C)
        # --- global path -> harmony gates (key, vertical, pitch_jitter) ---
        self.gmlp = nn.Sequential(nn.Linear(3 * C, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.ReLU())
        self.harm = nn.Linear(128, 3)
        # --- temporal path -> rhythm gates (tempo, timing, jitter) ---
        self.tstream = nn.Sequential(nn.Conv1d(C, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                                     nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        self.tconv = nn.Sequential(nn.Conv1d(3 * 128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rhy = nn.Linear(128 * 3, 3)
        # --- TIER 2: quality regression (register) off the global summary ---
        self.qhead = nn.Sequential(nn.Linear(3 * C, 128), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(128, len(QUALITY)))

    def forward(self, a, b):
        ga, sa = self.enc(a)
        gb, sb = self.enc(b)
        g = sym(ga, gb)
        h = self.harm(self.gmlp(g))                     # [B,3] harmony logits
        ta, tb = self.tstream(sa), self.tstream(sb)
        tc = self.tconv(sym(ta, tb))
        pooled = torch.cat([tc.mean(dim=2), tc.std(dim=2), tc.amax(dim=2)], dim=1)
        r = self.rhy(pooled)                            # [B,3] rhythm logits
        q = self.qhead(g)                               # [B,1] register logit (Tier-2)
        return torch.cat([h, r], dim=1), q              # (gates [B,6], quality [B,1])


# ------------------------------------------------------------------ eval
def evaluate(model, loader, device):
    model.eval()
    clash = {ax: [0, 0] for ax in AXES}    # per-axis clash-recall [correct, total] on label==0
    reg_abs, reg_n = 0.0, 0                # Tier-2 register MAE
    with torch.no_grad():
        for a, b, y, yq in loader:
            gates, q = model(a.to(device), b.to(device))
            pred = (torch.sigmoid(gates).cpu() > 0.5).float()
            for j, ax in enumerate(AXES):
                yj, pj = y[:, j], pred[:, j]
                neg = yj == 0
                clash[ax][0] += ((pj == yj) & neg).sum().item(); clash[ax][1] += int(neg.sum())
            qp = torch.sigmoid(q).cpu()
            reg_abs += (qp - yq).abs().sum().item(); reg_n += yq.numel()
    rec = {ax: clash[ax][0] / max(1, clash[ax][1]) for ax in AXES}
    mae = reg_abs / max(1, reg_n)
    return rec, mae


# ------------------------------------------------------------------ train
def train(manifest="dataset.csv", epochs=60, batch=32, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    train_df, val_df, val_tracks = split_by_track(manifest)
    print(f"Device: {device.type} | train pairs: {len(train_df)} | val pairs: {len(val_df)} "
          f"| val tracks: {val_tracks}")

    # trivial baseline: always predict the TRAIN mean register -> the MAE the model must beat
    reg_mean = float(train_df["register"].mean())
    base_seen = float((train_df["register"] - reg_mean).abs().mean())
    base_unseen = float((val_df["register"] - reg_mean).abs().mean())
    print(f"T2 register baseline (predict-mean {reg_mean:.3f}) MAE: "
          f"SEEN {base_seen:.3f}  UNSEEN {base_unseen:.3f}  <- model must beat these")

    tl = DataLoader(PairDataset(train_df), batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(PairDataset(val_df), batch_size=batch, shuffle=False, num_workers=2)

    model = CompatModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # clash (label 0) is ~12.5% of the gate data -> down-weight the majority "compatible" term
    pos_weight = torch.full((6,), 117.0 / 819.0, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    best_score, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for a, b, y, yq in tl:
            a, b, y, yq = a.to(device), b.to(device), y.to(device), yq.to(device)
            opt.zero_grad()
            gates, q = model(a, b)
            loss = bce(gates, y) + LAMBDA_REG * mse(torch.sigmoid(q), yq)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        tr_rec, tr_mae = evaluate(model, tl, device)          # SEEN
        va_rec, va_mae = evaluate(model, vl, device)          # UNSEEN
        tr = " ".join(f"{ax[:4]}={tr_rec[ax]:.2f}" for ax in AXES)
        va = " ".join(f"{ax[:4]}={va_rec[ax]:.2f}" for ax in AXES)
        mean_unseen = float(np.mean([va_rec[ax] for ax in AXES]))
        print(f"epoch {ep}/{epochs}  loss={tot/len(tl):.4f}  lr={sched.get_last_lr()[0]:.2e}")
        print(f"   T1 clash-recall SEEN : {tr}")
        print(f"   T1 clash-recall UNSEEN: {va}   (mean {mean_unseen:.2f})")
        print(f"   T2 register MAE      : SEEN {tr_mae:.3f}  UNSEEN {va_mae:.3f}   "
              f"(baseline {base_unseen:.3f}; lower=better)")

        # save-best on mean unseen clash-recall (penalise register error a little)
        score = mean_unseen - va_mae
        if score > best_score:
            best_score, best_state = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs("models", exist_ok=True)
    torch.save(best_state or model.state_dict(), "models/compat_model.pth")
    print(f"saved BEST -> models/compat_model.pth  (score={best_score:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    train(a.manifest, a.epochs, a.batch, a.lr)
