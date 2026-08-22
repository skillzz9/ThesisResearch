"""
train_model.py
--------------
Four-gate relational compatibility model (simplified scope).

  HARMONY (global path):  key, vertical
  RHYTHM  (temporal path): tempo, timing

Binary classification only — one BCE loss over the 4 gates. (The register regression
head is deferred to Phase 2, isolated; the quality axes pitch_jitter/jitter were dropped.)

  * 80/20 train/val split BY TRACK (no leakage)
  * pos_weight computed from the data (counters the compatible/clash imbalance)
  * reports per-axis clash-recall, SEEN vs UNSEEN, every epoch
  * cosine LR decay + save-best (on mean unseen clash-recall)

Run:  ./.venv/bin/python train_model.py --epochs 40
"""

import os
import argparse
import random

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

AXES = ["key", "vertical", "tempo", "timing"]   # 2 harmony + 2 rhythm
TARGET_T = 350


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
        return a, b, y


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
        # global path -> harmony gates: key, vertical
        self.gmlp = nn.Sequential(nn.Linear(3 * C, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.ReLU())
        self.harm = nn.Linear(128, 2)
        # temporal path -> rhythm gates: tempo, timing
        self.tstream = nn.Sequential(nn.Conv1d(C, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                                     nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        self.tconv = nn.Sequential(nn.Conv1d(3 * 128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rhy = nn.Linear(128 * 3, 2)

    def forward(self, a, b):
        ga, sa = self.enc(a)
        gb, sb = self.enc(b)
        h = self.harm(self.gmlp(sym(ga, gb)))           # [B,2] harmony logits
        ta, tb = self.tstream(sa), self.tstream(sb)
        tc = self.tconv(sym(ta, tb))
        pooled = torch.cat([tc.mean(dim=2), tc.std(dim=2), tc.amax(dim=2)], dim=1)
        r = self.rhy(pooled)                            # [B,2] rhythm logits
        return torch.cat([h, r], dim=1)                 # [B,4] logits, order = AXES


# ------------------------------------------------------------------ eval
def evaluate(model, loader, device):
    model.eval()
    clash = {ax: [0, 0] for ax in AXES}    # per-axis clash-recall [correct, total] on label==0
    with torch.no_grad():
        for a, b, y in loader:
            pred = (torch.sigmoid(model(a.to(device), b.to(device))).cpu() > 0.5).float()
            for j, ax in enumerate(AXES):
                yj, pj = y[:, j], pred[:, j]
                neg = yj == 0
                clash[ax][0] += ((pj == yj) & neg).sum().item(); clash[ax][1] += int(neg.sum())
    return {ax: clash[ax][0] / max(1, clash[ax][1]) for ax in AXES}


# ------------------------------------------------------------------ train
def train(manifest="dataset.csv", epochs=40, batch=32, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    train_df, val_df, val_tracks = split_by_track(manifest)
    print(f"Device: {device.type} | train pairs: {len(train_df)} | val pairs: {len(val_df)} "
          f"| val tracks: {val_tracks}")

    tl = DataLoader(PairDataset(train_df), batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(PairDataset(val_df), batch_size=batch, shuffle=False, num_workers=2)

    model = CompatModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # BCE pos_weight scales the label==1 ("compatible", majority) term. Set it to
    # clash/compatible (<1) so the rare clash class gets relatively more gradient.
    pw = [max(1, int((train_df[ax] == 0).sum())) / max(1, int((train_df[ax] == 1).sum())) for ax in AXES]
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    print("pos_weight per axis:", {ax: round(w, 3) for ax, w in zip(AXES, pw)})
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for a, b, y in tl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(a, b), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        tr = evaluate(model, tl, device)       # SEEN
        va = evaluate(model, vl, device)       # UNSEEN
        mean_unseen = float(np.mean([va[ax] for ax in AXES]))
        seen = " ".join(f"{ax[:4]}={tr[ax]:.2f}" for ax in AXES)
        unseen = " ".join(f"{ax[:4]}={va[ax]:.2f}" for ax in AXES)
        print(f"epoch {ep}/{epochs}  loss={tot/len(tl):.4f}  lr={sched.get_last_lr()[0]:.2e}")
        print(f"   clash-recall SEEN  : {seen}")
        print(f"   clash-recall UNSEEN: {unseen}   (mean {mean_unseen:.2f})")

        if mean_unseen > best_score:
            best_score, best_state = mean_unseen, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs("models", exist_ok=True)
    torch.save(best_state or model.state_dict(), "models/compat_model.pth")
    print(f"saved BEST -> models/compat_model.pth  (mean unseen clash-recall={best_score:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    train(a.manifest, a.epochs, a.batch, a.lr)
