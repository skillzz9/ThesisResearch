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
    def __init__(self, df, train=False, crop_t=256):
        self.df = df.reset_index(drop=True)
        self.train = train          # train mode -> random crop + SpecAugment
        self.crop_t = crop_t

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _specaug(t):
        """Light SpecAugment: one random frequency mask + one random time mask.
        Makes every epoch's view of a pair different -> fights image memorisation."""
        t = t.clone()
        if random.random() < 0.5:
            f0 = random.randint(0, t.shape[1] - 9)
            t[:, f0:f0 + random.randint(2, 8), :] = 0.0
        if random.random() < 0.5:
            T = t.shape[2]
            t0 = random.randint(0, max(0, T - 17))
            t[:, :, t0:t0 + random.randint(4, 16)] = 0.0
        return t

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = load_img(r["file_A"])
        b = load_img(r["file_B"])
        if self.train:
            # SAME random time-crop for both loops -- preserves their relative
            # alignment (a shared shift is label-preserving, like phase augmentation)
            T = a.shape[2]
            if T > self.crop_t:
                s = random.randint(0, T - self.crop_t)
                a = a[:, :, s:s + self.crop_t]
                b = b[:, :, s:s + self.crop_t]
            a = self._specaug(a)
            b = self._specaug(b)
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
        f2d = self.net(x)            # [B,C,~5,T]  -- keep the coarse frequency axis
        seq = f2d.mean(dim=2)        # pool frequency -> [B,C,T]  (rhythm + global)
        g = seq.mean(dim=2)          # pool time -> [B,C]  (global summary, for key)
        return g, seq, f2d


def sym(a, b):
    """Symmetric (order-invariant) combine, so score(A,B) == score(B,A)."""
    return torch.cat([a + b, (a - b).abs(), a * b], dim=1)


class CompatModel(nn.Module):
    def __init__(self, C=128):
        super().__init__()
        self.enc = Encoder(C=C)
        # KEY -> dedicated PITCH-CLASS branch, fully ISOLATED from the shared encoder.
        # Why: the encoder's MaxPool collapses 84 freq bins -> 5 bands, erasing the
        # 1-semitone shift a key clash is (and CNN translation-invariance suppresses it).
        # This branch reads the RAW input at full 84-bin resolution: two convs learn
        # harmonic cleanup (energy at f,2f,3f = one note), then the 7 octaves fold onto
        # 12 pitch classes where a semitone shift is a plain ROTATION. A vs B alignment
        # is read over all 12 rotations (same key -> peak at rotation 0). No weights are
        # shared with other axes, so key's gradients cannot disturb them.
        self.kconv = nn.Sequential(
            nn.Conv2d(3, 16, (7, 5), padding=(3, 2)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, (7, 5), padding=(3, 2)), nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.key_head = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
        # CONSONANCE -> freq-preserving, TIME-ALIGNED A-vs-B comparison. Compares the two
        # loops frame-by-frame WITH the coarse frequency axis intact, so it can see the
        # per-moment vertical intervals (which the time-averaged summary throws away).
        self.cconv = nn.Sequential(nn.Conv2d(3 * C, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.cseq = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.cons_head = nn.Linear(64 * 3, 1)
        # RHYTHM -> temporal path: tempo, timing
        self.tstream = nn.Sequential(nn.Conv1d(C, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                                     nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        self.tconv = nn.Sequential(nn.Conv1d(3 * 128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        # order-preserving reader: a GRU tracks the drift SHAPE over time (tempo needs
        # this; mean/std/max pooling alone destroys it). Concatenated with the pooling so
        # timing keeps its working summary-stat signal.
        self.rhy_gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.rhy = nn.Linear(128 * 3 + 128, 2)

    def _pc_profile(self, x):
        """[B,3,84,T] -> [B,16,12] learned pitch-class profile: harmonic-cleanup convs at
        full frequency resolution, pool time, fold 7 octaves onto 12 pitch classes,
        L2-normalise per channel (removes loudness)."""
        f = self.kconv(x)                          # [B,16,84,T] -- full freq resolution kept
        f = f.mean(dim=3)                          # pool time (key is time-invariant) -> [B,16,84]
        f = f.view(f.shape[0], f.shape[1], 7, 12).sum(dim=2)   # fold octaves -> [B,16,12]
        return f / (f.norm(dim=2, keepdim=True) + 1e-6)

    def _key_alignment(self, a, b):
        """Symmetric 12-dim rotation-alignment spectrum of the two pitch-class profiles.
        Same key -> alignment peaks at rotation 0; a transposed loop moves the peak."""
        pa, pb = self._pc_profile(a), self._pc_profile(b)
        xc = torch.stack([(pa * torch.roll(pb, r, dims=2)).sum(dim=(1, 2))
                          for r in range(12)], dim=1)          # [B,12]
        idx = [(-r) % 12 for r in range(12)]
        return 0.5 * (xc + xc[:, idx])             # symmetric under A<->B swap

    def forward(self, a, b):
        ga, sa, fa = self.enc(a)
        gb, sb, fb = self.enc(b)
        # key -- isolated pitch-class rotation alignment (sole input to the key head)
        key = self.key_head(self._key_alignment(a, b))              # [B,1]
        # consonance -- per-frame freq-preserving comparison, then pool over time
        cc = self.cconv(sym(fa, fb)).mean(dim=2)                    # [B,64,T]  (pool freq)
        cc = self.cseq(cc)                                          # [B,64,T]
        cpool = torch.cat([cc.mean(dim=2), cc.std(dim=2), cc.amax(dim=2)], dim=1)  # [B,192]
        cons = self.cons_head(cpool)                                # [B,1]
        # rhythm -- temporal path
        ta, tb = self.tstream(sa), self.tstream(sb)
        tc = self.tconv(sym(ta, tb))                               # [B,128,T]
        rpool = torch.cat([tc.mean(dim=2), tc.std(dim=2), tc.amax(dim=2)], dim=1)  # [B,384]
        _, h = self.rhy_gru(tc.transpose(1, 2))                    # h: [2,B,64] bidir GRU over time
        rseq = torch.cat([h[0], h[1]], dim=1)                      # [B,128] drift-shape summary
        r = self.rhy(torch.cat([rpool, rseq], dim=1))              # [B,2]
        return torch.cat([key, cons, r], dim=1)                     # [B,4]: key, consonance, tempo, timing


# ------------------------------------------------------------------ eval
def _auc(scores, labels):
    """Threshold-free separation: P(a compatible pair scores higher than a clash pair).
    0.5 = no signal (coin flip), 1.0 = perfect separation. Rank-based (Mann-Whitney);
    unlike thresholded recall it does NOT flip 0<->1 on tiny sets."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recalibrate_bn(model, loader, device, n_batches=8):
    """Refresh BatchNorm running statistics with a cumulative average over training data.
    On tiny datasets BN's momentum-EMA stats drift, making eval-mode behave unlike
    train-mode (documented: key 0.67 train-mode vs 0.05 eval-mode). This fixes the stats
    without touching optimisation (unlike the GroupNorm attempt, which hurt learning)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None            # None -> cumulative moving average
    model.train()
    with torch.no_grad():
        for i, (a, b, _) in enumerate(loader):
            model(a.to(device), b.to(device))
            if i + 1 >= n_batches:
                break
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.1


def predict_probs(model, loader, device):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for a, b, y in loader:
            P.append(torch.sigmoid(model(a.to(device), b.to(device))).cpu())
            Y.append(y)
    return torch.cat(P), torch.cat(Y)


def best_thresholds(P, Y):
    """Per-axis decision threshold maximising balanced accuracy -- chosen on TRAINING
    data only, then applied unchanged to unseen data. Converts the model's separation
    (AUC) into honest accuracy; without this, 0.5 sits arbitrarily vs the pos_weighted
    output distribution and accuracy reads as chance even when separation exists."""
    thr = {}
    grid = np.linspace(0.05, 0.95, 37)
    for j, ax in enumerate(AXES):
        p, y = P[:, j], Y[:, j]
        best_t, best_b = 0.5, -1.0
        for t in grid:
            pred = (p > t).float()
            rc = ((pred == y) & (y == 0)).sum().item() / max(1, int((y == 0).sum()))
            rk = ((pred == y) & (y == 1)).sum().item() / max(1, int((y == 1).sum()))
            b = 0.5 * (rc + rk)
            if b > best_b:
                best_b, best_t = b, float(t)
        thr[ax] = best_t
    return thr


def metrics(P, Y, thresholds=None):
    """(auc, balanced-accuracy) dicts per axis; bal-acc uses calibrated thresholds."""
    auc, bal = {}, {}
    for j, ax in enumerate(AXES):
        p, y = P[:, j], Y[:, j]
        auc[ax] = _auc(p, y)
        t = (thresholds or {}).get(ax, 0.5)
        pred = (p > t).float()
        rc = ((pred == y) & (y == 0)).sum().item() / max(1, int((y == 0).sum()))
        rk = ((pred == y) & (y == 1)).sum().item() / max(1, int((y == 1).sum()))
        bal[ax] = 0.5 * (rc + rk)
    return auc, bal


# ------------------------------------------------------------------ train
def train(manifest="dataset.csv", epochs=40, batch=32, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    train_df, val_df, val_tracks = split_by_track(manifest)
    print(f"Device: {device.type} | train pairs: {len(train_df)} | val pairs: {len(val_df)} "
          f"| val tracks: {val_tracks}")

    # train loader: augmented views (random shared crop + SpecAugment)
    tl = DataLoader(PairDataset(train_df, train=True), batch_size=batch, shuffle=True, num_workers=4)
    # clean loaders for BN recalibration + evaluation (full-length, no augmentation)
    cl = DataLoader(PairDataset(train_df), batch_size=batch, shuffle=False, num_workers=4)
    vl = DataLoader(PairDataset(val_df), batch_size=batch, shuffle=False, num_workers=4)

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
    history = []                                  # per-eval-epoch metrics -> CSV + graphs
    last_tr_bal, last_va_bal = None, None
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

        if ep % 3 != 0 and ep != epochs:            # evaluate every 3rd epoch + final
            print(f"epoch {ep}/{epochs}  loss={tot/len(tl):.4f}")
            continue

        recalibrate_bn(model, cl, device)           # fix BN running stats before eval
        Ptr, Ytr = predict_probs(model, cl, device)
        Pva, Yva = predict_probs(model, vl, device)
        thr = best_thresholds(Ptr, Ytr)             # thresholds from TRAIN only
        tr_auc, tr_bal = metrics(Ptr, Ytr, thr)
        va_auc, va_bal = metrics(Pva, Yva, thr)
        mean_unseen_auc = float(np.nanmean([va_auc[ax] for ax in AXES]))
        print(f"epoch {ep}/{epochs}  loss={tot/len(tl):.4f}  lr={sched.get_last_lr()[0]:.2e}")
        print(f"   AUC     SEEN  : " + " ".join(f"{ax[:4]}={tr_auc[ax]:.2f}" for ax in AXES))
        print(f"   AUC     UNSEEN: " + " ".join(f"{ax[:4]}={va_auc[ax]:.2f}" for ax in AXES)
              + f"   (mean {mean_unseen_auc:.2f})")
        print(f"   bal-acc SEEN  : " + " ".join(f"{ax[:4]}={tr_bal[ax]:.2f}" for ax in AXES))
        print(f"   bal-acc UNSEEN: " + " ".join(f"{ax[:4]}={va_bal[ax]:.2f}" for ax in AXES)
              + "   (calibrated thresholds)")

        if mean_unseen_auc > best_score:
            best_score = mean_unseen_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        row = {"ep": ep, "loss": tot / len(tl)}
        for ax in AXES:
            row[f"{ax}_tr_acc"] = tr_bal[ax]; row[f"{ax}_va_acc"] = va_bal[ax]
            row[f"{ax}_tr_auc"] = tr_auc[ax]; row[f"{ax}_va_auc"] = va_auc[ax]
        history.append(row)
        last_tr_bal, last_va_bal = tr_bal, va_bal

    os.makedirs("models", exist_ok=True)
    torch.save(best_state or model.state_dict(), "models/compat_model.pth")
    print(f"saved BEST -> models/compat_model.pth  (mean unseen AUC={best_score:.3f})")

    # evidence artifacts: per-epoch log + graphs (epoch curves, seen-vs-unseen gap)
    import plots
    os.makedirs("runs", exist_ok=True)
    pd.DataFrame(history).to_csv("runs/train_log.csv", index=False)
    if history:
        print("graph ->", plots.epoch_curves(history))
    if last_tr_bal and last_va_bal:
        print("graph ->", plots.gap_bars(last_tr_bal, last_va_bal))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    train(a.manifest, a.epochs, a.batch, a.lr)
