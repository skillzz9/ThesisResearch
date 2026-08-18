"""Side-by-side SEEN (train) vs UNSEEN (val) per-axis metrics for RUN 1 (old architecture)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import Encoder, sym, PairDataset, split_by_track, AXES


class OldCompatModel(nn.Module):
    """The architecture as it was for run 1: temporal path collapsed with mean(dim=2)."""
    def __init__(self, C=128):
        super().__init__()
        self.enc = Encoder(C=C)
        self.gmlp = nn.Sequential(nn.Linear(3 * C, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.ReLU())
        self.harm = nn.Linear(128, 3)
        self.tconv = nn.Sequential(nn.Conv1d(3 * C, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rhy = nn.Linear(128, 3)

    def forward(self, a, b):
        ga, sa = self.enc(a)
        gb, sb = self.enc(b)
        h = self.harm(self.gmlp(sym(ga, gb)))
        t = self.tconv(sym(sa, sb)).mean(dim=2)
        r = self.rhy(t)
        return torch.cat([h, r], dim=1)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = OldCompatModel().to(device)
model.load_state_dict(torch.load("models/compat_model_run1.pth", map_location=device))
model.eval()

train_df, val_df, val_tracks = split_by_track("dataset.csv")

def metrics(df):
    dl = DataLoader(PairDataset(df), batch_size=32)
    acc = {ax: [0, 0] for ax in AXES}
    rec = {ax: [0, 0] for ax in AXES}
    with torch.no_grad():
        for a, b, y in dl:
            pred = (torch.sigmoid(model(a.to(device), b.to(device))).cpu() > 0.5).float()
            for j, ax in enumerate(AXES):
                yj, pj = y[:, j], pred[:, j]
                acc[ax][0] += (pj == yj).sum().item(); acc[ax][1] += len(yj)
                neg = yj == 0
                rec[ax][0] += ((pj == yj) & neg).sum().item(); rec[ax][1] += int(neg.sum())
    A = {ax: acc[ax][0]/max(1, acc[ax][1]) for ax in AXES}
    R = {ax: rec[ax][0]/max(1, rec[ax][1]) for ax in AXES}
    return A, R

trA, trR = metrics(train_df)
vaA, vaR = metrics(val_df)

print(f"RUN 1 (old architecture) | val tracks held out: {val_tracks}\n")
print("ACCURACY  (note: 0.875 = the trivial 'always compatible' baseline)")
print(f"{'axis':13s} {'SEEN':>7s} {'UNSEEN':>7s} {'gap':>7s}")
for ax in AXES:
    print(f"{ax:13s} {trA[ax]:7.2f} {vaA[ax]:7.2f} {trA[ax]-vaA[ax]:+7.2f}")

print("\nCLASH-RECALL  (the real metric: caught clashes / actual clashes)")
print(f"{'axis':13s} {'SEEN':>7s} {'UNSEEN':>7s} {'gap':>7s}")
for ax in AXES:
    print(f"{ax:13s} {trR[ax]:7.2f} {vaR[ax]:7.2f} {trR[ax]-vaR[ax]:+7.2f}")
