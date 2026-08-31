"""
metrics_report.py
-----------------
Full per-axis metrics from a pooled k-fold run's saved predictions (runs/kfold_pooled.npz,
written by kfold.py). Detection target = CLASH (label 0), since detecting the clash is the
useful job. Reports precision, recall, F1, specificity, and MCC (Matthews correlation
coefficient -- the robust single-number metric for imbalanced classes; -1..+1, 0 = chance).

    python metrics_report.py [path/to/kfold_pooled.npz]
"""
import sys, os
import numpy as np

AXES = ["key", "vertical", "tempo", "timing"]
NICE = {"key": "key", "vertical": "consonance", "tempo": "tempo", "timing": "timing"}


def per_axis(P, Y, PR, j):
    y, pr = Y[:, j], PR[:, j]
    # positive class = CLASH = label 0
    clash, comp = (y == 0), (y == 1)
    pred_clash = (pr == 0)
    TP = int((pred_clash & clash).sum())
    FP = int((pred_clash & comp).sum())
    FN = int((~pred_clash & clash).sum())
    TN = int((~pred_clash & comp).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    spec = TN / (TN + FP) if TN + FP else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal_acc = 0.5 * (rec + spec)
    denom = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN - FP * FN) / denom) if denom else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec,
                specificity=spec, f1=f1, bal_acc=bal_acc, mcc=mcc)


def main(path):
    d = np.load(path)
    P, Y, PR = d["P"], d["Y"], d["PR"]
    print(f"pooled predictions: {P.shape[0]} pairs  (source: {path})\n")
    print(f"{'axis':11s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'spec':>6s} "
          f"{'bal-acc':>8s} {'MCC':>6s} | {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s}")
    print("-" * 78)
    agg = {k: [] for k in ["precision", "recall", "f1", "specificity", "bal_acc", "mcc"]}
    for j, ax in enumerate(AXES):
        m = per_axis(P, Y, PR, j)
        for k in agg:
            agg[k].append(m[k])
        print(f"{NICE[ax]:11s} {m['precision']:6.2f} {m['recall']:7.2f} {m['f1']:6.2f} "
              f"{m['specificity']:6.2f} {m['bal_acc']:8.2f} {m['mcc']:6.2f} | "
              f"{m['TP']:4d} {m['FP']:4d} {m['FN']:4d} {m['TN']:4d}")
    print("-" * 78)
    print(f"{'MEAN':11s} {np.mean(agg['precision']):6.2f} {np.mean(agg['recall']):7.2f} "
          f"{np.mean(agg['f1']):6.2f} {np.mean(agg['specificity']):6.2f} "
          f"{np.mean(agg['bal_acc']):8.2f} {np.mean(agg['mcc']):6.2f}")
    print("\nCLASH = positive class (label 0). precision = of flagged clashes, how many real; "
          "recall = of real clashes, how many caught; MCC = balanced correlation (0=chance).")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "runs/kfold_pooled.npz"
    if not os.path.exists(p):
        print(f"!! {p} not found -- pass the path to kfold_pooled.npz"); sys.exit(1)
    main(p)
