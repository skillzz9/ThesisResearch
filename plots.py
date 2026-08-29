"""
plots.py -- evidence-graph generation, shared by train_model / kfold / scaling_curve.
All figures land in graphs/ as PNG (300 dpi-ish), all raw numbers in runs/ as CSV/NPZ,
so every figure in the thesis is regenerable from logged data.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["key", "vertical", "tempo", "timing"]
NICE = {"key": "key", "vertical": "consonance", "tempo": "tempo", "timing": "timing"}
COLOR = {"key": "#9b2f7a", "vertical": "#b0472f", "tempo": "#0d7d8c", "timing": "#1f6f54"}

def _ensure():
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)


def epoch_curves(history, path="graphs/epoch_curves.png"):
    """history: list of dicts with keys ep, loss, {ax}_tr_acc, {ax}_va_acc (+ _auc)."""
    _ensure()
    eps = [h["ep"] for h in history]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), dpi=130)
    axs[0].plot([h["ep"] for h in history], [h["loss"] for h in history], "o-", color="#333")
    axs[0].set_title("training loss"); axs[0].set_xlabel("epoch"); axs[0].grid(alpha=.3)
    for i, (which, title) in enumerate([("va", "UNSEEN balanced accuracy"), ("tr", "SEEN balanced accuracy")]):
        ax = axs[1 + i]
        for a in AXES:
            ax.plot(eps, [h[f"{a}_{which}_acc"] for h in history], "o-", color=COLOR[a],
                    lw=2, ms=4, label=NICE[a])
        ax.axhline(0.5, color="#bbb", ls=":"); ax.axhline(0.6, color="#e0a63f", ls="--", lw=1)
        ax.set_ylim(0.3, 1.02); ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
        if i == 0: ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def gap_bars(seen, unseen, path="graphs/seen_unseen_gap.png"):
    """seen/unseen: {axis: balanced accuracy} (final)."""
    _ensure()
    x = np.arange(len(AXES)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=130)
    ax.bar(x - w/2, [seen[a] for a in AXES], w, color=[COLOR[a] for a in AXES], alpha=.45, label="SEEN (fit)")
    ax.bar(x + w/2, [unseen[a] for a in AXES], w, color=[COLOR[a] for a in AXES], label="UNSEEN (generalisation)")
    for xi, a in zip(x, AXES):
        ax.text(xi + w/2, unseen[a] + .01, f"{unseen[a]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.6, color="#e0a63f", ls="--", lw=1.2, label="criteria 0.60")
    ax.axhline(0.5, color="#bbb", ls=":")
    ax.set_xticks(x); ax.set_xticklabels([NICE[a] for a in AXES]); ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("balanced accuracy"); ax.set_title("fit vs generalisation (final model)")
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def _roc(scores, labels):
    """ROC for detecting CLASH: score = P(clash) = 1 - P(compatible); positive = label 0."""
    s = 1.0 - np.asarray(scores); y = (np.asarray(labels) == 0).astype(int)
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    tpr = tps / max(1, y.sum()); fpr = fps / max(1, (1 - y).sum())
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])


def roc_curves(P, Y, auc, path="graphs/roc_curves.png"):
    """P,Y: [N,4] arrays of P(compatible) and labels; auc: {axis: value} for legend."""
    _ensure()
    fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=130)
    for j, a in enumerate(AXES):
        fpr, tpr = _roc(P[:, j], Y[:, j])
        ax.plot(fpr, tpr, color=COLOR[a], lw=2, label=f"{NICE[a]}  (AUC {auc[a]:.2f})")
    ax.plot([0, 1], [0, 1], color="#bbb", ls=":", label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (clash recall)")
    ax.set_title("ROC per axis -- pooled held-out predictions")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def criteria_bars(acc, auc, path="graphs/criteria_scorecard.png",
                  acc_bar=0.60, auc_bar=0.70):
    _ensure()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
    for ax, vals, bar, title in [(axs[0], acc, acc_bar, "balanced accuracy (pooled unseen)"),
                                 (axs[1], auc, auc_bar, "AUC (pooled unseen)")]:
        names = [NICE[a] for a in AXES]
        ax.bar(names, [vals[a] for a in AXES], color=[COLOR[a] for a in AXES], alpha=.9)
        for i, a in enumerate(AXES):
            ax.text(i, vals[a] + .008, f"{vals[a]:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(bar, color="#e0a63f", ls="--", lw=1.4, label=f"criterion {bar:.2f}")
        ax.axhline(0.5, color="#bbb", ls=":")
        ax.set_ylim(0.4, 1.0); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def scaling_curves(sizes, mean, std, path="graphs/scaling_curves.png"):
    """mean/std: {axis: [v per size]} across seeds (std zeros if 1 seed)."""
    _ensure()
    fig, axs = plt.subplots(1, 5, figsize=(19, 3.9), dpi=130)
    panels = AXES + ["MEAN"]
    for i, name in enumerate(panels):
        ax = axs[i]
        if name == "MEAN":
            m = np.mean([mean[a] for a in AXES], axis=0); s = np.mean([std[a] for a in AXES], axis=0); c = "#15191b"
        else:
            m, s, c = np.array(mean[name]), np.array(std[name]), COLOR[name]
        ax.axhline(0.5, color="#bbb", ls=":"); ax.axhline(0.6, color="#e0a63f", ls="--", lw=1)
        ax.errorbar(sizes, m, yerr=s, fmt="o-", color=c, lw=2.2, ms=5, capsize=3)
        z = np.polyfit(sizes, m, 1)
        ax.plot(sizes, np.polyval(z, sizes), color=c, lw=1, alpha=.35)
        ax.set_title(NICE.get(name, name), fontsize=11)
        ax.set_xlabel("training songs"); ax.set_ylim(0.35, 0.95); ax.grid(alpha=.3)
        if i == 0: ax.set_ylabel("unseen balanced accuracy")
    plt.suptitle("Unseen accuracy vs training songs (mean ± std over seeds)", y=1.03, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path
