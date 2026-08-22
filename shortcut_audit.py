"""
shortcut_audit.py
-----------------
Guard against SHORTCUTS. A corruption must only be detectable by comparing the two
loops (relational). If it leaves a single-loop "tell" in loop B's own envelope --
a shifted energy onset, an early offset (trailing silence), or a changed total energy
-- the model can cheat by looking at B alone. This renders clean B vs each corrupted B
across several pairs and reports how much B's OWN envelope changed. Near-zero = clean.
"""
import copy, random, glob, os
import numpy as np
import build_dataset as BD, corruptions as C, midi_utils as M

SR = M.RENDER_SR

def envelope(pm, dur):
    y = M.render_midi(pm)
    n = int(round(dur * SR))
    y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), np.float32)])
    e = np.abs(y); thr = (e.max() or 1.0) * 0.05
    on = np.argmax(e > thr) / SR if e.max() > 0 else 0.0
    off = (len(e) - np.argmax(e[::-1] > thr)) / SR if e.max() > 0 else 0.0
    return on, off, float((y ** 2).sum())

def audit(n_pairs=6):
    rng = random.Random(0)
    axes = ["key", "vertical", "tempo", "timing"]
    acc = {ax: {"don": [], "doff": [], "dE": []} for ax in axes}
    got = 0
    for td in sorted(glob.glob(f"{BD.ROOT_DIR}/Track*")):
        if not os.path.isdir(td) or got >= n_pairs:
            continue
        sel = BD.select_track(td)
        if not sel:
            continue
        bpm, mdur, pairs = sel
        for anchor, cp in pairs:
            if got >= n_pairs:
                break
            A, B = cp["stem_A"], cp["stem_B"]; start = anchor * mdur
            pmA = M.slice_stem_midi(M.stem_midi_path(td, A), start, start + mdur)
            pmB = M.slice_stem_midi(M.stem_midi_path(td, B), start, start + mdur)
            on0, off0, E0 = envelope(pmB, mdur)
            for ax in axes:
                if ax == "vertical":
                    a_low = BD.mean_pitch(pmA) <= BD.mean_pitch(pmB)
                    ref = pmA if a_low else pmB
                    corr, inf = C.corrupt_vertical(copy.deepcopy(pmB if a_low else pmA), ref, rng)
                    base = (on0, off0, E0) if a_low else envelope(pmA, mdur)
                else:
                    fn = C.MIDI_CORRUPTIONS[ax]
                    corr, inf = (fn(copy.deepcopy(pmB), bpm, rng) if ax in C.NEEDS_BPM
                                 else fn(copy.deepcopy(pmB), rng))
                    base = (on0, off0, E0)
                on, off, E = envelope(corr, mdur)
                acc[ax]["don"].append(abs(on - base[0]))
                acc[ax]["doff"].append(abs(off - base[1]))
                acc[ax]["dE"].append(abs(E - base[2]) / (base[2] + 1e-9))
            got += 1
    print(f"single-loop 'tell' per axis (averaged over {got} pairs) -- near 0 = no shortcut")
    print(f"{'axis':11s} {'Δonset(s)':>10s} {'Δoffset(s)':>11s} {'ΔenergyFrac':>12s}   verdict")
    for ax in axes:
        don = np.mean(acc[ax]["don"]); doff = np.mean(acc[ax]["doff"]); dE = np.mean(acc[ax]["dE"])
        leak = don > 0.05 or doff > 0.05 or dE > 0.30
        print(f"{ax:11s} {don:10.3f} {doff:11.3f} {dE:12.2f}   {'LEAK ⚠' if leak else 'clean ✓'}")

if __name__ == "__main__":
    audit()
