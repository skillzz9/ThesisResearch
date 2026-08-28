"""
build_dataset.py
----------------
The full data pipeline: songs -> positive & negative pairs -> feature tensors -> manifest.

Everything is rendered from MIDI through one soundfont (consistent domain).
For each combo pair (A, B) at an anchor measure:
    * POSITIVE            (A, B)                     label [1,1,1,1,1]
    * AUGMENTED POSITIVE  (A+n, B+n)                 label [1,1,1,1,1]   (both-pitched only)
    * one negative per valid axis, corrupting B     label with that axis = 0

Axis validity:
    * key, mode   -> only when BOTH stems are pitched (a detuned bass doesn't clash with drums)
    * tempo, timing, polyrhythm -> always (rhythm applies to drums too)
    * any negative whose corruption changed nothing is skipped (avoids false negatives)

Output: features/*.pt  (each = {'image_tensor': [3,84,T]})  +  dataset.csv manifest.
"""

import os
import glob
import copy
import random
import csv

# keep each worker single-threaded so N processes don't oversubscribe the CPU
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
torch.set_num_threads(1)

import build_positive_pairs as BPP
import corruptions as C
import midi_utils as M
from left_eye.feature_extractor import VisualFeatureExtractor
from create_loops import extract_bpm_from_midi, is_4_4_time

# ---------------------------------------------------------------------------
ROOT_DIR = os.environ.get("SLAKH_ROOT", "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k")
OUT_DIR = os.environ.get("DATASET_OUT", "features")   # env-based so spawned workers see it
MANIFEST = os.environ.get("DATASET_MANIFEST", "dataset.csv")
SEED = 42
N_TRACKS = None          # None = all; set to a small int to test quickly
AUG_SHIFTS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]   # wider range -> better key invariance
ANCHORS_PER_SONG = 6     # rich 2-bar anchors extracted per song (was 2) -> ~3x the pairs
N_AUG = 2                # augmented positives per pair (different shared transpositions)
N_NEG = 2                # negative variants per axis per pair (guaranteed different)
MIN_FILL = 0.6           # a stem's MIDI must cover >=60% of the measure with notes to be
                         # paired (measured on the MIDI we render, not Slakh's reverb-filled
                         # audio) -> stops sparse/silent loops entering the dataset
MELODIC_MAX_POLY = 1.2   # avg simultaneous notes <= this -> "melodic" (a clean single line);
                         # higher -> "chordal". Monophonic-only: ALL paired stems must be melodic.
# A melody must also MOVE (a held pad/strings line is monophonic but sounds like a chord):
MELODY_MAX_NOTE_BEATS = 1.5   # avg note duration must be <= this many beats (short = moving line)
MELODY_MIN_NOTES = 4          # ...and at least this many note onsets in the 8-beat measure

VFE = VisualFeatureExtractor(sample_rate=M.RENDER_SR)
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COLS = ["key", "vertical", "tempo", "timing"]   # 4 relational compatibility gates


def crop_or_pad(y, n):
    if len(y) >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y), dtype=y.dtype)])


def mean_pitch(pm):
    """Average MIDI pitch of the pitched notes (used to pick the lower/foundation stem)."""
    ps = [n.pitch for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    return (sum(ps) / len(ps)) if ps else 0.0


def midi_coverage(pm, dur):
    """Fraction of [0, dur] covered by at least one note (how much of the RENDERED clip
    actually has sound). Merges overlapping notes."""
    ivs = sorted((n.start, min(n.end, dur)) for inst in pm.instruments if not inst.is_drum
                 for n in inst.notes if n.start < dur and n.end > 0)
    if not ivs:
        return 0.0
    cov, cs, ce = 0.0, ivs[0][0], ivs[0][1]
    for s, e in ivs[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            cov += ce - cs
            cs, ce = s, e
    cov += ce - cs
    return cov / dur


def fill_fraction(chunk, sr, frame_sec=0.05, thresh=1e-3):
    """Fraction of a slice that is non-silent (>= thresh RMS over 50ms frames).
    Used to reject near-silent 'one-note' loops."""
    fl = max(1, int(frame_sec * sr))
    n = len(chunk) // fl
    if n == 0:
        return 0.0
    fr = chunk[: n * fl].reshape(n, fl).astype(np.float64)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    return float((rms > thresh).mean())


def midi_to_pt(pm, measure_samples, path):
    """Render a PrettyMIDI, crop to the measure length, extract CQT+Chroma, save .pt."""
    y = M.render_midi(pm)
    y = crop_or_pad(y, measure_samples)
    feat = VFE.extract_and_stack(y)
    t = torch.nan_to_num(torch.tensor(feat, dtype=torch.float32), nan=0.0)
    torch.save({"image_tensor": t}, path)
    return path


def transpose_all(pm, semitones):
    """Transpose every pitched note (used for the augmented positive)."""
    out = copy.deepcopy(pm)
    for inst in out.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + semitones, 0, 127))
    return out


def select_track(track_dir):
    """Anchors + combos for a track, using MIDI note activity (no audio loaded).
    Returns (bpm, measure_dur, [(anchor, {combo_type, stem_A, stem_B, cat_A, cat_B}) ...])."""
    import pretty_midi
    info = BPP.load_stem_info(track_dir)
    if not info:
        return None

    bpm = extract_bpm_from_midi(os.path.join(track_dir, "all_src.mid"))
    measure_dur = (60.0 / bpm) * BPP.BEATS_PER_SLICE

    # each pitched stem's note intervals (loaded once, from MIDI -- no audio I/O)
    stem_ivs, max_end = {}, 0.0
    for s in info:
        mp = M.stem_midi_path(track_dir, s)
        if mp is None:
            continue
        ivs = [(n.start, n.end) for inst in pretty_midi.PrettyMIDI(mp).instruments
               if not inst.is_drum for n in inst.notes]
        if ivs:
            stem_ivs[s] = ivs
            max_end = max(max_end, max(e for _, e in ivs))
    if not stem_ivs:
        return None
    num_measures = int(max_end / measure_dur)
    if num_measures < 1:
        return None

    def active_at(ivs, m):
        lo, hi = m * measure_dur, (m + 1) * measure_dur
        return any(a < hi and b > lo for a, b in ivs)

    sec_per_beat = measure_dur / BPP.BEATS_PER_SLICE

    def stats_at(ivs, m):
        """(coverage, avg polyphony, avg note duration in beats, note count) in measure m."""
        lo, hi = m * measure_dur, (m + 1) * measure_dur
        segs = [(max(a, lo), min(b, hi)) for a, b in ivs if a < hi and b > lo]
        if not segs:
            return 0.0, 0.0, 0.0, 0
        total = sum(b - a for a, b in segs)           # sum of note durations (voices x time)
        avg_dur_beats = (total / len(segs)) / sec_per_beat
        segs.sort()
        cov, cs, ce = 0.0, segs[0][0], segs[0][1]
        for a, b in segs[1:]:
            if a <= ce:
                ce = max(ce, b)
            else:
                cov += ce - cs
                cs, ce = a, b
        cov += ce - cs
        return cov / measure_dur, (total / cov if cov > 0 else 0.0), avg_dur_beats, len(segs)

    active_count = [sum(1 for s in stem_ivs if active_at(stem_ivs[s], m)) for m in range(num_measures)]

    anchors = BPP.top_measures(active_count, num_measures, k=ANCHORS_PER_SONG)

    rng = random.Random(SEED)
    out = []
    for anchor in anchors:
        active_here, melodic = [], set()
        for s in stem_ivs:
            if not active_at(stem_ivs[s], anchor):
                continue
            coverage, poly, avg_dur_beats, n_notes = stats_at(stem_ivs[s], anchor)
            if coverage < MIN_FILL:
                continue
            active_here.append(s)
            # melodic = monophonic AND actually moving (excludes held pads/sustained strings)
            if (poly <= MELODIC_MAX_POLY and avg_dur_beats <= MELODY_MAX_NOTE_BEATS
                    and n_notes >= MELODY_MIN_NOTES):
                melodic.add(s)
        combos = BPP.make_combos(active_here, info, rng, melodic=melodic)
        for combo_type, a, b in combos:
            out.append((anchor, {
                "combo_type": combo_type,
                "stem_A": a, "cat_A": info[a]["category"],
                "stem_B": b, "cat_B": info[b]["category"],
            }))
    return bpm, measure_dur, out


def process_track(track_dir):
    """Build all pairs for ONE track (independent unit of work -> run in parallel).
    Returns (rows, stats)."""
    track = os.path.basename(track_dir)
    sel = select_track(track_dir)
    if sel is None:
        return [], {}
    bpm, measure_dur, pairs = sel
    measure_samples = int(round(measure_dur * M.RENDER_SR))
    rng = random.Random(SEED)
    rows, uid = [], 0
    stats = {"pos": 0, "aug": 0}
    for ax in LABEL_COLS:
        stats[ax] = 0
    clean_cache = {}

    def pos_label():
        return {c: 1 for c in LABEL_COLS}

    def emit(fa, fb, label, kind, anchor, combo):
        rows.append({"file_A": fa, "file_B": fb, **label,
                     "type": kind, "track": track, "anchor": anchor, "combo": combo})

    def phase_of(anchor):
        """Deterministic random absolute phase per anchor (same for every loop at that
        anchor, so positives stay aligned and the clean cache stays consistent)."""
        return random.Random(SEED * 7919 + anchor).uniform(0.0, measure_dur)

    def pt_at_phase(pm, anchor, path):
        """Render a loop rolled by the anchor's phase (phase augmentation) -> .pt.
        Operates on a COPY so the caller's pm (used for corruption logic) is untouched."""
        rolled = C.circular_shift(copy.deepcopy(pm), phase_of(anchor), measure_dur)
        midi_to_pt(rolled, measure_samples, path)
        return path

    def clean_pt(stem, anchor):
        k = (stem, anchor)
        if k not in clean_cache:
            start = anchor * measure_dur
            pm = M.slice_stem_midi(M.stem_midi_path(track_dir, stem), start, start + measure_dur)
            path = os.path.join(OUT_DIR, f"{track}_{stem}_p{anchor}_clean.pt")
            pt_at_phase(pm, anchor, path)                 # render phase-rolled; keep pm unrolled for logic
            clean_cache[k] = (path, pm)
        return clean_cache[k]

    for anchor, cp in pairs:
        A, B = cp["stem_A"], cp["stem_B"]
        fa_clean, pmA = clean_pt(A, anchor)
        fb_clean, pmB = clean_pt(B, anchor)

        emit(fa_clean, fb_clean, pos_label(), "positive", anchor, cp["combo_type"])
        stats["pos"] += 1

        # augmented positives (both shifted equally) -- anti-shortcut for key/vertical
        for n in rng.sample(AUG_SHIFTS, N_AUG):
            pmA_aug, pmB_aug = transpose_all(pmA, n), transpose_all(pmB, n)
            fa_aug = os.path.join(OUT_DIR, f"{track}_{A}_p{anchor}_aug{uid}.pt")
            fb_aug = os.path.join(OUT_DIR, f"{track}_{B}_p{anchor}_aug{uid}.pt")
            pt_at_phase(pmA_aug, anchor, fa_aug)
            pt_at_phase(pmB_aug, anchor, fb_aug)
            emit(fa_aug, fb_aug, pos_label(), "augmented", anchor, cp["combo_type"])
            stats["aug"] += 1
            uid += 1

        # negatives: N_NEG variants per axis, corrupting A or B (chosen per variant)
        for axis in LABEL_COLS:
            for v in range(N_NEG):
                if axis == "vertical":
                    # relational: corrupt the HIGHER-pitched loop vs the lower reference
                    a_low = mean_pitch(pmA) <= mean_pitch(pmB)
                    ref_pm = pmA if a_low else pmB
                    clean_ref = fa_clean if a_low else fb_clean
                    corr_pm, inf = C.corrupt_vertical(
                        copy.deepcopy(pmB if a_low else pmA), ref_pm, rng, rank=v)
                    if inf.get("notes_changed", 1) == 0:
                        continue
                    f_neg = os.path.join(OUT_DIR, f"{track}_p{anchor}_vertical{uid}.pt")
                    pt_at_phase(corr_pm, anchor, f_neg)
                    emit(clean_ref, f_neg, dict(zip(LABEL_COLS, C.LABELS[axis])), axis,
                         anchor, cp["combo_type"])
                    stats[axis] += 1
                    uid += 1
                    continue

                # corrupt A or B at random (model must not assume "B is the broken one")
                corrupt_A = rng.random() < 0.5
                tgt_pm = pmA if corrupt_A else pmB
                tgt_stem = A if corrupt_A else B
                clean_partner = fb_clean if corrupt_A else fa_clean
                pmc = copy.deepcopy(tgt_pm)
                if axis == "key":
                    # variant 0: +/-1 semitone (the hardest, most dissonant case);
                    # variant 1: a wider transposition (+/-2..6) so the model learns
                    # "different key" in general, not just "shifted by exactly 1".
                    # (+/-12 excluded: an octave has the SAME pitch classes = same key.)
                    steps = (rng.choice([-1, 1]) if v == 0
                             else rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]))
                    pmc, inf = C.corrupt_key_midi(pmc, rng, steps=steps)
                elif axis == "tempo":
                    pmc, inf = C.corrupt_tempo_midi(pmc, bpm, rng,
                                                    direction=("fast" if v == 0 else "slow"))
                else:  # timing -- rng draws a different offset each variant
                    pmc, inf = C.corrupt_timing_midi(pmc, bpm, rng)
                if inf.get("notes_changed", 1) == 0:
                    continue
                f_neg = os.path.join(OUT_DIR, f"{track}_{tgt_stem}_p{anchor}_{axis}{uid}.pt")
                pt_at_phase(pmc, anchor, f_neg)
                emit(clean_partner, f_neg, dict(zip(LABEL_COLS, C.LABELS[axis])), axis,
                     anchor, cp["combo_type"])
                stats[axis] += 1
                uid += 1

    return rows, stats


def build(root_dir=ROOT_DIR, manifest=MANIFEST, n_tracks=N_TRACKS, workers=None):
    import multiprocessing as mp
    tracks = sorted(d for d in glob.glob(os.path.join(root_dir, "**", "Track*"), recursive=True)
                    if os.path.isdir(d))
    tracks = [t for t in tracks
              if os.path.exists(os.path.join(t, "all_src.mid")) and is_4_4_time(os.path.join(t, "all_src.mid"))]
    if n_tracks:
        tracks = tracks[:n_tracks]

    workers = workers or max(1, mp.cpu_count() - 1)
    print(f"Building {len(tracks)} tracks on {workers} workers -> {OUT_DIR}/")

    rows = []
    stats = {"pos": 0, "aug": 0}
    for ax in LABEL_COLS:
        stats[ax] = 0

    with mp.Pool(workers) as pool:
        for i, (trows, tstats) in enumerate(pool.imap_unordered(process_track, tracks), 1):
            rows.extend(trows)
            for k, v in tstats.items():
                stats[k] = stats.get(k, 0) + v
            print(f"  [{i}/{len(tracks)}] +{len(trows)} pairs  (total {len(rows)})")

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 55)
    print(" DATASET BUILD COMPLETE")
    print("=" * 55)
    print(f" total pairs : {len(rows)}")
    print(f" positives   : {stats['pos']}  | augmented: {stats['aug']}")
    for ax in LABEL_COLS:
        print(f" neg {ax:11s}: {stats[ax]}")
    print(f" manifest    : {manifest}  | features in: {OUT_DIR}/")
    print("=" * 55)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the compatibility dataset (parallel).")
    ap.add_argument("--root", default=ROOT_DIR, help="Slakh root dir (or set SLAKH_ROOT)")
    ap.add_argument("--out", default=OUT_DIR, help="feature output dir (or set DATASET_OUT)")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--n-tracks", type=int, default=None, help="limit tracks (for testing)")
    ap.add_argument("--workers", type=int, default=None, help="parallel workers (default cpu-1)")
    a = ap.parse_args()
    # push to env BEFORE the pool so spawned workers inherit the right paths
    os.environ["DATASET_OUT"] = a.out
    OUT_DIR = a.out
    os.makedirs(OUT_DIR, exist_ok=True)
    build(root_dir=a.root, manifest=a.manifest, n_tracks=a.n_tracks, workers=a.workers)
