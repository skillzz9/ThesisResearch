"""
demo_pair.py
------------
Sanity-check the pipeline BY EAR. Renders positive + each negative as MIXED audio
(A+B together) so you can listen and confirm each corruption clashes.

Does MULTIPLE pairs across MULTIPLE songs, one folder per pair:
    demo/<Track>_p<anchor>_<combo>/positive.wav, key.wav, tempo.wav, ...

Usage:  ./.venv/bin/python demo_pair.py Track00002 Track00003 ...
        (no args -> a default set of songs)
"""

import os
import sys
import copy
import random

import numpy as np
import soundfile as sf

import build_dataset as BD
import corruptions as C
import midi_utils as M

SR = M.RENDER_SR
PAIRS_PER_SONG = 2


def mix(yA, yB):
    n = min(len(yA), len(yB))
    m = yA[:n].copy() + yB[:n]
    peak = float(np.max(np.abs(m))) or 1.0
    return (m / peak * 0.9).astype(np.float32)


def render_slice(track_dir, stem, start, dur, pm):
    y = M.render_midi(pm)
    n = int(round(dur * SR))
    return y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), np.float32)])


def demo_pair(track_dir, anchor, cp, bpm, measure_dur, rng):
    track = os.path.basename(track_dir)
    A, B = cp["stem_A"], cp["stem_B"]
    start = anchor * measure_dur
    outdir = f"demo/{track}_p{anchor}_{cp['combo_type']}_{A}-{B}"
    os.makedirs(outdir, exist_ok=True)

    pmA = M.slice_stem_midi(M.stem_midi_path(track_dir, A), start, start + measure_dur)
    pmB = M.slice_stem_midi(M.stem_midi_path(track_dir, B), start, start + measure_dur)
    yA = render_slice(track_dir, A, start, measure_dur, pmA)
    yB = render_slice(track_dir, B, start, measure_dur, pmB)
    sf.write(f"{outdir}/positive.wav", mix(yA, yB), SR)

    made = ["positive"]
    for axis in ["key", "vertical", "pitch_jitter", "tempo", "timing", "jitter"]:
        if axis == "vertical":
            # reference = lower-pitched stem (foundation); corrupt the higher one
            a_low = BD.mean_pitch(pmA) <= BD.mean_pitch(pmB)
            ref_pm = pmA if a_low else pmB
            corr_pm, info = C.corrupt_vertical(copy.deepcopy(pmB if a_low else pmA), ref_pm, rng)
            if info.get("notes_changed", 1) == 0:
                continue
            corr_y = render_slice(track_dir, "", start, measure_dur, corr_pm)
            clean_y = yA if a_low else yB
            sf.write(f"{outdir}/vertical.wav", mix(clean_y, corr_y), SR)
            made.append("vertical")
            continue

        pmBc = copy.deepcopy(pmB)
        fn = C.MIDI_CORRUPTIONS[axis]
        pmBc, info = (fn(pmBc, bpm, rng) if axis in C.NEEDS_BPM else fn(pmBc, rng))
        if info.get("notes_changed", 1) == 0:
            continue
        yBc = render_slice(track_dir, B, start, measure_dur, pmBc)
        sf.write(f"{outdir}/{axis}.wav", mix(yA, yBc), SR)
        made.append(axis)
    print(f"  {outdir}  ({cp['cat_A']}+{cp['cat_B']}, {bpm:.0f} BPM)  ->  {', '.join(made)}")


def demo_song(track_dir):
    rng = random.Random(0)
    sel = BD.select_track(track_dir)
    if sel is None:
        print(f"  {os.path.basename(track_dir)}: no usable pairs"); return
    bpm, measure_dur, pairs = sel
    # prefer melody+melody pairs; fall back to any pitched pair
    pool = [(a, cp) for a, cp in pairs if cp["combo_type"] == "mel_mel"]
    if not pool:
        pool = [(a, cp) for a, cp in pairs]
    if not pool:
        print(f"  {os.path.basename(track_dir)}: no usable pairs"); return

    # prefer pairs from different anchors
    chosen, seen_anchors = [], set()
    for a, cp in pool:
        if a not in seen_anchors:
            chosen.append((a, cp)); seen_anchors.add(a)
        if len(chosen) >= PAIRS_PER_SONG:
            break
    for a, cp in pool:                               # top up if only one anchor available
        if len(chosen) >= PAIRS_PER_SONG:
            break
        if (a, cp) not in chosen:
            chosen.append((a, cp))

    print(os.path.basename(track_dir) + ":")
    for anchor, cp in chosen:
        demo_pair(track_dir, anchor, cp, bpm, measure_dur, rng)


if __name__ == "__main__":
    tracks = sys.argv[1:] or ["Track00002", "Track00003", "Track00005", "Track00006"]
    for t in tracks:
        td = t if os.path.isdir(t) else f"{BD.ROOT_DIR}/{t}"
        demo_song(td)
    print("\nListen: each folder = one pair. positive.wav should sit well; each other file should clash on its axis.")
