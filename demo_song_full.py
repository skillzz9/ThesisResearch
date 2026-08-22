"""
demo_song_full.py
-----------------
Run the WHOLE pipeline on ONE song and render every pair it produces to mixed audio,
so you can hear AND see the complete output of the data builder for that song:

    song -> anchors -> combos -> for each combo:
            positive, augmented, key, vertical, tempo, timing  (all mixed A+B)

Output: demo_fullsong/<Track>/<anchor>_<combo>_<A>-<B>/*.wav  + a printed summary table.

Usage:  ./.venv/bin/python demo_song_full.py Track00002
"""

import os
import sys
import copy
import random

import numpy as np
import soundfile as sf

import build_dataset as BD
import build_positive_pairs as BPP
import corruptions as C
import midi_utils as M

SR = M.RENDER_SR
AUG_SHIFTS = BD.AUG_SHIFTS


def mix(yA, yB):
    n = min(len(yA), len(yB))
    m = yA[:n].copy() + yB[:n]
    peak = float(np.max(np.abs(m))) or 1.0
    return (m / peak * 0.9).astype(np.float32)


def render(pm, dur):
    y = M.render_midi(pm)
    n = int(round(dur * SR))
    return y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), np.float32)])


def run(track_dir):
    track = os.path.basename(track_dir)
    sel = BD.select_track(track_dir)
    if sel is None:
        print(f"{track}: no usable pairs"); return
    bpm, measure_dur, pairs = sel
    rng = random.Random(BD.SEED)

    print(f"\n{'='*72}\n {track}  |  {bpm:.0f} BPM  |  {len(pairs)} combos across "
          f"{len(set(a for a,_ in pairs))} anchors\n{'='*72}")
    print(f" {'folder':42s} {'combo':9s} {'files rendered'}")
    print(f" {'-'*42} {'-'*9} {'-'*30}")

    total = 0
    for anchor, cp in pairs:
        A, B = cp["stem_A"], cp["stem_B"]
        outdir = f"demo_fullsong/{track}/p{anchor}_{cp['combo_type']}_{A}-{B}"
        os.makedirs(outdir, exist_ok=True)
        start = anchor * measure_dur

        pmA = M.slice_stem_midi(M.stem_midi_path(track_dir, A), start, start + measure_dur)
        pmB = M.slice_stem_midi(M.stem_midi_path(track_dir, B), start, start + measure_dur)
        yA, yB = render(pmA, measure_dur), render(pmB, measure_dur)

        made = []
        # positive
        sf.write(f"{outdir}/positive.wav", mix(yA, yB), SR); made.append("positive")
        # augmented (both shifted equally)
        n = rng.choice(AUG_SHIFTS)
        yAa = render(BD.transpose_all(pmA, n), measure_dur)
        yBa = render(BD.transpose_all(pmB, n), measure_dur)
        sf.write(f"{outdir}/augmented.wav", mix(yAa, yBa), SR); made.append("augmented")

        # the 4 negatives
        for axis in BD.LABEL_COLS:
            if axis == "vertical":
                a_low = BD.mean_pitch(pmA) <= BD.mean_pitch(pmB)
                ref_pm = pmA if a_low else pmB
                corr, info = C.corrupt_vertical(copy.deepcopy(pmB if a_low else pmA), ref_pm, rng)
                if info.get("notes_changed", 1) == 0:
                    continue
                clean_y = yA if a_low else yB
                sf.write(f"{outdir}/vertical.wav", mix(clean_y, render(corr, measure_dur)), SR)
                made.append("vertical")
                continue
            fn = C.MIDI_CORRUPTIONS[axis]
            pmBc = copy.deepcopy(pmB)
            pmBc, info = (fn(pmBc, bpm, rng) if axis in C.NEEDS_BPM else fn(pmBc, rng))
            if info.get("notes_changed", 1) == 0:
                continue
            sf.write(f"{outdir}/{axis}.wav", mix(yA, render(pmBc, measure_dur)), SR)
            made.append(axis)

        total += len(made)
        short = os.path.basename(outdir)
        print(f" {short:42s} {cp['combo_type']:9s} {', '.join(made)}")

    print(f"\n Rendered {total} clips into demo_fullsong/{track}/")
    print(" Each folder = one combo. positive/augmented should sit well; key/vertical/tempo/timing each clash.")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "Track00002"
    td = t if os.path.isdir(t) else f"{BD.ROOT_DIR}/{t}"
    run(td)
