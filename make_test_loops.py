"""
make_test_loops.py
------------------
Render 10 labelled test loops to test_loops/*.wav to drag into the web GUI (app.py).
Two songs, each giving a reference + compatible partner + single-axis clashes, so you
can hear/test compatible pairs and every kind of clash.

    python make_test_loops.py
"""
import os, copy, random, glob
import numpy as np
import soundfile as sf
import build_dataset as BD, corruptions as C, midi_utils as M

OUT = "test_loops"; os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)


def render(pm, mdur, path):
    y = M.render_midi(pm); n = int(round(mdur * M.RENDER_SR))
    y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), np.float32)])
    sf.write(path, y, M.RENDER_SR); return path


def pick_tracks(n=2):
    """Return n (track, bpm, mdur, anchor, combo) tuples from different songs."""
    picks = []
    for td in sorted(glob.glob(f"{BD.ROOT_DIR}/Track*")):
        sel = BD.select_track(td)
        if not sel:
            continue
        bpm, mdur, pairs = sel
        mm = [(a, c) for a, c in pairs if c["combo_type"] == "mel_mel"]
        if mm:
            picks.append((td, bpm, mdur, mm[0][0], mm[0][1]))
        if len(picks) >= n:
            break
    return picks


def loops_for(tag, td, bpm, mdur, anchor, cp, which):
    """Render the requested loop variants for one song. `which` = list of axis names
    (or 'A'/'ok'). Returns list of (filename, human label)."""
    start = anchor * mdur
    A, B = cp["stem_A"], cp["stem_B"]
    pmA = M.slice_stem_midi(M.stem_midi_path(td, A), start, start + mdur)
    pmB = M.slice_stem_midi(M.stem_midi_path(td, B), start, start + mdur)
    made = []
    for w in which:
        if w == "A":
            f = f"{OUT}/{tag}_A.wav"; render(pmA, mdur, f); made.append((f, "reference loop A"))
        elif w == "ok":
            f = f"{OUT}/{tag}_B_ok.wav"; render(pmB, mdur, f); made.append((f, "fits A (compatible)"))
        elif w == "key":
            f = f"{OUT}/{tag}_B_KEY.wav"; render(C.corrupt_key_midi(copy.deepcopy(pmB), rng, steps=5)[0], mdur, f); made.append((f, "KEY clash"))
        elif w == "tempo":
            f = f"{OUT}/{tag}_B_TEMPO.wav"; render(C.corrupt_tempo_midi(copy.deepcopy(pmB), bpm, rng, direction="fast")[0], mdur, f); made.append((f, "TEMPO clash"))
        elif w == "timing":
            f = f"{OUT}/{tag}_B_TIMING.wav"; render(C.corrupt_timing_midi(copy.deepcopy(pmB), bpm, rng)[0], mdur, f); made.append((f, "TIMING clash"))
        elif w == "consonance":
            a_low = BD.mean_pitch(pmA) <= BD.mean_pitch(pmB)
            ref = pmA if a_low else pmB; tgt = copy.deepcopy(pmB if a_low else pmA)
            f = f"{OUT}/{tag}_B_CONSONANCE.wav"; render(C.corrupt_vertical(tgt, ref, rng)[0], mdur, f); made.append((f, "CONSONANCE clash"))
    return made


def main():
    picks = pick_tracks(2)
    made = []
    # song1: reference + compatible + key/tempo/timing  (5 loops)
    made += loops_for("song1", *picks[0], which=["A", "ok", "key", "tempo", "timing"])
    # song2: reference + compatible + consonance/tempo/timing  (5 loops)
    made += loops_for("song2", *picks[1], which=["A", "ok", "consonance", "tempo", "timing"])

    print(f"\nWrote {len(made)} loops to {OUT}/:\n")
    for f, lbl in made:
        print(f"  {os.path.basename(f):26s} {lbl}")
    print("\nPairs to try in the GUI (drag two files in):")
    print("  song1_A + song1_B_ok        -> all compatible")
    print("  song1_A + song1_B_KEY       -> KEY clash")
    print("  song1_A + song1_B_TEMPO     -> TEMPO clash")
    print("  song1_A + song1_B_TIMING    -> TIMING clash")
    print("  song2_A + song2_B_ok        -> all compatible")
    print("  song2_A + song2_B_CONSONANCE-> CONSONANCE clash")
    print("  song1_A + song2_A           -> different songs (often several clashes)")


if __name__ == "__main__":
    main()
