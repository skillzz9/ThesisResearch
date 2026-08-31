"""
make_test_loops.py
------------------
Render a set of labelled test loops to test_loops/*.wav so you can play with
predict_from_audio.py (and listen to them yourself). Produces:
    A.wav            -- reference loop A
    B_compatible.wav -- loop B, fits A on all axes  (pair A + this = all compatible)
    B_key.wav        -- loop B transposed           (key CLASH)
    B_tempo.wav      -- loop B sped up              (tempo CLASH)
    B_timing.wav     -- loop B phase-shifted        (timing CLASH)
    B_consonance.wav -- loop B made dissonant vs A  (consonance CLASH)

Then, e.g.:
    python predict_from_audio.py test_loops/A.wav test_loops/B_compatible.wav
    python predict_from_audio.py test_loops/A.wav test_loops/B_key.wav
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
    sf.write(path, y, M.RENDER_SR)
    return path


def main():
    # find a track with a good bass+melody combo (so consonance corruption has a bass ref)
    pick = None
    for td in sorted(glob.glob(f"{BD.ROOT_DIR}/Track*")):
        sel = BD.select_track(td)
        if not sel:
            continue
        bpm, mdur, pairs = sel
        mm = [(a, c) for a, c in pairs if c["combo_type"] == "mel_mel"]
        if mm:
            pick = (td, bpm, mdur, mm[0]); break
    td, bpm, mdur, (anchor, cp) = pick
    start = anchor * mdur
    A, B = cp["stem_A"], cp["stem_B"]
    print(f"source: {os.path.basename(td)}  stems {A}+{B}  bpm {bpm:.0f}")

    pmA = M.slice_stem_midi(M.stem_midi_path(td, A), start, start + mdur)
    pmB = M.slice_stem_midi(M.stem_midi_path(td, B), start, start + mdur)

    render(pmA, mdur, f"{OUT}/A.wav")
    render(pmB, mdur, f"{OUT}/B_compatible.wav")
    render(C.corrupt_key_midi(copy.deepcopy(pmB), rng, steps=5)[0], mdur, f"{OUT}/B_key.wav")
    render(C.corrupt_tempo_midi(copy.deepcopy(pmB), bpm, rng, direction="fast")[0], mdur, f"{OUT}/B_tempo.wav")
    render(C.corrupt_timing_midi(copy.deepcopy(pmB), bpm, rng)[0], mdur, f"{OUT}/B_timing.wav")
    # consonance: make B dissonant against A (the lower stem is the reference)
    a_low = BD.mean_pitch(pmA) <= BD.mean_pitch(pmB)
    ref = pmA if a_low else pmB
    tgt = copy.deepcopy(pmB if a_low else pmA)
    render(C.corrupt_vertical(tgt, ref, rng)[0], mdur, f"{OUT}/B_consonance.wav")

    print(f"\nwrote 6 files to {OUT}/. Try:")
    print(f"  python predict_from_audio.py {OUT}/A.wav {OUT}/B_compatible.wav   # expect all compatible")
    print(f"  python predict_from_audio.py {OUT}/A.wav {OUT}/B_key.wav          # expect KEY clash")
    print(f"  python predict_from_audio.py {OUT}/A.wav {OUT}/B_tempo.wav        # expect TEMPO clash")
    print(f"  python predict_from_audio.py {OUT}/A.wav {OUT}/B_timing.wav       # expect TIMING clash")
    print(f"  python predict_from_audio.py {OUT}/A.wav {OUT}/B_consonance.wav   # expect CONSONANCE clash")


if __name__ == "__main__":
    main()
