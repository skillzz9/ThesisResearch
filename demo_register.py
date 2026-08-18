"""Validate the register-masking metric on real babyslakh stems.
Slides stem B across octaves; masking should PEAK when the two overlap in register
and DROP as they separate."""
import os, glob
import pretty_midi
import numpy as np
from music_theory import register_masking, all_notes, transpose_notes, register_profile, note_to_freq
from build_positive_pairs import load_stem_info, ROOT_DIR
from midi_utils import stem_midi_path

def mean_pitch(notes):
    return float(np.mean([n.pitch for n in notes])) if notes else float("nan")

# find a track with >=2 pitched stems that have MIDI
tracks = sorted(d for d in glob.glob(os.path.join(ROOT_DIR, "**", "Track*"), recursive=True) if os.path.isdir(d))
for track_dir in tracks:
    info = load_stem_info(track_dir)
    stems = [s for s in info if stem_midi_path(track_dir, s)]
    if len(stems) < 2:
        continue
    # load notes for each stem, keep the two with the most notes
    loaded = []
    for s in stems:
        pm = pretty_midi.PrettyMIDI(stem_midi_path(track_dir, s))
        notes = all_notes(pm)
        if notes:
            loaded.append((s, info[s]["category"], notes))
    if len(loaded) < 2:
        continue
    loaded.sort(key=lambda x: -len(x[2]))
    (sa, ca, na), (sb, cb, nb) = loaded[0], loaded[1]
    print(f"track {os.path.basename(track_dir)}")
    print(f"  A = {sa} ({ca}) mean_pitch={mean_pitch(na):.1f}  {len(na)} notes")
    print(f"  B = {sb} ({cb}) mean_pitch={mean_pitch(nb):.1f}  {len(nb)} notes")
    print(f"  natural register-masking (as-is) = {register_masking(na, nb):.3f}")
    print("  --- slide B by octaves (±) : masking should peak near register overlap ---")
    for oct_shift in (-24, -12, 0, +12, +24):
        m = register_masking(na, transpose_notes(nb, oct_shift))
        bar = "#" * int(m * 40)
        print(f"    B {oct_shift:+3d} st (mean_pitch {mean_pitch(nb)+oct_shift:5.1f}) : {m:.3f} {bar}")
    break
