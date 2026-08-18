"""
music_theory.py
---------------
Symbolic (MIDI-based) music-theory metrics used to LABEL pairs. These run on the
exact notes at TRAINING time only; the model learns to predict them from audio.

Tier-2 "quality of match" metric #1: REGISTER MASKING.
Two parts that occupy the same frequency region fight for space and sound muddy;
parts in different registers leave room for each other. We measure this with a
critical-band (Bark) occupancy overlap: 1.0 = fully overlapping registers (max
masking, bad), 0.0 = disjoint registers (no masking, good).
"""
import math
import numpy as np

# ------------------------------------------------------------------ pitch -> frequency -> Bark
def note_to_freq(pitch):
    """MIDI note number -> fundamental frequency in Hz (A4=69=440Hz)."""
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def hz_to_bark(f):
    """Zwicker/Traunmüller Hz -> Bark critical-band scale."""
    return 13.0 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500.0) ** 2)


N_BARK = 24                     # ~24 critical bands cover human hearing
HARMONICS = 4                   # include first few harmonics -- masking happens across partials


def register_profile(notes, n_bark=N_BARK, n_harmonics=HARMONICS):
    """Energy-per-critical-band occupancy for a list of pretty_midi Notes.
    Weighted by note duration x velocity; each note contributes its first few
    harmonics (weaker with harmonic number)."""
    prof = np.zeros(n_bark, dtype=np.float64)
    for n in notes:
        dur = max(1e-3, n.end - n.start)
        w = dur * (n.velocity / 127.0)
        for h in range(1, n_harmonics + 1):
            f = note_to_freq(n.pitch) * h
            b = hz_to_bark(f)
            idx = int(np.clip(round(b), 0, n_bark - 1))
            prof[idx] += w / h              # higher harmonics contribute less energy
    return prof


def register_masking(notes_a, notes_b, n_bark=N_BARK, n_harmonics=HARMONICS):
    """Register-overlap score in [0,1]. 1 = same register (muddy/masking, BAD),
    0 = disjoint registers (clean, GOOD). Cosine similarity of Bark profiles."""
    pa = register_profile(notes_a, n_bark, n_harmonics)
    pb = register_profile(notes_b, n_bark, n_harmonics)
    na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(pa, pb) / (na * nb))


# ------------------------------------------------------------------ helpers
def all_notes(pm):
    """Flatten all notes from every (non-drum) instrument of a PrettyMIDI object."""
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    return notes


def transpose_notes(notes, semitones):
    """Return a new list of notes shifted by `semitones` (for octave-collision tests)."""
    import pretty_midi
    out = []
    for n in notes:
        p = int(np.clip(n.pitch + semitones, 0, 127))
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=p, start=n.start, end=n.end))
    return out
