"""
corruptions.py
--------------
One helper per negative axis. Each takes a known-good slice and breaks EXACTLY ONE
musical property, holding everything else constant, so the resulting pair gets a
clean single-axis label.

Two families:

  AUDIO corruptions  -> operate on a mono audio array, return corrupted audio.
      * corrupt_key      (transpose +/-1 semitone)      -> label [0,1,1,1,1]
      * corrupt_tempo    (time-stretch +/-10-20%)        -> label [1,1,0,1,1]
      * corrupt_timing   (offset off the downbeat)       -> label [1,0,1,1,1]

  MIDI corruptions   -> operate on a pretty_midi.PrettyMIDI, return corrupted MIDI
                        (must then be RENDERED to audio).
      * corrupt_mode        (flip the 3rd, major<->minor)  -> label [1,1,1,0,1]
      * corrupt_polyrhythm  (re-quantize to triplet grid)  -> label [1,1,1,1,0]

Each corrupts LOOP B only; loop A is left unchanged.

Every function returns (result, info) where info is a small dict describing exactly
what was done (for logging / reproducibility).

NOTE on domain consistency: key/tempo/timing work directly on Slakh audio slices.
mode/polyrhythm require the MIDI edit + render step. For a fully consistent dataset
every slice should ultimately come from the same render domain (see PLAN.md).
"""

import numpy as np
import librosa

# Per-axis label vectors: [key, vertical, pitch_jitter, tempo, timing, jitter]
# (1 = compatible). Symmetric: 3 harmony axes + 3 rhythm axes.
#   HARMONY  key          - whole melody in the wrong key (global)
#            vertical     - melody clashes with the bass moment-to-moment (relational)
#            pitch_jitter - each note randomly detuned -> out of tune (random)
#   RHYTHM   tempo        - wrong speed (global)
#            timing       - constant phase offset (relational)
#            jitter       - each note randomly off the grid (random)
LABELS = {
    "positive":     [1, 1, 1, 1, 1, 1],
    "key":          [0, 1, 1, 1, 1, 1],
    "vertical":     [1, 0, 1, 1, 1, 1],
    "pitch_jitter": [1, 1, 0, 1, 1, 1],
    "tempo":        [1, 1, 1, 0, 1, 1],
    "timing":       [1, 1, 1, 1, 0, 1],
    "jitter":       [1, 1, 1, 1, 1, 0],
}


# =========================================================================
# AUDIO corruptions  (input: mono float array `y`, sample rate `sr`)
# =========================================================================
def corrupt_key(y, sr, rng):
    """Transpose by +/-1 semitone -> guaranteed harmonic (key) clash.
    +/-1 (minor 2nd) is the most dissonant interval; do NOT use +/-2."""
    steps = rng.choice([-1, 1])
    y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    return y_out.astype(np.float32), {"axis": "key", "semitones": int(steps)}


def corrupt_tempo(y, sr, rng):
    """Time-stretch by +/-10-20% -> tempo mismatch.
    Length CHANGES on purpose: the different transient spacing IS the tempo signal.
    (rate > 1 = faster/shorter, rate < 1 = slower/longer)."""
    pct = rng.uniform(0.10, 0.20)
    rate = 1.0 + pct if rng.random() < 0.5 else 1.0 - pct
    y_out = librosa.effects.time_stretch(y, rate=rate)
    return y_out.astype(np.float32), {"axis": "tempo", "rate": round(rate, 3)}


def corrupt_timing(y, sr, bpm, rng):
    """Delay by a NON-metric fraction of a beat (~30-70%) -> downbeats no longer align.
    Off-grid so the clash is always perceptible. Length preserved (silence in, tail out)."""
    beat_samples = sr * 60.0 / bpm
    frac = rng.uniform(0.3, 0.7)
    offset = int(round(frac * beat_samples))
    if offset <= 0:
        return y.astype(np.float32), {"axis": "timing", "offset_beats": 0.0}
    y_out = np.concatenate([np.zeros(offset, dtype=y.dtype), y])[: len(y)]
    return y_out.astype(np.float32), {"axis": "timing", "offset_beats": round(frac, 3)}


# =========================================================================
# MIDI corruptions  (input: pretty_midi.PrettyMIDI `pm`; returns corrupted pm)
# These edit the notes; the result must be RENDERED to audio afterwards.
# =========================================================================
def corrupt_mode(pm, rng):
    """Swap major<->minor by flipping the 3rd, 6th AND 7th scale degrees relative to
    the estimated tonic (these three degrees are what define the mode). Flipping all
    three (not just the 3rd) makes the clash audible while staying a true mode change.
    Only affects pitched instruments (drums untouched)."""
    from collections import Counter

    pcs = [n.pitch % 12 for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not pcs:
        return pm, {"axis": "mode", "tonic": None, "notes_changed": 0}

    tonic = Counter(pcs).most_common(1)[0][0]
    major_degrees = {(tonic + 4) % 12, (tonic + 9) % 12, (tonic + 11) % 12}  # 3rd, 6th, 7th
    minor_degrees = {(tonic + 3) % 12, (tonic + 8) % 12, (tonic + 10) % 12}

    changed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pc = n.pitch % 12
            if pc in major_degrees:      # major -> minor (lower a semitone)
                n.pitch -= 1
                changed += 1
            elif pc in minor_degrees:    # minor -> major (raise a semitone)
                n.pitch += 1
                changed += 1
    return pm, {"axis": "mode", "tonic": int(tonic), "notes_changed": changed}


def corrupt_polyrhythm(pm, bpm, rng):
    """Re-quantize note onsets to a TRIPLET grid (3 subdivisions/beat).
    Against a straight-grid partner this produces a 3-against-4 polyrhythm.
    Corrupt clearly so the clash is reliable."""
    triplet = (60.0 / bpm) / 3.0
    changed = 0
    for inst in pm.instruments:
        for n in inst.notes:
            dur = n.end - n.start
            new_start = round(n.start / triplet) * triplet
            if abs(new_start - n.start) > 1e-6:
                changed += 1
            n.start = new_start
            n.end = new_start + max(dur, triplet * 0.5)
    return pm, {"axis": "polyrhythm", "triplet_interval": round(triplet, 4), "notes_changed": changed}


# =========================================================================
# MIDI versions of key / tempo / timing
# (used when rendering EVERYTHING from MIDI, so all 5 axes share one domain
#  with no audio-processing artefacts that could leak the label)
# =========================================================================
def corrupt_key_midi(pm, rng):
    """Transpose every pitched note by +/-1 semitone -> key clash.
    notes_changed = 0 means the stem has no pitched content (e.g. drums) -> skip."""
    steps = rng.choice([-1, 1])
    changed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + steps, 0, 127))
            changed += 1
    return pm, {"axis": "key", "semitones": int(steps), "notes_changed": changed}


def corrupt_tempo_midi(pm, rng):
    """Scale all note times by +/-10-20% -> tempo mismatch.
    factor < 1 compresses (faster), factor > 1 stretches (slower)."""
    pct = rng.uniform(0.10, 0.20)
    factor = 1.0 - pct if rng.random() < 0.5 else 1.0 + pct
    changed = 0
    for inst in pm.instruments:
        for n in inst.notes:
            n.start *= factor
            n.end *= factor
            changed += 1
    return pm, {"axis": "tempo", "factor": round(factor, 3), "notes_changed": changed}


def corrupt_timing_midi(pm, bpm, rng):
    """Delay every onset by a NON-metric fraction of a beat (~30-70%)."""
    beat = 60.0 / bpm
    frac = rng.uniform(0.3, 0.7)
    off = frac * beat
    changed = 0
    for inst in pm.instruments:
        for n in inst.notes:
            n.start += off
            n.end += off
            changed += 1
    return pm, {"axis": "timing", "offset_beats": round(frac, 3), "notes_changed": changed}


def corrupt_jitter_midi(pm, bpm, rng, max_frac=0.2):
    """Move EVERY note by an independent random amount (up to +/-max_frac of a beat)
    -> the loop is sloppy / off the grid everywhere. Distinct from `timing` (a constant
    offset): this destroys tightness, not just phase. Cross-correlation goes flat."""
    beat = 60.0 / bpm
    changed = 0
    for inst in pm.instruments:
        for n in inst.notes:
            shift = rng.uniform(-max_frac, max_frac) * beat
            dur = n.end - n.start
            n.start = max(0.0, n.start + shift)
            n.end = n.start + dur
            changed += 1
    return pm, {"axis": "jitter", "max_beat": max_frac, "notes_changed": changed}


def corrupt_pitch_jitter_midi(pm, rng):
    """Move each pitched note by a random +/-1 or +/-2 semitones -> the melody wanders
    out of tune (the harmony twin of rhythmic jitter). Drums untouched. The bass stays
    in key, so the tonal centre is intact while the melody sounds out of tune."""
    changed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + rng.choice([-2, -1, 1, 2]), 0, 127))
            changed += 1
    return pm, {"axis": "pitch_jitter", "notes_changed": changed}


def corrupt_vertical(pm_melody, pm_ref, rng):
    """Move each melody note to a TRITONE (max Circle-of-Fifths distance) above the
    reference note (bass / other loop) sounding at the same moment -> guaranteed
    moment-to-moment harmonic clash. RELATIONAL: needs the partner loop `pm_ref`.
    Corrupts pm_melody in place."""
    ref = [(n.start, n.end, n.pitch) for inst in pm_ref.instruments
           if not inst.is_drum for n in inst.notes]
    if not ref:
        return pm_melody, {"axis": "vertical", "notes_changed": 0}

    changed = 0
    for inst in pm_melody.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            r = next((rp for (rs, re, rp) in ref if rs <= n.start < re), None)
            if r is None:
                r = min(ref, key=lambda x: abs(x[0] - n.start))[2]
            target_pc = (r + 6) % 12                      # tritone from ref = max dissonance
            new = n.pitch + ((target_pc - n.pitch) % 12)  # nearest pitch with that class
            if new - n.pitch > 6:
                new -= 12
            if new != n.pitch:
                n.pitch = int(np.clip(new, 0, 127))
                changed += 1
    return pm_melody, {"axis": "vertical", "notes_changed": changed}


# =========================================================================
# Dispatch tables
# =========================================================================
# B-only MIDI corruptions (edit loop B's notes, then RENDER).
MIDI_CORRUPTIONS = {
    "key":          corrupt_key_midi,          # (pm, rng)
    "pitch_jitter": corrupt_pitch_jitter_midi, # (pm, rng)
    "tempo":        corrupt_tempo_midi,         # (pm, rng)
    "timing":       corrupt_timing_midi,        # (pm, bpm, rng)
    "jitter":       corrupt_jitter_midi,        # (pm, bpm, rng)
}
# `vertical` is RELATIONAL (needs the partner loop) -> handled separately, not here.

NEEDS_BPM = {"timing", "jitter"}

# Audio-domain versions kept for reference / augmentation (not used in the MIDI pipeline)
AUDIO_CORRUPTIONS = {
    "key": corrupt_key,
    "tempo": corrupt_tempo,
    "timing": corrupt_timing,
}
