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
import pretty_midi


def _emit_tiled(out_notes, pitch, velocity, start, dur, L, period):
    """Place a note (and its cyclic repeats every `period`) so the loop TILES to fill
    the window [0, L] with no leading/trailing silence. Keeps only the portions
    overlapping [0, L]. This is what stops rhythm corruptions from leaving a single-loop
    envelope 'tell' (silence at the start/end) that the model could cheat on."""
    if period <= 1e-6:
        return
    kmin = int(np.floor((-start - dur) / period)) - 1
    kmax = int(np.ceil((L - start) / period)) + 1
    for k in range(kmin, kmax + 1):
        s = start + k * period
        e = s + dur
        if e > 0 and s < L:
            ss, ee = max(s, 0.0), min(e, L)
            if ee - ss > 1e-3:
                out_notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=ss, end=ee))

# Per-axis label vectors: [key, vertical, tempo, timing]  (1 = compatible).
# Four RELATIONAL compatibility axes: 2 harmony + 2 rhythm. (The single-loop
# "quality" axes pitch_jitter/jitter were dropped; register moved to Phase 2.)
#   HARMONY  key      - whole loop in the wrong key   (global, wrong scale)
#            vertical - clashes with the bass moment-to-moment, but STAYS in key
#   RHYTHM   tempo    - wrong speed                    (global)
#            timing   - constant phase offset          (relational)
LABELS = {
    "positive": [1, 1, 1, 1],
    "key":      [0, 1, 1, 1],
    "vertical": [1, 0, 1, 1],
    "tempo":    [1, 1, 0, 1],
    "timing":   [1, 1, 1, 0],
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
def circular_shift(pm, offset, L):
    """Roll every onset by `offset`, tiling with period L so the loop stays full (no
    leading/trailing silence). Used for PHASE AUGMENTATION: applying a random shared
    offset to both loops of a pair randomises absolute onset position, so the model
    can't cheat off 'when does B start' -- only the RELATIVE A-vs-B phase (the real
    timing signal) survives."""
    for inst in pm.instruments:
        new = []
        for n in inst.notes:
            _emit_tiled(new, n.pitch, n.velocity, n.start + offset, min(n.end - n.start, L), L, L)
        inst.notes = new
    return pm


def corrupt_key_midi(pm, rng, steps=None):
    """Transpose every pitched note by +/-1 semitone -> key clash. `steps` can be forced
    (e.g. -1 for variant 1, +1 for variant 2) so multiple negative variants differ.
    notes_changed = 0 means the stem has no pitched content (e.g. drums) -> skip."""
    if steps is None:
        steps = rng.choice([-1, 1])
    changed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + steps, 0, 127))
            changed += 1
    return pm, {"axis": "key", "semitones": int(steps), "notes_changed": changed}


def corrupt_tempo_midi(pm, bpm, rng, beats_per_slice=8, direction=None, pct_range=(0.10, 0.20)):
    """Play the loop at a different tempo. The loop's cycle period becomes factor*L, and
    we TILE it to fill the window so a faster loop simply cycles more times -- NOT leaving
    trailing silence (a single-loop shortcut). factor < 1 faster, > 1 slower.
    `direction` ('fast'/'slow') and `pct_range` can be forced so variants differ.
    Range bounds are perceptually motivated: floor >= 5% (below ~4%, drift over a 2-bar
    loop is < ~0.3 beats = inaudible -> would be a FALSE negative). Ceiling: note the
    fast side compounds (factor 1-pct -> tempo ratio 1/(1-pct)), so pct <= 0.28 keeps the
    tempo ratio under ~1.39 -- safely below 3:2 (structured polymeter) and far from 2:1
    (double-time, which RE-SYNCS and is musically compatible)."""
    beat = 60.0 / bpm
    L = beat * beats_per_slice
    pct = rng.uniform(*pct_range)
    if direction is None:
        direction = "fast" if rng.random() < 0.5 else "slow"
    factor = 1.0 - pct if direction == "fast" else 1.0 + pct
    period = factor * L                       # the retimed loop repeats every factor*L
    changed = 0
    for inst in pm.instruments:
        new = []
        for n in inst.notes:
            _emit_tiled(new, n.pitch, n.velocity, n.start * factor,
                        min((n.end - n.start) * factor, L), L, period)
            changed += 1
        inst.notes = new
    return pm, {"axis": "tempo", "factor": round(factor, 3), "notes_changed": changed}


def corrupt_timing_midi(pm, bpm, rng, jitter_frac=0.15, beats_per_slice=8):
    """HYBRID phase clash via a CIRCULAR (wrap-around) shift: move every onset by a
    constant 0.3-0.7 beat offset + a small per-note nudge, then wrap modulo the loop
    length so notes pushed past the end reappear at the front.

    Why wrap-around: a plain delay leaves LEADING SILENCE at the start of the loop, which
    the model can detect from loop B ALONE ("does B start late?") -- a shortcut that has
    nothing to do with A-vs-B alignment. Wrapping keeps the loop full from t=0, so the
    ONLY way to detect the clash is to genuinely compare the two loops' onset patterns.
    (A circular shift is also what phase-shifting a *loop* actually means.)"""
    beat = 60.0 / bpm
    L = beat * beats_per_slice                         # the 2-bar (8-beat) slice length
    off = rng.uniform(0.3, 0.7) * beat
    changed = 0
    for inst in pm.instruments:
        new = []
        for n in inst.notes:
            nudge = rng.uniform(-jitter_frac, jitter_frac) * beat
            dur = min(n.end - n.start, L)
            # TILE with period L (a loop repeats every L) so the wrap fills the front:
            # notes from the previous cycle populate [0, off] -> no leading silence.
            _emit_tiled(new, n.pitch, n.velocity, n.start + off + nudge, dur, L, L)
            changed += 1
        inst.notes = new
    return pm, {"axis": "timing", "offset_beats": round(off / beat, 3),
                "jitter_beat": jitter_frac, "wrap": True, "notes_changed": changed}


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


def corrupt_vertical(pm_melody, pm_ref, rng, rank=0):
    """Move each melody note to the most DISSONANT interval against the concurrent
    reference (bass) note -- but ONLY to pitch classes already present in the pair, so
    the loop STAYS IN KEY. This is what separates `vertical` from `key`: key changes the
    scale (out of key); vertical keeps the scale, breaks only the moment-to-moment
    interval. RELATIONAL: needs the partner loop `pm_ref`. Corrupts pm_melody in place.
    `rank` picks the rank-th most dissonant in-scale class (0=worst, 1=second-worst) so
    multiple negative variants differ."""
    ref = [(n.start, n.end, n.pitch) for inst in pm_ref.instruments
           if not inst.is_drum for n in inst.notes]
    if not ref:
        return pm_melody, {"axis": "vertical", "notes_changed": 0}

    # the shared scale = pitch classes actually used in either loop (their key material)
    scale = {n.pitch % 12 for inst in pm_ref.instruments if not inst.is_drum for n in inst.notes}
    scale |= {n.pitch % 12 for inst in pm_melody.instruments if not inst.is_drum for n in inst.notes}
    if not scale:
        return pm_melody, {"axis": "vertical", "notes_changed": 0}

    def dissonance(pc, r):
        ic = abs(pc - r) % 12
        ic = min(ic, 12 - ic)                 # interval class 0..6
        return {1: 3, 6: 2, 2: 1}.get(ic, 0)  # m2/M7 worst, then tritone, then M2/m7

    changed = 0
    for inst in pm_melody.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            r = next((rp for (rs, re, rp) in ref if rs <= n.start < re), None)
            if r is None:
                r = min(ref, key=lambda x: abs(x[0] - n.start))[2]
            # pick the rank-th most dissonant IN-SCALE pitch class vs the bass note
            ranked = sorted(scale, key=lambda pc: -dissonance(pc, r))
            target_pc = ranked[min(rank, len(ranked) - 1)]
            new = n.pitch + ((target_pc - n.pitch) % 12)      # nearest pitch of that class
            if new - n.pitch > 6:
                new -= 12
            if new != n.pitch:
                n.pitch = int(np.clip(new, 0, 127))
                changed += 1
    return pm_melody, {"axis": "vertical", "notes_changed": changed, "in_scale": True}


# =========================================================================
# Dispatch tables
# =========================================================================
# B-only MIDI corruptions (edit loop B's notes, then RENDER).
MIDI_CORRUPTIONS = {
    "key":    corrupt_key_midi,     # (pm, rng)
    "tempo":  corrupt_tempo_midi,   # (pm, bpm, rng)
    "timing": corrupt_timing_midi,  # (pm, bpm, rng)
}
# `vertical` is RELATIONAL (needs the partner loop) -> handled separately, not here.
# (pitch_jitter / jitter dropped: single-loop "quality", not pair compatibility.)

NEEDS_BPM = {"timing", "tempo"}   # both need the loop length (from bpm) to tile the window

# Audio-domain versions kept for reference / augmentation (not used in the MIDI pipeline)
AUDIO_CORRUPTIONS = {
    "key": corrupt_key,
    "tempo": corrupt_tempo,
    "timing": corrupt_timing,
}
