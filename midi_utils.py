"""
midi_utils.py
-------------
Slice a stem's MIDI to a single anchor measure, and render MIDI -> audio with
FluidSynth. Everything downstream (positives + all 5 negatives) is rendered through
this so the whole dataset lives in one consistent synthetic domain.
"""

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import pretty_midi

# Path to the General-MIDI soundfont used to render EVERYTHING.
# MuseScore_General is fuller/more realistic than the compact FluidR3Mono.
# Env-overridable so cloud workers can point at their own copy.
SOUNDFONT = os.environ.get(
    "SOUNDFONT", "/Users/hugoposthuma/Downloads/Thesis/soundfonts/MuseScore_General.sf3")
RENDER_SR = 22050


def slice_stem_midi(stem_midi_path, start_sec, end_sec):
    """Return a PrettyMIDI containing only the notes inside [start_sec, end_sec),
    shifted so the measure starts at t=0. Instrument program/drum-flag preserved
    so it renders with a sensible timbre."""
    src = pretty_midi.PrettyMIDI(stem_midi_path)
    out = pretty_midi.PrettyMIDI()
    for inst in src.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            if n.start < end_sec and n.end > start_sec:          # note overlaps the window
                s = max(n.start, start_sec) - start_sec
                e = min(n.end, end_sec) - start_sec
                if e > s:
                    new_inst.notes.append(
                        pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=s, end=e)
                    )
        if new_inst.notes:
            out.instruments.append(new_inst)
    return out


def render_midi(pm, sr=RENDER_SR, soundfont=SOUNDFONT):
    """Render a PrettyMIDI to a mono float32 audio array via the fluidsynth binary."""
    fd_mid, mid_path = tempfile.mkstemp(suffix=".mid"); os.close(fd_mid)
    fd_wav, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd_wav)
    try:
        pm.write(mid_path)
        try:
            subprocess.run(
                ["fluidsynth", "-ni", "-F", wav_path, "-r", str(sr), "-q", soundfont, mid_path],
                check=True, capture_output=True, timeout=60,   # never let one render hang the pool
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return np.zeros(1, dtype=np.float32)               # bad/hung render -> silence, keep going
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return np.zeros(1, dtype=np.float32)
        audio, _ = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)
    finally:
        for p in (mid_path, wav_path):
            if os.path.exists(p):
                os.remove(p)


def stem_midi_path(track_dir, stem_key):
    """Locate the per-stem MIDI file (S00 -> MIDI/S00.mid)."""
    p = os.path.join(track_dir, "MIDI", f"{stem_key}.mid")
    return p if os.path.exists(p) else None
