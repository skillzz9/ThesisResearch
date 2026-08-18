"""
build_positive_pairs.py
------------------------
Phase 1 of the data pipeline: POSITIVE pair generation.

For every 4/4 track it:
  1. Reads BPM from the MIDI and builds a 2-bar (8-beat) measure grid.
  2. Detects, per measure, which stems are actually playing (RMS activity).
  3. Categorises stems into Drums / Bass / Melody (from metadata; Sound Effects excluded).
  4. Picks the 2 anchor measures: the RICHEST golden measure in Region A (10-50%)
     and Region B (50-90%) of the song.
  5. Generates the 7 stem-combos at each anchor:
         1x drum+bass, 1x bass+main-melody, 1x drum+main-melody, 4x melody+melody
  6. Slices those stems at the anchors and writes a manifest of positive pairs.

Negatives are NOT created here (see the negative generator, later). Every pair
emitted here is a POSITIVE with label [key, timing, tempo, mode, polyrhythm] = all 1.

NOTE: this slices the existing Slakh audio for now. The final pipeline will render
from MIDI for consistency with the note-level negatives (mode/polyrhythm) — that swap
happens when the negative generator + render pipeline are built.
"""

import os
import glob
import random
from itertools import combinations

import numpy as np
import soundfile as sf
import yaml

from create_loops import extract_bpm_from_midi, is_4_4_time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k"
OUTPUT_CSV = "positive_pairs.csv"
SEED = 42

BEATS_PER_SLICE = 8            # 2 bars in 4/4

# Region boundaries as fractions of the song (by measure index)
REGION_A = (0.10, 0.50)
REGION_B = (0.50, 0.90)

# A measure must have at least this many active stems to count as "golden"
MIN_STEMS_GOLDEN = 3

# Activity detection
ACTIVITY_REL = 0.15           # measure counts as active if RMS >= REL * stem's peak-measure RMS
ACTIVITY_FLOOR = 1e-3         # ...and above this absolute floor

# Melody-melody pairs to sample per anchor
MEL_MEL_COUNT = 4

# Instrument-class -> category (allow-list; anything not listed and not drum/bass is dropped)
MELODY_CLASSES = {
    "Guitar", "Piano", "Strings", "Strings (continued)", "Brass", "Organ",
    "Synth Pad", "Synth Lead", "Pipe", "Reed", "Chromatic Percussion", "Ethnic",
}
EXCLUDE_CLASSES = {"Sound Effects", "Sound effects"}


# ---------------------------------------------------------------------------
# Metadata / categorisation
# ---------------------------------------------------------------------------
def load_stem_info(track_dir):
    """Returns {stem_key: {category, loudness}} for usable stems only."""
    meta_path = os.path.join(track_dir, "metadata.yaml")
    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    info = {}
    for key, s in (meta.get("stems") or {}).items():
        inst = s.get("inst_class")
        if inst in EXCLUDE_CLASSES:
            continue
        if s.get("is_drum") or inst == "Drums":
            continue  # drums dropped -- we only model pitched (bass/melody) loops
        elif inst == "Bass":
            category = "Bass"
        elif inst in MELODY_CLASSES:
            category = "Melody"
        else:
            continue  # unknown class -> drop
        info[key] = {"category": category, "loudness": s.get("integrated_loudness", -99.0)}
    return info


def stem_wav_path(track_dir, stem_key):
    """Locate the audio file for a stem key (S00 -> stems/S00.wav or .flac)."""
    for ext in ("wav", "flac"):
        p = os.path.join(track_dir, "stems", f"{stem_key}.{ext}")
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Activity per measure
# ---------------------------------------------------------------------------
def measure_rms(audio, samples_per_measure, num_measures):
    """RMS energy of each measure-length chunk of a mono audio array."""
    rms = np.zeros(num_measures, dtype=np.float32)
    for m in range(num_measures):
        chunk = audio[m * samples_per_measure:(m + 1) * samples_per_measure]
        if len(chunk) == 0:
            continue
        rms[m] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return rms


def active_measures(rms):
    """Boolean array: which measures this stem is active in."""
    peak = rms.max()
    if peak <= 0:
        return np.zeros_like(rms, dtype=bool)
    thresh = max(ACTIVITY_FLOOR, ACTIVITY_REL * peak)
    return rms >= thresh


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------
def richest_measure_in_region(active_count, num_measures, region):
    """Index of the measure with the most active stems within a region, or None."""
    lo = int(region[0] * num_measures)
    hi = int(region[1] * num_measures)
    if hi <= lo:
        return None
    best_idx, best_count = None, -1
    for m in range(lo, hi):
        c = active_count[m]
        if c >= MIN_STEMS_GOLDEN and c > best_count:
            best_idx, best_count = m, c
    return best_idx


# ---------------------------------------------------------------------------
# Combo generation
# ---------------------------------------------------------------------------
def make_combos(active_stems, info, rng, melodic=None):
    """Generate the stem-combos from the stems active at an anchor.
    `melodic` = set of MONOPHONIC (single-line) stems. Melody-melody pairs must
    include at least one melodic stem so we avoid muddy chord-on-chord pairings."""
    melodic = melodic or set()
    by_cat = {"Bass": [], "Melody": []}
    for s in active_stems:
        by_cat[info[s]["category"]].append(s)

    loud = lambda lst: max(lst, key=lambda s: info[s]["loudness"]) if lst else None
    bass = loud(by_cat["Bass"])
    mels = by_cat["Melody"]
    mel_lines = [s for s in mels if s in melodic]
    main = loud(mel_lines) if mel_lines else loud(mels)   # prefer a monophonic main line

    combos = []
    if bass and main:
        combos.append(("bass_main", bass, main))

    # melody-melody: require >=1 monophonic stem (no chord-on-chord); fall back if none
    mel_pairs = [p for p in combinations(sorted(mels), 2) if p[0] in melodic or p[1] in melodic]
    if not mel_pairs:
        mel_pairs = list(combinations(sorted(mels), 2))
    if mel_pairs:
        for a, b in rng.sample(mel_pairs, min(MEL_MEL_COUNT, len(mel_pairs))):
            combos.append(("mel_mel", a, b))
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(root_dir=ROOT_DIR, output_csv=OUTPUT_CSV):
    rng = random.Random(SEED)
    tracks = sorted(d for d in glob.glob(os.path.join(root_dir, "**", "Track*"), recursive=True)
                    if os.path.isdir(d))

    rows = []
    stats = {"tracks": 0, "skipped_4_4": 0, "anchors": 0, "pairs": 0}

    for track_dir in tracks:
        track_name = os.path.basename(track_dir)
        midi_path = os.path.join(track_dir, "all_src.mid")
        if not os.path.exists(midi_path):
            continue
        if not is_4_4_time(midi_path):
            stats["skipped_4_4"] += 1
            continue

        stats["tracks"] += 1
        bpm = extract_bpm_from_midi(midi_path)
        sec_per_beat = 60.0 / bpm
        measure_dur = sec_per_beat * BEATS_PER_SLICE

        info = load_stem_info(track_dir)
        if not info:
            continue

        # Load audio + compute per-measure activity for every usable stem
        audio_by_stem, sr = {}, None
        for stem_key in info:
            path = stem_wav_path(track_dir, stem_key)
            if path is None:
                continue
            audio, sr = sf.read(path)
            if audio.ndim > 1:                 # collapse stereo to mono
                audio = audio.mean(axis=1)
            audio_by_stem[stem_key] = audio.astype(np.float32)

        if sr is None or not audio_by_stem:
            continue

        samples_per_measure = int(round(measure_dur * sr))
        num_measures = min(len(a) // samples_per_measure for a in audio_by_stem.values())
        if num_measures < 1:
            continue

        active = {s: active_measures(measure_rms(a, samples_per_measure, num_measures))
                  for s, a in audio_by_stem.items()}
        active_count = np.zeros(num_measures, dtype=int)
        for s in active:
            active_count += active[s].astype(int)

        # Pick the 2 anchor measures
        anchors = []
        for region in (REGION_A, REGION_B):
            m = richest_measure_in_region(active_count, num_measures, region)
            if m is not None:
                anchors.append(m)
        anchors = sorted(set(anchors))
        if not anchors:
            continue

        slices_dir = os.path.join(track_dir, "pos_slices")
        os.makedirs(slices_dir, exist_ok=True)

        for anchor in anchors:
            active_here = [s for s in active if active[s][anchor]]
            combos = make_combos(active_here, info, rng)
            if not combos:
                continue
            stats["anchors"] += 1

            start = anchor * samples_per_measure
            end = start + samples_per_measure
            start_sec, end_sec = start / sr, end / sr

            # Slice every stem that appears in a combo at this anchor (once each)
            needed = {s for _, a, b in combos for s in (a, b)}
            slice_path = {}
            for s in needed:
                p = os.path.join(slices_dir, f"{track_name}_{s}_place{anchor:03d}.wav")
                if not os.path.exists(p):
                    sf.write(p, audio_by_stem[s][start:end], sr)
                slice_path[s] = p

            for combo_type, a, b in combos:
                rows.append({
                    "track": track_name,
                    "anchor_place": anchor,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "combo_type": combo_type,
                    "stem_A": a, "cat_A": info[a]["category"], "file_A": slice_path[a],
                    "stem_B": b, "cat_B": info[b]["category"], "file_B": slice_path[b],
                    "key": 1, "timing": 1, "tempo": 1, "mode": 1, "polyrhythm": 1,
                })
                stats["pairs"] += 1

    # Write manifest
    import csv
    if rows:
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("=" * 55)
    print(" POSITIVE PAIR GENERATION COMPLETE")
    print("=" * 55)
    print(f" 4/4 tracks processed : {stats['tracks']}")
    print(f" non-4/4 skipped      : {stats['skipped_4_4']}")
    print(f" anchors used         : {stats['anchors']}")
    print(f" positive pairs       : {stats['pairs']}")
    print(f" manifest             : {output_csv}")
    print("=" * 55)
    return rows


if __name__ == "__main__":
    build()
