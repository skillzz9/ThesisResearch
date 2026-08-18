# Thesis Plan — Explainable Loop Compatibility

*Single source of truth. Consolidates all design decisions.*

---

## 1. Goal (what this thesis is)

Build an AI that takes **two music loops** and predicts **whether they are compatible** — and, crucially, **explains *why* they clash**, broken down by the underlying musical factors, instead of returning one opaque score.

- Existing tools (and prior research) return a single compatibility number. A producer who is told "72% incompatible" learns nothing and knows nothing to fix.
- This system instead reports a score **per axis** (key, timing, tempo, mode, polyrhythm), so it says *"these clash on tempo — everything else fits,"* which is both actionable (implies the fix) and educational (teaches the non-expert what's wrong).

**One-line statement:**
> Prior compatibility models output a single opaque score trained on confounded/temporal negatives. This thesis uses *controlled single-axis corruptions* to train a model that decomposes compatibility into interpretable per-axis factors and explains *why* two loops clash.

---

## 2. What makes it original

Measured against the closest prior work:

| Prior work | What they do | How this thesis differs |
|---|---|---|
| **Chen et al. 2020** (Neural Loop Combiner) | loop compatibility; same-song positives; **temporal-only** negatives (shift, reverse, rearrange); single score; CNN vs Siamese | adds harmony + tempo + note-level negatives; **controlled single-axis**; **per-axis explainable** output |
| **Huang et al. 2021** (Stem Mashups) | stem compatibility; same-song positives; key/tempo/phase negatives but **random cross-song (confounded)**; single score | **controlled** single-axis (all else held constant) vs confounded; **per-axis** output; clean stems |
| **Stem-JEPA 2024** | stem compatibility via joint-embedding; single embedding | interpretable per-axis output vs single opaque embedding |

**Core novelty:**
1. **Controlled single-axis negatives** — every negative breaks exactly one musical property while holding everything else constant → clean per-axis labels.
2. **Per-axis explainable output** — the model says *which* factor clashes (interpretability by construction).
3. **MIDI-derived note-level negatives** (mode clash, polyrhythm) — uniquely enabled by Slakh's per-stem MIDI; no prior work does this.

**Do NOT claim as novel:** same-song positives (Chen/Huang), using a CNN/Siamese (Chen), the concept of key/tempo negatives (Huang).

---

## 3. The 5 compatibility axes

Each axis is a musical factor that determines whether two loops fit, and each has a **guaranteed controlled corruption** used to generate its negatives.

| # | Axis | Dimension | What a clash means |
|---|---|---|---|
| 1 | **Key** | Harmony | wrong key — notes clash |
| 2 | **Tempo** | Rhythm | different speed — loops drift |
| 3 | **Timing** | Rhythm | same tempo, downbeats don't line up |
| 4 | **Mode** (major↔minor) | Harmony (fine) | same key, quality clashes (false relation) |
| 5 | **Polyrhythm** (3-against-4) | Rhythm (fine) | same tempo, subdivisions clash |

**Per-axis label vector:** `[key, timing, tempo, mode, polyrhythm]`, where 1 = compatible, 0 = clash.
- Positive = `[1, 1, 1, 1, 1]`
- Each negative flips exactly one value to 0.

**Honest notes:**
- Key and Mode are both *harmony* → their scores will correlate. The three rhythm axes are more distinguishable.
- Masking (frequency overlap) was **dropped** — it was the noisy, timbre-sensitive axis with no guaranteed corruption.

---

## 4. Dataset

- **Slakh2100** — multitrack dataset with per-stem audio **and per-stem MIDI**.
- Filter to **4/4 only** (~95% of tracks → ~1,950 usable songs).
- Per-stem MIDI is the key asset — it enables note-level corruptions no audio-only dataset can do.

---

## 5. Data pipeline (how a slice becomes model input)

**Key principle: MIDI is a *data-creation tool*, never a model input.**
- MIDI is used to (a) find the beat grid for slicing, and (b) create the corruptions by editing notes and rendering.
- The **model only ever sees audio-derived CQT + Chromagram.** MIDI never enters the model.
- Reasons MIDI is not a model input: real loops at deployment have no MIDI, and MIDI is the exact-notes "answer key" (would let the model cheat instead of learning from audio).

### Why we render everything from MIDI
- Mode and polyrhythm require **editing notes**, which can only be done in MIDI, then **rendered** to audio.
- To avoid a render-style confound (model cheating on timbre), **all** slices — positives and every negative — are rendered from MIDI with **one consistent synth**.
- **Cost:** synthetic (soundfont) audio quality, a real MIDI→audio render pipeline, and a synthetic→real deployment gap (mitigated below).

### The pipeline per slice
1. **Grid:** read BPM from MIDI → compute 2-bar (8-beat) measure boundaries.
2. **Slice length:** 2 bars = 8 beats (4/4 only).
3. **(Corrupt if negative):** edit the MIDI for the target axis (see §7). Positive = no edit.
4. **Render:** MIDI → audio with the consistent synth.
5. **Feature extraction:** audio → **CQT (84 × time)** + **Chromagram (12 × time)** → silence-mask → stack into 3-channel image tensor `[3, 84, T]` → save `.pt`.
6. **Pair + label:** pair the two slices' tensors with the 5-dim label.

> **Flow:** `MIDI → (edit) → render → audio → CQT+Chroma tensor → model`

---

## 6. Positive pair generation

**Definition:** two *different* stems from the **same song, same measure**, both unmodified → compatible (label `[1,1,1,1,1]`).

### Per song — 7 stem combinations
| # | Combination | Count | Selection |
|---|---|---|---|
| 1 | Drums + Bass | 1 | loudest drum + loudest bass |
| 2 | Bass + main-melody | 1 | loudest bass + main melody |
| 3 | Drums + main-melody | 1 | loudest drum + main melody |
| 4 | Melody + Melody | 4 | 4 distinct pairs, sampled uniformly at random from all C(M,2), fixed seed |

- **Main melody** = melody stem with the highest `integrated_loudness` (from metadata).
- **Melody bucket** (allow-list): Guitar, Piano, Strings, Brass, Organ, Synth Pad/Lead, Pipe, Reed, Chromatic Percussion, Ethnic. **Exclude Sound Effects.**
- Graceful degradation: skip combos when a bucket is missing; melody-melody = `min(4, C(M,2))`.

### Anchor measures — 2 per song (k = 2)
- **Region A** = first half of the song, excluding the outer ~10% (≈ 10–50%).
- **Region B** = second half, excluding the outer ~10% (≈ 50–90%).
- **Anchor** = the *richest* golden measure in each region (most active stems — found via a cheap activity scan, not full extraction).
- Only slice/extract at the 2 anchors — **do NOT slice the whole song.**

### Volume
- `7 combos × 2 places = 14 positives/song`.
- `≈ 14 × 1,950 ≈ 27,000 positive pairs`.
- Naive count to *avoid* (why we cap): `y · C(x,2)` — O(x²y) combinatorial explosion → redundancy → overfitting. We deliberately keep it O(1) per song for diversity.

---

## 7. Negative pair generation

**Core principle:** every negative = take a known-good pair and **break exactly one axis** by editing **Loop B's MIDI**, holding everything else constant. This guarantees a clean per-axis label.

Corruptions applied to **Loop B only** (Loop A unchanged):

| Axis | MIDI edit | Label | Notes |
|---|---|---|---|
| **Key** | transpose Loop B ±1 semitone | `[0,1,1,1,1]` | pitched stems only |
| **Timing** | offset Loop B by a non-metric fraction of a beat (~30–70%) | `[1,0,1,1,1]` | off-grid so clash is clear |
| **Tempo** | change Loop B's tempo ±10–20% | `[1,1,0,1,1]` | — |
| **Mode** | flip Loop B's 3rd (major↔minor) | `[1,1,1,0,1]` | pitched stems only; needs MIDI |
| **Polyrhythm** | re-quantize Loop B to a triplet grid (3-against-4) | `[1,1,1,1,0]` | mostly (not absolutely) guaranteed; needs MIDI |

### Anti-shortcut positive (required)
- **Augmented positive:** shift *both* A and B by the *same* amount (e.g. both +2 semitones). Still compatible → `[1,1,1,1,1]`.
- Prevents the model learning the trivial rule "was anything edited? = bad." Forces it to compare A *against* B.

### Balance
- Target ~**1:1** positive:negative (allow up to ~1:1.5 negative-heavy to help the weaker "clash" recall).
- Split negatives **roughly evenly across the 5 axes** so the model learns all failure modes, not one.

---

## 8. Model architecture (audio-only)

**Type:** Siamese-*encoder* network (shared weights) with a **learned per-axis comparison head** — NOT a distance readout. (This is the answer to "why won't it fail like Chen's Siamese": Chen reduced compatibility to a single distance; we keep shared encoders but replace distance with a time-preserving comparison + per-axis heads.)

### Per loop (shared encoder)
- Input: CQT + Chroma image `[3, 84, T]`.
- **CNN** (fine-tuned ResNet-18) with **frequency-only pooling** (keep the time axis — critical for tempo/timing/polyrhythm). Asymmetric strides so ~40–80 time steps survive, not ~11.
- Output: a **time-sequence** `[C, T]` per loop.

### Comparison (two paths)
- **Temporal path** — cross-attention over time between Loop A's and Loop B's sequences → **Tempo, Timing, Polyrhythm** heads. (These need *when* things happen + a comparison between the two loops.)
- **Global path** — pool the time axis → symmetric combine of A and B (`[gA+gB, |gA−gB|, gA·gB]`) → **Key, Mode** heads. (These are ~time-invariant.)

### Output
- 5 sigmoid heads → per-axis scores in [0,1].
- **Overall = min(key, timing, tempo, mode, polyrhythm)** — weakest-link; derived from the 5 so the overall is always consistent with the explanation.

### Symmetry
- Compatibility is commutative (A+B = B+A). Enforce with symmetric ops (global path) and symmetric cross-attention / both-orders training (temporal path).

---

## 9. Training

- **Loss:** multi-label binary cross-entropy — one BCE per head, summed. (No triplet loss, no hard-negative mining — simpler than the old design.)
- **Split:** by **track** (no leakage; pitch/edit variants of a stem never cross the train/test boundary). Use Slakh's official split when scaling up.
- **Threshold:** fit on train only, apply to test.

---

## 10. Deployment & the tool

- **Input at deployment:** two **real audio loops** → CQT + Chroma → model. (Real loops have no MIDI — which is exactly why the model is audio-only.)
- **Output:** overall score + 5 per-axis scores + a plain-language explanation ("clash on tempo").
- **Optional demo:** a Gradio web app showing the per-axis breakdown.
- **Optional extension:** *diagnosis-driven fix* — the per-axis result targets a standard DSP correction (key→transpose, tempo→time-stretch, timing→nudge). Frame the *diagnosis* as the contribution; the fix is standard DSP that the diagnosis targets (don't claim to invent beatmatching).

---

## 11. Known caveats / risks (be honest about these)

1. **Synthetic → real deployment gap.** Training audio is soundfont-rendered, not real instruments. Key transfers well (chroma is timbre-invariant); tempo/timing less so (transients differ). **Mitigate:** audio augmentation (reverb/EQ/noise, vary the soundfont) + **validate on a small set of real audio loop pairs**. This validation is non-negotiable — it's the proof the model deploys.
2. **Mode correlates with Key** (both harmony) — their explanations will often move together. Frame honestly as "harmony coarse vs fine."
3. **Polyrhythm label is *mostly* guaranteed**, not absolute (polyrhythm is occasionally intentional). Corrupt clearly; consider down-weighting.
4. **Render pipeline is real engineering** (build MIDI→audio renderer, render all of Slakh). This is the price of the mode + polyrhythm axes.
5. **Model must never see BPM or MIDI** — both are data-creation tools only. (Currently `temporal_analyzer.py` outputs BPM into the vector — this must be removed.)

---

## 12. Feasibility justification (why it should work)

The building blocks are individually proven:
- **Key and tempo are learnable from a spectrogram** — Schreiber & Müller, *Musical Tempo and Key Estimation using CNNs* (arXiv:1903.10839). Validates the key/tempo heads *and* the BPM-free design.
- **Beat/downbeat position learnable from spectrograms** — standard MIR (TCN/CNN beat tracking).
- **Pitch-shift and time-stretch are valid, standard transforms** — CLMR (Contrastive Learning of Musical Representations).
- **Interpretable per-dimension prediction works** — Chowdhury et al., explainable MER via mid-level features (arXiv:1907.03572).

Novel part (the contribution): combining these into a **relational, per-axis, explainable compatibility model trained on controlled single-axis corruptions.**

---

## 13. Key references

- Chen, Smith, Yang (2020) — *Neural Loop Combiner* — arXiv:2008.02011
- Huang, Wang, Smith, Song, Wang (2021) — *Modeling the Compatibility of Stem Tracks* — arXiv:2103.14208
- Stem-JEPA (2024) — arXiv:2408.02514
- Davies, Hamel, Yoshii, Goto (2014) — *AutoMashUpper* — IEEE/ACM TASLP
- Bernardes et al. — *Tonal Interval Space / Hierarchical Harmonic Mixing* (harmonic compatibility theory)
- Schreiber & Müller (2019) — *Tempo and Key Estimation using CNNs* — arXiv:1903.10839
- Chowdhury et al. (2019) — *Explainable MER via Mid-level Features* — arXiv:1907.03572

---

## 14. Build order (suggested)

1. Rewrite `generate_pairs.py` — positives (7 combos, 2 anchors) + 5 axis-labeled negatives + 5-dim labels.
2. Build the MIDI edit + render pipeline (the 5 corruptions).
3. Update `extract_features.py` — anchor selection + CQT/Chroma from rendered audio (remove BPM from the vector).
4. Update the model — single-stream, frequency-only pooling, two paths, 5 heads, min-overall.
5. Update `train.py` — multi-label BCE, remove triplet/mining.
6. Update evaluation — per-axis metrics.
7. Augmentation + real-audio validation set.
8. (Optional) Gradio demo + diagnosis-driven fix.
