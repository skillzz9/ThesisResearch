# Pod Runbook — 100-Song Slakh2100 Slice Test

Goal: extend the babyslakh PoC to ~100 real Slakh2100 songs and re-measure against the
pre-registered criteria (per axis: balanced accuracy >= 0.60, AUC >= 0.70). If the numbers
hold or rise, the full 2100-song run is justified as the thesis main experiment.

Recommended pod: 32+ CPU cores (the build is CPU-parallel rendering), 16 GB+ RAM,
~30 GB free disk. GPU optional (training is light; CUDA is auto-detected if present).

## 1. Clone + dependencies (~5 min)

```bash
git clone https://github.com/skillzz9/ThesisResearch.git && cd ThesisResearch
bash setup_pod.sh          # installs fluidsynth, python deps, downloads the soundfont
```

## 2. Get Slakh2100 MIDI + metadata WITHOUT storing the 105 GB archive

Slakh2100 (flac redux) lives at https://zenodo.org/records/4599666. We only need each
track's `MIDI/*.mid`, `all_src.mid`, and `metadata.yaml` — stream the tarball through tar
and keep only those (disk cost: a few GB; the FLAC stems are never written):

```bash
mkdir -p /data && cd /data
curl -L https://zenodo.org/record/4599666/files/slakh2100_flac_redux.tar.gz \
  | tar -xz --wildcards \
      'slakh2100_flac_redux/train/*/MIDI/*' \
      'slakh2100_flac_redux/train/*/all_src.mid' \
      'slakh2100_flac_redux/train/*/metadata.yaml'
cd -
```

Notes:
- This still *transfers* ~105 GB (gzip streams aren't seekable) but never stores it.
  On a datacentre pipe that's ~15-45 min. If bandwidth is scarce, download the tarball
  to disk instead and extract the same patterns, then delete it.
- `train/` alone contains 1500 tracks — far more than the 100 we need; the build's
  `--n-tracks 100` takes the first 100 usable (4/4) ones.

## 3. Build the 100-song dataset (~20-60 min depending on cores)

```bash
export SOUNDFONT=$PWD/soundfonts/MuseScore_General.sf3
python build_dataset.py \
  --root /data/slakh2100_flac_redux/train \
  --out features --manifest dataset.csv \
  --n-tracks 100 --workers $(nproc)
python shortcut_audit.py          # must print clean/expected tells (see repo history)
```

Expected yield: very roughly 4,000-6,000 pairs (babyslakh yielded ~45 pairs/productive
song). Timbre + phase augmentation are on by default (TIMBRE_AUG=1).

## 4. Train + measure (the same protocol as the PoC)

```bash
python train_model.py --epochs 20                          # single split, sanity + demo model
python kfold.py --k 5 --epochs 15                          # THE criteria measurement (pooled)
python scaling_curve.py --epochs 15 --sizes 10,25,50,85 --seeds 3   # scaling evidence with ERROR BARS
python predict_pair.py                                     # qualitative per-axis demo
```

Every script auto-generates its evidence graphs into `graphs/` and raw logs into `runs/`
(all thesis figures are regenerable from the logged numbers):

| Graph | Produced by | Shows |
|---|---|---|
| `graphs/epoch_curves.png`      | train_model | loss + per-axis seen/unseen accuracy across epochs |
| `graphs/seen_unseen_gap.png`   | train_model | fit vs generalisation per axis (overfit picture) |
| `graphs/roc_curves.png`        | kfold       | per-axis ROC on pooled held-out predictions |
| `graphs/criteria_scorecard.png`| kfold       | pooled acc + AUC bars vs the 0.60 / 0.70 criteria |
| `graphs/scaling_curves.png`    | scaling_curve | unseen accuracy vs #songs, mean ± std over seeds |

Raw logs: `runs/train_log.csv`, `runs/kfold_pooled.npz`, `runs/scaling_log.csv`.

## 5. What to bring back

1. The `POOLED HELD-OUT` table from `kfold.py` (per-axis acc + AUC) — score it against:
   every axis acc >= 0.60 and AUC >= 0.70; means >= 0.65 / 0.75.
2. The `TREND` block from `scaling_curve.py` (does the curve keep rising at 10->85 songs?).
3. The whole `graphs/` and `runs/` directories (thesis figures + regenerable raw numbers).
4. `nohup`/tmux logs of all runs (for the thesis appendix).

Decision gate (pre-registered):
- PASS  (all axes clear both bars)      -> run full 2100 for headline numbers
- PART  (3 of 4 axes)                   -> valid PoC + one documented scale-dependent axis
- FLAT  (numbers ~= babyslakh)          -> data was not the constraint at this scale;
                                           investigate representation/capacity before full run

## Gotchas
- If fluidsynth renders hang, the 60 s timeout guard turns them into silence and the
  build continues (fixed in `midi_utils.render_midi`).
- Memory: workers are single-threaded numpy (`OMP_NUM_THREADS=1` set in build_dataset);
  32 workers x ~1 GB is the expected envelope.
- All scripts auto-detect CUDA > MPS > CPU.
