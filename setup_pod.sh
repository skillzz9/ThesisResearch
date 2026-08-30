#!/usr/bin/env bash
# setup_pod.sh — prepare a fresh cloud pod to build the compatibility dataset.
#
# Usage:
#   bash setup_pod.sh
#   SOUNDFONT=$PWD/soundfonts/MuseScore_General.sf3 \
#     python build_dataset.py --root /data/slakh2100 --out /data/features \
#     --manifest /data/dataset.csv --workers $(nproc)
set -e

echo "== 1. system: fluidsynth =="
SUDO=""; [ "$(id -u)" != "0" ] && SUDO="sudo"   # cloud pods are usually root with no sudo
if command -v apt-get >/dev/null; then
  $SUDO apt-get update -y && $SUDO apt-get install -y fluidsynth curl
elif command -v brew >/dev/null; then
  brew install fluid-synth curl
else
  echo "!! install fluidsynth manually (no apt-get/brew found)"; exit 1
fi
fluidsynth --version | head -1

echo "== 2. python deps =="
python -c "import torch" 2>/dev/null || pip install -q torch   # keep the pod's preinstalled CUDA torch
# PIN numpy<2: numpy 2.x against a librosa/numba built for 1.x SEGFAULTS inside
# librosa.cqt (observed on the pod: worker crashes on the first productive track).
# librosa 0.10.2 + numba>=0.59 is a known-good trio with numpy 1.26.
pip install -q "numpy<2" "librosa==0.10.2" "numba>=0.59" \
  pretty_midi soundfile pyyaml mido pandas matplotlib
python -c "import numpy,librosa,numba; print('  versions:',numpy.__version__,librosa.__version__,numba.__version__)"

echo "== 3. soundfont =="
mkdir -p soundfonts
if [ ! -f soundfonts/MuseScore_General.sf3 ]; then
  curl -sL --max-time 300 -o soundfonts/MuseScore_General.sf3 \
    "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3"
fi
ls -lh soundfonts/MuseScore_General.sf3

echo
echo "== READY =="
echo "Run the build with, e.g.:"
echo "  SOUNDFONT=\$PWD/soundfonts/MuseScore_General.sf3 \\"
echo "    python build_dataset.py --root /data/slakh2100 --out /data/features \\"
echo "    --manifest /data/dataset.csv --workers \$(nproc)"
