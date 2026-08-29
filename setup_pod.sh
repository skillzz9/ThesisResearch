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
if command -v apt-get >/dev/null; then
  sudo apt-get update -y && sudo apt-get install -y fluidsynth curl
elif command -v brew >/dev/null; then
  brew install fluid-synth curl
else
  echo "!! install fluidsynth manually (no apt-get/brew found)"; exit 1
fi
fluidsynth --version | head -1

echo "== 2. python deps =="
pip install -q pretty_midi librosa soundfile numpy pyyaml mido torch torchvision pandas matplotlib

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
