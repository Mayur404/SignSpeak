#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This setup script supports Ubuntu/Debian systems with apt-get." >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y \
    git python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
    libxkbcommon-x11-0 espeak-ng libespeak1

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "Setup complete. Start SignSpeak with ./run.sh"
