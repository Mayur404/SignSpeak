#!/usr/bin/env bash
# One-time setup for the edge VM (Ubuntu/Debian). Safe to re-run.
#   git clone https://github.com/Mayur404/SignSpeak.git && cd SignSpeak
#   ./setup_ubuntu.sh        # installs everything, asks for camera IP + Sarvam key
#   ./start.sh               # runs in the background, prints the URL to open
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This setup script supports Ubuntu/Debian systems with apt-get." >&2
    exit 1
fi

SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi

echo "==> Installing system packages"
$SUDO apt-get update
$SUDO apt-get install -y \
    git python3 python3-venv python3-pip \
    libgl1 libegl1 libglib2.0-0 libgomp1 libfontconfig1 libdbus-1-3 \
    libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0

# MediaPipe publishes wheels for Python 3.9-3.12. Prefer a supported version.
PY=""
for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)'; then
            PY="$candidate"
            break
        fi
    fi
done
if [[ -z "$PY" ]]; then
    echo "Error: Python 3.9-3.12 is required (MediaPipe has no wheels for $(python3 --version 2>&1))." >&2
    echo "Install one, e.g.: sudo apt-get install -y python3.12 python3.12-venv" >&2
    exit 1
fi
echo "==> Using $($PY --version)"

if [[ ! -x .venv/bin/python ]]; then
    "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Configuration (.env is git-ignored, so it is created here).
[[ -f .env ]] || cp .env.example .env

set_env() {  # set_env KEY VALUE  -> replace or append in .env
    local key="$1" value="$2"
    if grep -qE "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
}

if [[ -t 0 ]]; then
    current_cam=$(grep -E '^CAMERA_HOST=' .env | cut -d= -f2- || true)
    read -r -p "Camera IP (CAMERA_HOST) [${current_cam:-10.101.0.74}]: " cam
    if [[ -n "${cam}" ]]; then set_env CAMERA_HOST "$cam"; fi

    current_key=$(grep -E '^SARVAM_API_KEY=' .env | cut -d= -f2- || true)
    if [[ -z "$current_key" ]]; then
        read -r -p "Sarvam API key for Indian-language speech (Enter to skip): " key
        if [[ -n "${key}" ]]; then set_env SARVAM_API_KEY "$key"; fi
    else
        echo "Sarvam API key already set in .env"
    fi
fi

# Let other machines on the network open the dashboard (default port 18473;
# 8080 is the NiralOS console on the edge VM).
if command -v ufw >/dev/null 2>&1 && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
    echo "==> Opening ports 18473-18475 and 28473 in the ufw firewall"
    $SUDO ufw allow 18473:18475/tcp
    $SUDO ufw allow 28473/tcp
fi

echo
echo "==> Verifying the install"
.venv/bin/python - <<'EOF'
import re
import cv2, mediapipe, numpy, sklearn, flask  # noqa: F401
from signspeak.web.server import create_app  # noqa: F401  (imports must work headless)
ffmpeg = bool(re.search(r"FFMPEG:\s+YES", cv2.getBuildInformation()))
print(f"  OpenCV {cv2.__version__} | MediaPipe {mediapipe.__version__} | RTSP/FFmpeg: {'yes' if ffmpeg else 'NO'}")
EOF

echo
echo "Setup complete. Start SignSpeak with ./start.sh"
