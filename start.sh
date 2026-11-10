#!/usr/bin/env bash
# Start SignSpeak in the background. Safe to run right after `git clone`:
# it creates .env, installs dependencies on first run, opens the port,
# and connects to the RTSP camera on CAMERA_HOST automatically.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# shellcheck source=scripts/port.sh
source scripts/port.sh

# HTTPS (needed for the mic from other machines): ./start.sh --https or WEB_HTTPS=1 in .env
SCHEME=http
if [[ " $* " == *" --https "* ]] || grep -qiE '^WEB_HTTPS=(1|true|yes|on)\s*$' .env 2>/dev/null; then
    SCHEME=https
fi

if [[ -f signspeak.pid ]] && kill -0 "$(cat signspeak.pid)" 2>/dev/null; then
    PORT=$(cat signspeak.port 2>/dev/null || echo 18473)
    echo "SignSpeak is already running (PID: $(cat signspeak.pid))."
    for ip in $(hostname -I 2>/dev/null); do echo "  $SCHEME://$ip:$PORT"; done
    exit 0
fi

# 1. Config: create .env from the template on a fresh clone.
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example (CAMERA_HOST=$(grep -E '^CAMERA_HOST=' .env | cut -d= -f2))."
    echo "Edit .env to change the camera IP or add SARVAM_API_KEY."
fi

if ! grep -qE '^SARVAM_API_KEY=.+' .env; then
    echo "Note: SARVAM_API_KEY is empty in .env - Indian-language speech (Sarvam) is off;"
    echo "      English uses the browser's Microsoft voice. Add the key to .env to enable it."
fi

# 2. Python env: run setup automatically the first time.
if [[ ! -x .venv/bin/python ]]; then
    echo "First run: installing dependencies (this takes a few minutes)..."
    ./setup_ubuntu.sh
fi
PYTHON=.venv/bin/python

PORT=$(pick_port)
echo "$PORT" > signspeak.port

# 3. Firewall: let other machines on the network reach the dashboard.
if command -v ufw >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
        if sudo -n ufw status 2>/dev/null | grep -q "Status: active"; then
            sudo -n ufw allow "$PORT"/tcp >/dev/null 2>&1 || true
        fi
    elif ! ufw status 2>/dev/null | grep -qE "^${PORT}(/tcp)?\s+ALLOW"; then
        echo "Tip: if the dashboard does not open from your PC, allow the port once with:"
        echo "     sudo ufw allow ${PORT}/tcp"
    fi
fi

nohup "$PYTHON" app.py --host 0.0.0.0 --port "$PORT" --no-browser "$@" > signspeak.log 2>&1 &
PID=$!
echo "$PID" > signspeak.pid

# Wait (up to 30 s) for the startup banner; loading the model takes a moment.
for _ in $(seq 1 60); do
    grep -q "Camera:" signspeak.log 2>/dev/null && break
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$PID" 2>/dev/null; then
    echo "=========================================================="
    echo "  SignSpeak Web UI is running (PID: $PID)"
    echo "=========================================================="
    echo "  Open one of these from your PC's browser:"
    for ip in $(hostname -I 2>/dev/null); do echo "    $SCHEME://$ip:$PORT"; done
    [[ "$SCHEME" == https ]] && echo "  (Self-signed certificate: click Advanced -> Proceed once in the browser.)"
    echo "  Camera: $(grep -m1 'Camera:' signspeak.log | sed 's/.*Camera: *//' || true)"
    echo "  Live logs:  tail -f signspeak.log"
    echo "  Stop:       ./stop.sh"
    echo "=========================================================="
else
    echo "SignSpeak failed to start. Check signspeak.log:"
    cat signspeak.log
    exit 1
fi
