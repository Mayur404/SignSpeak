#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f signspeak.pid ]] && kill -0 "$(cat signspeak.pid)" 2>/dev/null; then
    echo "SignSpeak is already running (PID: $(cat signspeak.pid))."
    echo "Access the dashboard at http://localhost:8080 or http://172.16.0.10:8080"
    exit 0
fi

PYTHON=""
if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "Error: Python 3 not found. Run ./setup_ubuntu.sh first." >&2
    exit 1
fi

nohup "$PYTHON" app.py --host 0.0.0.0 --port 8080 "$@" > signspeak.log 2>&1 &
PID=$!
echo "$PID" > signspeak.pid

sleep 1
if kill -0 "$PID" 2>/dev/null; then
    echo "=========================================================="
    echo "  SignSpeak Web UI is running in background (PID: $PID)"
    echo "=========================================================="
    echo "  * Local browser:    http://localhost:8080"
    echo "  * 5G Edge Server:   http://172.16.0.10:8080"
    echo "  * View live logs:   tail -f signspeak.log"
    echo "  * Stop server:      ./stop.sh"
    echo "=========================================================="
else
    echo "SignSpeak failed to start. Check signspeak.log:"
    cat signspeak.log
    exit 1
fi
