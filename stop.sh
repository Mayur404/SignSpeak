#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

STOPPED=0

if [[ -f signspeak.pid ]]; then
    PID=$(cat signspeak.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping SignSpeak (PID: $PID)..."
        kill "$PID" || true
        for _ in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force terminating..."
            kill -9 "$PID" 2>/dev/null || true
        fi
        STOPPED=1
    fi
    rm -f signspeak.pid signspeak.port
fi

# Fallback for a SignSpeak process whose PID file was lost. Only stop
# app.py processes running from this folder: a bare `pkill -f app.py` would
# also kill unrelated Python apps on the shared edge VM.
HERE=$(pwd -P)
for pid in $(pgrep -f "python[0-9.]* .*app\.py" 2>/dev/null || true); do
    if [[ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" == "$HERE" ]]; then
        echo "Stopping leftover SignSpeak process (PID: $pid)..."
        kill "$pid" 2>/dev/null && STOPPED=1 || true
    fi
done

if [[ $STOPPED -eq 1 ]]; then
    echo "SignSpeak stopped successfully."
else
    echo "SignSpeak was not running."
fi
