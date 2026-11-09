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

# Fallback check for any dangling app.py processes
pkill -f "python.*app.py" 2>/dev/null && STOPPED=1 || true

if [[ $STOPPED -eq 1 ]]; then
    echo "SignSpeak stopped successfully."
else
    echo "SignSpeak was not running."
fi
