#!/usr/bin/env bash
# Run SignSpeak in the foreground (Ctrl+C to stop).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

[[ -f .env ]] || cp .env.example .env
[[ -x .venv/bin/python ]] || ./setup_ubuntu.sh

source scripts/port.sh
PORT=$(pick_port)
echo "Open http://<this-machine-ip>:$PORT from your PC (Ctrl+C to stop)."
exec .venv/bin/python app.py --host 0.0.0.0 --port "$PORT" --no-browser "$@"
