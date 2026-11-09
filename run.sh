#!/usr/bin/env bash
# Run SignSpeak in the foreground (Ctrl+C to stop).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

[[ -f .env ]] || cp .env.example .env
[[ -x .venv/bin/python ]] || ./setup_ubuntu.sh

exec .venv/bin/python app.py --host 0.0.0.0 --port "${WEB_PORT:-8080}" --no-browser "$@"
