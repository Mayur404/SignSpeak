#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "Python environment not found. Run ./setup_ubuntu.sh first." >&2
    exit 1
fi

exec "$PYTHON" app.py --host 0.0.0.0 --port 8080 "$@"
