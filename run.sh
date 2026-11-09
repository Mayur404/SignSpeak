#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python ]]; then
    echo "Python environment not found. Run ./setup_ubuntu.sh first." >&2
    exit 1
fi

exec .venv/bin/python app.py "$@"
