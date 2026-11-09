"""Qt-free Sarvam constants and .env loading.

Kept separate from ``tts.py`` (which needs PySide6.QtMultimedia) so the
headless web server on the edge VM never has to import Qt.
"""
from __future__ import annotations

import os

from .paths import APP_ROOT

SARVAM_LANGUAGES = {
    "Bengali": "bn-IN", "English (India)": "en-IN", "Gujarati": "gu-IN",
    "Hindi": "hi-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN",
    "Marathi": "mr-IN", "Odia": "od-IN", "Punjabi": "pa-IN",
    "Tamil": "ta-IN", "Telugu": "te-IN",
}


def _load_project_env() -> None:
    """Load simple KEY=VALUE entries from the project's ignored .env file."""
    env_file = APP_ROOT / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() and key.strip() not in os.environ:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key.strip()] = value
