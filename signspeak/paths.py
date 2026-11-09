"""Filesystem locations used by SignSpeak."""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = APP_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATA_DIR = APP_ROOT / "data"

LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"
LEGACY_MODEL = MODELS_DIR / "asl_legacy_rf.pkl"
ASL_CHART = ASSETS_DIR / "asl_alphabet.png"

USER_MODEL = DATA_DIR / "user_model.joblib"
USER_DATASET = DATA_DIR / "user_dataset.npz"
SETTINGS_FILE = DATA_DIR / "settings.json"
ENV_FILE = APP_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE entries from the project's .env file into os.environ."""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


load_env_file()
DATA_DIR.mkdir(parents=True, exist_ok=True)
