"""Filesystem locations used by SignSpeak."""
from __future__ import annotations

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

DATA_DIR.mkdir(parents=True, exist_ok=True)
