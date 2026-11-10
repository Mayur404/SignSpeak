"""User settings with safe persistence."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .paths import SETTINGS_FILE


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    camera_source: str = ""
    mirror: bool = True
    inference_width: int = 800
    capture_delay: float = 3.0
    # Recognition
    confidence: float = 0.55
    hold_ms: int = 500
    smoothing: int = 8
    min_vote_ratio: float = 0.60
    cooldown_ms: int = 200
    release_ms: int = 250
    # Speech
    auto_word: bool = True
    auto_sentence: bool = True
    speak_letters: bool = False
    tts_voice: str = ""
    tts_provider: str = "sarvam"
    tts_language: str = "en-IN"
    tts_rate: int = 0
    tts_volume: int = 100
    # Display
    draw_overlay: bool = True
    show_fps: bool = True
    # Model
    active_model: str = "user"

    def sanitized(self) -> "Settings":
        self.camera_index = int(_clamp(self.camera_index, 0, 15))
        self.camera_source = str(getattr(self, "camera_source", "") or "").strip()
        self.inference_width = int(_clamp(self.inference_width, 320, 1280))
        self.capture_delay = float(_clamp(self.capture_delay, 0.0, 10.0))
        self.confidence = float(_clamp(self.confidence, 0.20, 0.95))
        self.hold_ms = int(_clamp(self.hold_ms, 150, 2500))
        self.smoothing = int(_clamp(self.smoothing, 4, 40))
        self.min_vote_ratio = float(_clamp(self.min_vote_ratio, 0.35, 0.95))
        self.cooldown_ms = int(_clamp(self.cooldown_ms, 100, 5000))
        self.release_ms = int(_clamp(self.release_ms, 100, 2000))
        self.tts_rate = int(_clamp(self.tts_rate, -10, 10))
        self.tts_volume = int(_clamp(self.tts_volume, 0, 100))
        if self.tts_provider not in ("local", "sarvam"):
            self.tts_provider = "local"
        # SignSpeak uses the personal classifier as its only active model.
        # Older settings that selected the bundled classifier migrate here.
        self.active_model = "user"
        return self

    def copy(self) -> "Settings":
        return Settings(**asdict(self)).sanitized()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, values: dict[str, Any]) -> None:
        """Apply ``values`` all-or-nothing.

        The candidate is validated on a copy first, so an invalid value (e.g.
        ``"abc"`` for a number) raises without leaving a half-applied,
        unsanitized field behind for the camera worker to trip over.
        """
        candidate = Settings(**asdict(self))
        for f in fields(self):
            if f.name in values:
                setattr(candidate, f.name, _coerce(values[f.name], f.default))
        candidate.sanitized()
        for f in fields(self):
            setattr(self, f.name, getattr(candidate, f.name))


_FALSE_WORDS = {"0", "false", "no", "off", ""}


def _coerce(value: Any, default: Any) -> Any:
    """Convert a raw JSON/settings value to the type of the field's default."""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() not in _FALSE_WORDS
        return bool(value)
    if isinstance(default, int):
        if isinstance(value, str):
            value = float(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Numeric settings must be finite.")
        return int(value)
    if isinstance(default, float):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Numeric settings must be finite.")
        return value
    if isinstance(default, str):
        if value is None:
            return ""
        if not isinstance(value, (str, int, float)):
            raise TypeError("Expected text.")
        return str(value)
    return value


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    settings = Settings()
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # One corrupt field must not throw away every saved setting.
                for key, value in raw.items():
                    try:
                        settings.update({key: value})
                    except (ValueError, TypeError, OverflowError):
                        pass
    except (OSError, ValueError, TypeError):
        pass
    return settings.sanitized()


def save_settings(settings: Settings, path: Path = SETTINGS_FILE) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass
