"""User settings with safe persistence."""
from __future__ import annotations

import json
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
    tts_rate: int = 0
    tts_volume: int = 100
    # Display
    draw_overlay: bool = True
    show_fps: bool = True
    # Model
    active_model: str = "legacy"

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
        if self.active_model not in ("legacy", "user"):
            self.active_model = "legacy"
        return self

    def copy(self) -> "Settings":
        return Settings(**asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, values: dict[str, Any]) -> None:
        valid = {f.name for f in fields(self)}
        for key, value in values.items():
            if key in valid:
                setattr(self, key, value)
        self.sanitized()


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    settings = Settings()
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                settings.update(raw)
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
