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


# --------------------------------------------------------------------------
# Sarvam REST helpers (translate + Bulbul TTS), shared by web and desktop.
# --------------------------------------------------------------------------
import base64
import json
import re
import threading
from collections import OrderedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_SARVAM_BASE = "https://api.sarvam.ai"
# Any character from an Indian script block (Devanagari .. Malayalam).
_INDIC_SCRIPT = re.compile(r"[ऀ-ൿ]")
_CACHE_LOCK = threading.Lock()
_TRANSLATIONS: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_AUDIO: "OrderedDict[tuple[str, str, float], bytes]" = OrderedDict()
_CACHE_SIZE = 128


def _cache_get(cache: OrderedDict, key):
    with _CACHE_LOCK:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    return None


def _cache_put(cache: OrderedDict, key, value) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)


def sarvam_api_key() -> str:
    _load_project_env()
    return os.environ.get("SARVAM_API_KEY", "").strip()


def _post(path: str, body: dict, api_key: str, timeout: float) -> dict:
    request = Request(
        _SARVAM_BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
    )
    # One retry for dropped connections (common on the 5G link); HTTP errors
    # from Sarvam itself are raised straight away.
    for attempt in (1, 2):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError:
            raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def needs_translation(text: str, language: str) -> bool:
    """English (Latin) text headed for a non-English Indian voice."""
    return (
        not language.lower().startswith("en")
        and not _INDIC_SCRIPT.search(text)
        and bool(re.search(r"[A-Za-z]", text))
    )


def sarvam_translate(text: str, language: str, api_key: str, timeout: float = 20) -> str:
    """Translate English text into ``language``; returns ``text`` unchanged
    when no translation is needed. Signed words arrive in CAPS, so they are
    lower-cased first to stop them being read as acronyms."""
    if not needs_translation(text, language):
        return text
    source = text.lower()
    key = (source, language)
    cached = _cache_get(_TRANSLATIONS, key)
    if cached is not None:
        return cached
    result = _post("/translate", {
        "input": source[:1000],
        "source_language_code": "en-IN",
        "target_language_code": language,
    }, api_key, timeout)
    translated = str(result.get("translated_text") or "").strip() or text
    _cache_put(_TRANSLATIONS, key, translated)
    return translated


def sarvam_tts(text: str, language: str, pace: float, api_key: str, timeout: float = 30) -> bytes:
    """Return WAV bytes of ``text`` spoken by Sarvam Bulbul in ``language``."""
    pace = round(max(0.5, min(2.0, pace)), 2)
    key = (text, language, pace)
    cached = _cache_get(_AUDIO, key)
    if cached is not None:
        return cached
    result = _post("/text-to-speech", {
        "text": text[:2500],
        "language_code": language,
        "speaker": "shubh",
        "model": "bulbul:v3",
        "pace": pace,
        "output_audio_codec": "wav",
    }, api_key, timeout)
    audio = base64.b64decode(result["audios"][0])
    _cache_put(_AUDIO, key, audio)
    return audio
