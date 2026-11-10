"""Sarvam speech transcription used to create custom spoken sign labels."""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .sarvam_env import _load_project_env

try:  # Qt is only needed by the desktop GUI; the web server runs without it.
    from PySide6.QtCore import QThread, Signal
except Exception:  # pragma: no cover - headless VM without Qt libraries
    class QThread:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PySide6 is required for the desktop GUI.")

    def Signal(*args, **kwargs):  # type: ignore[no-redef]
        return None


_AUDIO_TYPES = {
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
}


def transcribe_audio(path_or_bytes: Path | bytes, filename: str = "phrase.wav") -> tuple[str, str]:
    _load_project_env()
    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Add SARVAM_API_KEY to the project .env file first.")
    if isinstance(path_or_bytes, bytes):
        audio = path_or_bytes
    else:
        audio = Path(path_or_bytes).read_bytes()
    if not audio:
        raise RuntimeError("No audio was recorded. Check that a microphone is connected.")
    boundary = f"----SignSpeak{uuid.uuid4().hex}"
    parts = [
        ("model", b"saaras:v3"),
        ("mode", b"transcribe"),
        ("language_code", b"unknown"),
    ]
    body = bytearray()
    for name, value in parts:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode())
        body.extend(value + b"\r\n")
    # The name comes from the browser; keep it to a safe header token.
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(filename)).name)[:64] or "phrase.wav"
    extension = Path(filename).suffix.lower()
    content_type = _AUDIO_TYPES.get(extension, "audio/wav")

    body.extend(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
    )
    body.extend(audio + f"\r\n--{boundary}--\r\n".encode())
    request = Request(
        "https://api.sarvam.ai/speech-to-text",
        data=bytes(body),
        headers={
            "api-subscription-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Sarvam transcription failed ({exc.code}): {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Sarvam: {exc.reason}") from exc
    transcript = " ".join(str(result.get("transcript", "")).split())
    language = str(result.get("language_code") or "")
    if not transcript:
        raise RuntimeError("Sarvam did not detect any speech. Try recording again.")
    return transcript, language


class TranscriptionThread(QThread):
    completed = Signal(str, str)
    failed = Signal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            transcript, language = transcribe_audio(self.path)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(transcript, language)
