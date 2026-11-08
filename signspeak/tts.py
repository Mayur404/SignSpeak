"""Platform speech backends, isolated from the UI thread."""
from __future__ import annotations

import asyncio
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


EDGE_VOICES = [
    {"id": "edge:en-US-JennyNeural", "name": "Jenny · Microsoft Edge · US English"},
    {"id": "edge:en-US-AvaNeural", "name": "Ava · Microsoft Edge · US English"},
    {"id": "edge:en-US-GuyNeural", "name": "Guy · Microsoft Edge · US English"},
]


class _Pyttsx3Backend:
    """Use installed local voices on Linux and other non-Windows systems."""

    def __init__(self) -> None:
        import pyttsx3

        self.engine = pyttsx3.init()
        self.voices = [
            {"id": str(voice.id), "name": str(voice.name)}
            for voice in self.engine.getProperty("voices") or []
        ]

    def apply(self, voice_id: str, rate: int, volume: int) -> None:
        if voice_id:
            try:
                self.engine.setProperty("voice", voice_id)
            except Exception:
                pass
        self.engine.setProperty("rate", int(150 + rate * 12))
        self.engine.setProperty("volume", max(0.0, min(1.0, volume / 100.0)))

    def speak(self, text: str, cancel: threading.Event, on_start, on_done) -> None:
        try:
            if cancel.is_set():
                return
            on_start()
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            on_done()

    def close(self) -> None:
        self.engine.stop()


class _EdgeBackend:
    def __init__(self) -> None:
        from winrt.windows.media.playback import MediaPlayer, MediaPlayerAudioCategory

        self.player = MediaPlayer()
        self.player.audio_category = MediaPlayerAudioCategory.SPEECH
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ended: asyncio.Event | None = None
        self._voice_id = EDGE_VOICES[0]["id"]
        self.voices = list(EDGE_VOICES)
        self.player.add_media_ended(self._on_media_ended)

    def _on_media_ended(self, _sender, _args) -> None:
        if self._loop is not None and self._ended is not None:
            self._loop.call_soon_threadsafe(self._ended.set)

    def apply(self, voice_id: str, rate: int, volume: int) -> None:
        self._voice_id = voice_id if voice_id.startswith("edge:") else EDGE_VOICES[0]["id"]
        self.player.playback_rate = max(0.7, min(1.4, 1.0 + int(rate) * 0.035))
        self.player.volume = max(0.0, min(1.0, int(volume) / 100.0))

    def speak(self, text: str, cancel: threading.Event, on_start, on_done) -> None:
        try:
            asyncio.run(self._synthesize_and_play(text, cancel, on_start))
        finally:
            self._loop = None
            self._ended = None
            on_done()

    async def _synthesize_and_play(self, text: str, cancel: threading.Event, on_start) -> None:
        import edge_tts
        from winrt.windows.media.core import MediaSource
        from winrt.windows.storage import StorageFile

        voice = self._voice_id.removeprefix("edge:")
        with tempfile.TemporaryDirectory(prefix="signspeak-voice-") as folder:
            audio_path = Path(folder) / "speech.mp3"
            await edge_tts.Communicate(text, voice).save(str(audio_path))
            if cancel.is_set():
                return
            audio_file = await StorageFile.get_file_from_path_async(str(audio_path))
            self.player.set_source(MediaSource.create_from_storage_file(audio_file))
            self._loop = asyncio.get_running_loop()
            self._ended = asyncio.Event()
            self.player.play()
            on_start()
            while not self._ended.is_set() and not cancel.is_set():
                await asyncio.sleep(0.04)
            if cancel.is_set():
                self.player.pause()

    def close(self) -> None:
        self.player.close()


class SpeechWorker(QThread):
    voicesReady = Signal(list)
    speakingChanged = Signal(bool)
    failed = Signal(str)

    def __init__(self, voice_id: str = "", rate: int = 0, volume: int = 100, parent=None):
        super().__init__(parent)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._cancel = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._voice_id = voice_id
        self._rate = rate
        self._volume = volume

    def configure(self, voice_id: str | None = None, rate: int | None = None, volume: int | None = None) -> None:
        with self._lock:
            if voice_id is not None:
                self._voice_id = voice_id
            if rate is not None:
                self._rate = int(rate)
            if volume is not None:
                self._volume = int(volume)

    def speak(self, text: str) -> None:
        text = " ".join(str(text).split())
        if text:
            self._cancel.clear()
            self._queue.put(("say", text))

    def stop(self) -> None:
        self._cancel.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.speakingChanged.emit(False)

    def shutdown(self) -> None:
        self._cancel.set()
        self._shutdown.set()
        self.wait(15000)

    def run(self) -> None:
        try:
            if sys.platform == "win32" and sys.version_info >= (3, 11):
                import edge_tts  # noqa: F401
                backend = _EdgeBackend()
            else:
                backend = _Pyttsx3Backend()
        except Exception as exc:
            self.voicesReady.emit([])
            self.failed.emit(f"Speech is unavailable: {exc}")
            return

        self.voicesReady.emit(list(backend.voices))
        try:
            while not self._shutdown.is_set():
                try:
                    kind, payload = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if kind != "say":
                    continue
                with self._lock:
                    voice_id, rate, volume = self._voice_id, self._rate, self._volume
                try:
                    backend.apply(voice_id, rate, volume)
                    backend.speak(
                        payload,
                        self._cancel,
                        lambda: self.speakingChanged.emit(True),
                        lambda: self.speakingChanged.emit(False),
                    )
                except Exception as exc:
                    self.failed.emit(f"Speech failed: {exc}")
                    self.speakingChanged.emit(False)
        finally:
            backend.close()


class SpeechEngine(QObject):
    """Main-thread facade for platform text-to-speech."""

    voicesReady = Signal(list)
    speakingChanged = Signal(bool)
    failed = Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._worker = SpeechWorker(settings.tts_voice, settings.tts_rate, settings.tts_volume)
        self._worker.voicesReady.connect(self.voicesReady)
        self._worker.speakingChanged.connect(self.speakingChanged)
        self._worker.failed.connect(self.failed)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._worker.start()

    def speak(self, text: str) -> None:
        self._worker.speak(text)

    def stop(self) -> None:
        self._worker.stop()

    def configure(self, voice_id: str | None = None, rate: int | None = None, volume: int | None = None) -> None:
        self._worker.configure(voice_id, rate, volume)

    def shutdown(self) -> None:
        if self._started:
            self._worker.shutdown()
