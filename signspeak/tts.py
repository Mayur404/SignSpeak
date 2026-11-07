"""Offline text-to-speech with a dedicated worker thread.

Uses the Windows SAPI voice (instant, offline, supports immediate stop) and
falls back to pyttsx3 on other platforms or if SAPI is unavailable.
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


class _SapiBackend:
    def __init__(self) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self._pythoncom = pythoncom
        self.voice = win32com.client.Dispatch("SAPI.SpVoice")
        self.voices: list[dict[str, str]] = []
        tokens = self.voice.GetVoices()
        for index in range(tokens.Count):
            token = tokens.Item(index)
            try:
                self.voices.append({"id": token.Id, "name": token.GetDescription()})
            except Exception:
                continue
        self._tokens = tokens

    def apply(self, voice_id: str, rate: int, volume: int) -> None:
        try:
            if voice_id:
                for index in range(self._tokens.Count):
                    token = self._tokens.Item(index)
                    if token.Id == voice_id:
                        self.voice.Voice = token
                        break
        except Exception:
            pass
        try:
            self.voice.Rate = int(rate)
        except Exception:
            pass
        try:
            self.voice.Volume = int(volume)
        except Exception:
            pass

    def speak(self, text: str, cancel: threading.Event, on_start, on_done) -> None:
        try:
            self.voice.Speak(text, 1)  # async
            on_start()
            while True:
                if cancel.is_set():
                    try:
                        self.voice.Speak("", 2)  # purge
                    except Exception:
                        pass
                    break
                if self.voice.WaitUntilDone(30):
                    break
        except Exception:
            pass
        finally:
            on_done()


class _Pyttsx3Backend:
    def __init__(self) -> None:
        import pyttsx3

        self.engine = pyttsx3.init()
        self.voices = []
        for voice in self.engine.getProperty("voices") or []:
            self.voices.append({"id": str(voice.id), "name": str(voice.name)})
        self._rate = 0
        self._volume = 100

    def apply(self, voice_id: str, rate: int, volume: int) -> None:
        try:
            if voice_id:
                self.engine.setProperty("voice", voice_id)
            self.engine.setProperty("rate", int(150 + rate * 12))
            self.engine.setProperty("volume", max(0.0, min(1.0, volume / 100.0)))
        except Exception:
            pass

    def speak(self, text: str, cancel: threading.Event, on_start, on_done) -> None:
        try:
            on_start()
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass
        finally:
            on_done()


class _WinRTBackend:
    """Windows' installed natural voices, including offline Narrator voices."""

    def __init__(self) -> None:
        import asyncio

        from winrt.windows.media.playback import MediaPlayer, MediaPlayerAudioCategory
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer

        self._asyncio = asyncio
        self.synth = SpeechSynthesizer()
        self._voices = list(SpeechSynthesizer.all_voices)
        self.voices = [
            {
                "id": str(voice.id),
                "name": f"{voice.display_name} ({voice.language})",
                "language": str(voice.language),
            }
            for voice in self._voices
        ]
        self.player = MediaPlayer()
        self.player.audio_category = MediaPlayerAudioCategory.SPEECH
        self._loop = None
        self._ended_event = None
        self.player.add_media_ended(self._on_media_ended)

    def _on_media_ended(self, _sender, _args) -> None:
        if self._loop is not None and self._ended_event is not None:
            self._loop.call_soon_threadsafe(self._ended_event.set)

    def apply(self, voice_id: str, rate: int, volume: int) -> None:
        if voice_id:
            for voice in self._voices:
                if str(voice.id) == voice_id:
                    self.synth.voice = voice
                    break
        self.player.playback_rate = max(0.7, min(1.4, 1.0 + int(rate) * 0.035))
        self.player.volume = max(0.0, min(1.0, int(volume) / 100.0))

    def speak(self, text: str, cancel: threading.Event, on_start, on_done) -> None:
        async def play() -> None:
            stream = await self.synth.synthesize_text_to_stream_async(text)
            loop = self._asyncio.get_running_loop()
            ended = self._asyncio.Event()
            self._loop, self._ended_event = loop, ended
            self.player.set_stream_source(stream)
            self.player.play()
            on_start()
            while not ended.is_set() and not cancel.is_set():
                await self._asyncio.sleep(0.04)
            if cancel.is_set():
                self.player.pause()

        try:
            self._asyncio.run(play())
        finally:
            self._loop = None
            self._ended_event = None
            on_done()

    def close(self) -> None:
        try:
            self.player.close()
            self.synth.close()
        except Exception:
            pass


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
        if not text:
            return
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
        self.wait(2500)

    def run(self) -> None:
        backend = None
        if sys.platform == "win32":
            try:
                backend = _WinRTBackend()
            except Exception:
                try:
                    backend = _SapiBackend()
                except Exception:
                    backend = None
        else:
            try:
                backend = _Pyttsx3Backend()
            except Exception:
                backend = None
        if backend is None:
            self.failed.emit("Speech is unavailable. Check your installed Windows voices.")
            return
        try:
            self.voicesReady.emit(backend.voices)
        except Exception:
            pass

        while not self._shutdown.is_set():
            try:
                kind, payload = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "stop":
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
        close = getattr(backend, "close", None)
        if close is not None:
            close()


class SpeechEngine(QObject):
    """Main-thread facade for the speech worker."""

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
