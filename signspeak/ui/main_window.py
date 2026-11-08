from __future__ import annotations

import sys
import time
import shutil
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..composer import TextComposer
from ..config import Settings, save_settings
from ..labels import display_name, is_custom_word_label, spoken_name
from ..paths import USER_MODEL
from ..recognition import (
    ModelLoadError,
    RecognitionEngine,
    load_classifier,
    load_legacy_classifier,
    load_user_classifier,
)
from ..trainer import DatasetStore, TrainingResult
from ..tts import SpeechEngine
from ..vision import CameraProbe, CameraWorker, shutdown_tracker
from . import theme
from .icons import make_icon
from .pages.guide import GuidePage
from .pages.settings import SettingsPage
from .pages.train import TrainPage, TrainingThread
from .pages.translate import TranslatePage
from .widgets import Inset, LogoMark, Pill, Toast


def _window_icon() -> QIcon:
    pixmap = QPixmap(128, 128)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(theme.ACCENT))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(8, 8, 112, 112, 30, 30)
    painter.end()
    from .icons import icon_pixmap

    hand = icon_pixmap("hand", "#FFFFFF", 64)
    painter = QPainter(pixmap)
    painter.drawPixmap(32, 32, hand)
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, start_camera: bool = True, enable_speech: bool = True):
        super().__init__()
        self.settings = settings
        self.composer = TextComposer()
        self.store = DatasetStore()
        self.classifier = self._load_initial_classifier()
        self.speech: SpeechEngine | None = None
        self.worker: CameraWorker | None = None
        self._training_thread: TrainingThread | None = None
        self._speaking = False
        self._paused = False
        self._closing = False
        self._camera_live = False
        self._ignore_stopped_until = 0.0
        self._last_fps = 0.0
        self._cameras: list[tuple[int, str]] = [(self.settings.camera_index, f"Camera {self.settings.camera_index}")]
        self._start_camera_when_ready = start_camera
        self._camera_started = False
        self._camera_restart_pending = False
        self._probe: CameraProbe | None = None

        self.setWindowTitle("SignSpeak — Sign Language to Speech")
        self.setWindowIcon(_window_icon())
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)

        self._build_ui()
        self._connect_pages()
        self._sync_pages()
        self._install_shortcuts()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._save_settings)

        if enable_speech:
            self._start_speech()
        else:
            self._set_side(self.side_speech, "Off", theme.FAINT)

        if start_camera:
            current_cam = self.settings.camera_source if self.settings.camera_source else self.settings.camera_index
            self._probe = CameraProbe(current_cam, self)
            self._probe.finishedList.connect(self._on_cameras_found)
            self._probe.start()
            QTimer.singleShot(250, self._start_camera_if_needed)
            self.translate_page.set_camera_status("STARTING", "amber", False)
            self.train_page.set_camera_status("STARTING", "amber", False)
            self._set_side(self.side_camera, "Starting", theme.AMBER)
        else:
            self.translate_page.set_camera_status("OFFLINE", "red", False)
            self.train_page.set_camera_status("OFFLINE", "red", False)
            self._set_side(self.side_camera, "Off", theme.FAINT)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _load_initial_classifier(self):
        kind = self.settings.active_model
        try:
            return load_classifier(kind)
        except ModelLoadError:
            if kind != "legacy":
                try:
                    classifier = load_legacy_classifier()
                    self.settings.active_model = "legacy"
                    return classifier
                except ModelLoadError:
                    pass
            raise

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(central)

        # ---- sidebar ------------------------------------------------------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(240)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 22, 18, 18)
        side.setSpacing(8)

        brand = QHBoxLayout()
        brand.setSpacing(11)
        brand.addWidget(LogoMark())
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        title = QLabel("SignSpeak")
        title.setStyleSheet(f"font-size: 19px; font-weight: 700; color: {theme.TEXT};")
        subtitle = QLabel("SIGN LANGUAGE · SPEECH")
        subtitle.setStyleSheet(
            f"font-size: 9px; font-weight: 700; color: {theme.FAINT}; letter-spacing: 1.5px;"
        )
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand.addLayout(brand_text)
        brand.addStretch(1)
        side.addLayout(brand)
        side.addSpacing(18)

        section = QLabel("WORKSPACE")
        section.setObjectName("Section")
        side.addWidget(section)
        side.addSpacing(2)

        self.nav_buttons: list[QPushButton] = []
        nav_specs = [
            ("Translate", "hand"),
            ("Train", "wand"),
            ("Guide", "book"),
            ("Settings", "settings"),
        ]
        for index, (text, icon) in enumerate(nav_specs):
            button = QPushButton(text)
            button.setObjectName("Nav")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(42)
            button.setIcon(make_icon(icon, theme.MUTED, 18))
            button.setIconSize(QSize(18, 18))
            button.clicked.connect(lambda _checked=False, i=index: self.set_page(i))
            button.toggled.connect(
                lambda checked, b=button, name=icon: b.setIcon(
                    make_icon(name, theme.ACCENT if checked else theme.MUTED, 18)
                )
            )
            side.addWidget(button)
            self.nav_buttons.append(button)

        side.addSpacing(16)
        status_section = QLabel("STATUS")
        status_section.setObjectName("Section")
        side.addWidget(status_section)
        status_box = Inset(padding=12, spacing=9)
        self.side_model = self._status_row(status_box, "Model")
        self.side_camera = self._status_row(status_box, "Camera")
        self.side_speech = self._status_row(status_box, "Speech")
        side.addWidget(status_box)

        side.addSpacing(10)
        tip = Inset(padding=13, spacing=7)
        tip_title = QLabel("Personalize your signs")
        tip_title.setObjectName("Value")
        tip_text = QLabel("Teach a personal model the handshapes you use most.")
        tip_text.setObjectName("Muted")
        tip_text.setWordWrap(True)
        tip_button = QPushButton("Train a model")
        tip_button.setObjectName("Ghost")
        tip_button.setIcon(make_icon("sparkles", theme.ACCENT, 14))
        tip_button.setCursor(Qt.CursorShape.PointingHandCursor)
        tip_button.clicked.connect(lambda: self.set_page(1))
        tip.body.addWidget(tip_title)
        tip.body.addWidget(tip_text)
        tip.body.addWidget(tip_button, 0, Qt.AlignmentFlag.AlignLeft)
        side.addWidget(tip)

        side.addStretch(1)
        privacy = QLabel("●  All processing stays on this device")
        privacy.setStyleSheet(f"font-size: 9px; color: {theme.FAINT};")
        privacy.setWordWrap(True)
        side.addWidget(privacy)
        version = QLabel("v1.0 · MediaPipe + Qt")
        version.setStyleSheet(f"font-size: 9px; color: {theme.FAINT};")
        side.addWidget(version)

        layout.addWidget(sidebar)

        # ---- content ------------------------------------------------------
        self.content = QWidget()
        self.content.setObjectName("Content")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(26, 20, 26, 16)
        content_layout.setSpacing(0)
        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack)
        layout.addWidget(self.content, 1)

        self.translate_page = TranslatePage()
        self.train_page = TrainPage()
        self.guide_page = GuidePage()
        self.settings_page = SettingsPage()
        for page in (self.translate_page, self.train_page, self.guide_page, self.settings_page):
            self.stack.addWidget(page)

        self.toast = Toast(self.content)
        self.set_page(0)
        self.nav_buttons[0].setChecked(True)

    @staticmethod
    def _status_row(parent: Inset, caption: str) -> QLabel:
        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel(caption)
        label.setStyleSheet(f"font-size: 10px; color: {theme.FAINT};")
        value = QLabel("—")
        value.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {theme.MUTED};")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(value)
        parent.body.addLayout(row)
        return value

    def _connect_pages(self) -> None:
        tp = self.translate_page
        tp.manual_commit.connect(self.manual_commit)
        tp.speak_toggled.connect(self.toggle_speech)
        tp.backspace.connect(self.on_backspace)
        tp.space.connect(self.on_space)
        tp.clear_word.connect(self.on_clear_word)
        tp.delete_word.connect(self.on_delete_word)
        tp.clear_all.connect(self.on_clear_all)
        tp.pause_toggled.connect(self.on_pause)
        tp.mirror_toggled.connect(lambda value: self.update_setting("mirror", value))
        tp.camera_changed.connect(self.on_camera_changed)
        tp.retry_camera.connect(self.restart_camera)
        tp.auto_word_changed.connect(lambda value: self.update_setting("auto_word", value))
        tp.auto_sentence_changed.connect(
            lambda value: self.update_setting("auto_sentence", value)
        )
        tp.toast_requested.connect(self.show_toast)

        train = self.train_page
        train.record_requested.connect(self.start_recording)
        train.cancel_recording.connect(self.cancel_recording)
        train.capture_delay_changed.connect(
            lambda value: self.update_setting("capture_delay", value)
        )
        train.train_requested.connect(self.start_training)
        train.activate_model.connect(self.activate_model)
        train.save_model.connect(self.save_model_copy)
        train.load_model.connect(self.load_model_copy)
        train.remove_word_requested.connect(self.remove_custom_word)
        train.clear_label.connect(self.clear_label_data)
        train.clear_dataset.connect(self.clear_dataset)
        train.toast_requested.connect(self.show_toast)

        settings = self.settings_page
        settings.changed.connect(self.update_setting)
        settings.test_voice.connect(self.test_voice)
        settings.reset_defaults.connect(self.reset_defaults)
        settings.toast_requested.connect(self.show_toast)

    def _sync_pages(self) -> None:
        tp = self.translate_page
        tp.set_auto(self.settings.auto_word, self.settings.auto_sentence)
        tp.set_mirror(self.settings.mirror)
        self.settings_page.set_values(self.settings)
        self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        self.train_page.set_capture_delay(self.settings.capture_delay)
        self._sync_camera_combos()
        self._update_model_ui()
        self._update_text_ui()

    def _install_shortcuts(self) -> None:
        shortcuts = [
            ("Return", self.manual_commit),
            ("Ctrl+Return", self.toggle_speech),
            ("Backspace", self.on_backspace),
            ("Ctrl+Backspace", self.on_delete_word),
            ("Esc", self.toggle_pause),
            ("Ctrl+R", self.on_clear_all),
        ]
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def _on_cameras_found(self, cameras: list) -> None:
        if self._closing:
            return
        found = [(item[0], str(item[1])) for item in cameras]
        current = self.settings.camera_source if self.settings.camera_source else self.settings.camera_index
        keys = [item[0] for item in found]
        if current not in keys:
            label = "Sparsh CCTV" if "192.168.128.10" in str(current) else f"Camera {current}"
            found.insert(0, (current, label))
        self._cameras = found
        self._sync_camera_combos()
        self._start_camera_if_needed()

    def _sync_camera_combos(self) -> None:
        current = self.settings.camera_source if self.settings.camera_source else self.settings.camera_index
        self.translate_page.set_cameras(self._cameras, current)
        self.settings_page.set_cameras(self._cameras, current)

    def _start_camera_if_needed(self) -> None:
        if not self._start_camera_when_ready or self._camera_started or self._closing:
            return
        self._camera_started = True
        self.start_camera()

    def start_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.worker is not None:
            try:
                self.worker.wait(500)
            except Exception:
                pass
            self.worker = None
        try:
            engine = RecognitionEngine(self.classifier, self.settings)
        except Exception as exc:
            self.on_camera_error(f"Could not initialise recognition: {exc}")
            return
        self._paused = False
        self.translate_page.set_paused(False)
        self._camera_live = False
        self.translate_page.set_camera_status("STARTING", "amber", False)
        self.train_page.set_camera_status("STARTING", "amber", False)
        self._set_side(self.side_camera, "Starting", theme.AMBER)
        self.worker = CameraWorker(self.settings, engine, self)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.stateReady.connect(self._on_state)
        self.worker.committed.connect(self.on_commit)
        self.worker.fpsReady.connect(self._on_fps)
        self.worker.statusChanged.connect(self._on_camera_status)
        self.worker.error.connect(self.on_camera_error)
        self.worker.recordingProgress.connect(self.train_page.set_recording_progress)
        self.worker.recordingCountdown.connect(self.train_page.set_recording_countdown)
        self.worker.recordingFinished.connect(self.on_recording_finished)
        self.worker.start()

    def stop_camera(self) -> bool:
        if self.worker is None:
            return True
        self._ignore_stopped_until = time.monotonic() + 1.2
        worker = self.worker
        stopped = worker.stop(8000)
        if not stopped:
            return False
        self._camera_restart_pending = False
        self.worker = None
        worker.deleteLater()
        self._camera_live = False
        return True

    def restart_camera(self) -> None:
        self._start_camera_when_ready = True
        self._camera_started = True
        if not self.stop_camera():
            worker = self.worker
            if worker is not None:
                if not worker.isRunning():
                    self._camera_restart_pending = False
                    self.worker = None
                    worker.deleteLater()
                    self._camera_started = False
                    self.start_camera()
                elif not self._camera_restart_pending:
                    self._camera_restart_pending = True
                    worker.finished.connect(self._restart_after_worker_stops)
            return
        self._camera_started = False
        self.start_camera()

    def _restart_after_worker_stops(self) -> None:
        if self.sender() is not self.worker:
            return
        self._camera_restart_pending = False
        if self._closing:
            return
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        self._camera_started = False
        self.start_camera()

    def on_camera_changed(self, source: object) -> None:
        self.update_setting("camera_source", source)

    def _on_camera_status(self, status: str) -> None:
        if self._closing:
            return
        if status == "live":
            self._camera_live = True
            self.translate_page.set_camera_live()
            self.train_page.set_camera_status("LIVE", "green", True)
            self._set_side(self.side_camera, "Live", theme.GREEN)
            if self.translate_page.pause_button.isChecked():
                self.translate_page.set_paused(False)
        elif status == "stopped":
            if time.monotonic() < self._ignore_stopped_until:
                return
            self._camera_live = False
            self.translate_page.set_camera_status("OFFLINE", "red", False)
            self.train_page.set_camera_status("OFFLINE", "red", False)
            self._set_side(self.side_camera, "Offline", theme.RED)
            self.translate_page.set_fps(0)

    def on_camera_error(self, message: str) -> None:
        if self._closing:
            return
        self._camera_live = False
        self.translate_page.set_camera_error(message)
        self.train_page.set_camera_error(message)
        self._set_side(self.side_camera, "Error", theme.RED)
        self.show_toast(message, "error", 5200)

    def _on_frame(self, image) -> None:
        index = self.stack.currentIndex()
        if index == 0:
            self.translate_page.video.set_frame(image)
        elif index == 1:
            self.train_page.preview.set_frame(image)

    def _on_state(self, state) -> None:
        if self.stack.currentIndex() == 0:
            self.translate_page.set_state(state)

    def _on_fps(self, fps: float) -> None:
        self._last_fps = fps
        self.translate_page.set_fps(fps if self.settings.show_fps else 0)

    def on_pause(self, paused: bool) -> None:
        self._paused = bool(paused)
        if self.worker is not None:
            self.worker.set_paused(self._paused)
        if self._paused:
            self.train_page.cancel_recording_ui()
            if self.worker is not None:
                self.worker.cancel_recording()
        self.translate_page.set_paused(self._paused)

    def toggle_pause(self) -> None:
        if self.worker is None:
            self.show_toast("The camera is not running.", "info")
            return
        self.on_pause(not self._paused)

    # ------------------------------------------------------------------
    # Recognition → text
    # ------------------------------------------------------------------
    def on_commit(self, label: str) -> None:
        result = self.composer.commit(label)
        if result.changed:
            self._update_text_ui()
        if label and len(label) == 1 and label not in (" ", "."):
            if self.settings.speak_letters:
                self._speak(spoken_name(label))
        if result.speak_word and self.settings.auto_word:
            self._speak(result.speak_word)
        if result.speak_sentence and self.settings.auto_sentence:
            self._speak(result.speak_sentence)

    def manual_commit(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            self.show_toast("The camera is not running.", "info")
            return
        if self._paused:
            self.show_toast("Resume the camera before typing a sign.", "info")
            return
        self.worker.request_manual_commit()

    def on_backspace(self) -> None:
        if self.composer.backspace():
            self._update_text_ui()

    def on_space(self) -> None:
        result = self.composer.space()
        if result.changed:
            self._update_text_ui()
        if result.speak_word and self.settings.auto_word:
            self._speak(result.speak_word)

    def on_clear_word(self) -> None:
        if self.composer.clear_word():
            self._update_text_ui()

    def on_delete_word(self) -> None:
        if self.composer.delete_word():
            self._update_text_ui()

    def on_clear_all(self) -> None:
        if self.speech is not None:
            self.speech.stop()
        self.composer.clear_all()
        self._update_text_ui()
        self.show_toast("Transcript cleared.", "info", 1800)

    def _update_text_ui(self) -> None:
        self.translate_page.set_text(self.composer.word, self.composer.transcript)

    def toggle_speech(self) -> None:
        if self.speech is None:
            self.show_toast("Speech is not available.", "error")
            return
        if self._speaking:
            self.speech.stop()
            return
        text = self.composer.speak_text
        if not text:
            self.show_toast("There is nothing to speak yet.", "info", 2000)
            return
        self.speech.speak(text)

    def _speak(self, text: str) -> None:
        if self.speech is not None and text:
            self.speech.speak(text)

    def _start_speech(self) -> None:
        try:
            self.speech = SpeechEngine(self.settings, self)
        except Exception as exc:
            self.show_toast(f"Speech is unavailable: {exc}", "error", 5000)
            self._set_side(self.side_speech, "Unavailable", theme.RED)
            self.speech = None
            return
        self.speech.voicesReady.connect(self._on_voices)
        self.speech.speakingChanged.connect(self._on_speaking)
        self.speech.failed.connect(self._on_speech_failed)
        self._set_side(self.side_speech, "Starting", theme.AMBER)
        self.speech.start()

    def _on_voices(self, voices: list) -> None:
        available = {str(voice.get("id", "")) for voice in voices}
        voice_id = self.settings.tts_voice
        if not voice_id or voice_id not in available:
            voice_id = str(voices[0].get("id", "")) if voices else ""
            self.settings.tts_voice = voice_id
            self._schedule_save()
        if not voice_id:
            self.settings_page.set_voices([], "")
            self._set_side(self.side_speech, "Unavailable", theme.AMBER)
            return
        if self.speech is not None:
            self.speech.configure(
                voice_id=voice_id,
                rate=self.settings.tts_rate,
                volume=self.settings.tts_volume,
            )
        self.settings_page.set_voices(voices, voice_id)
        self._set_side(self.side_speech, "Ready", theme.GREEN)

    def _on_speaking(self, speaking: bool) -> None:
        self._speaking = bool(speaking)
        self.translate_page.set_speaking(self._speaking)
        self._set_side(
            self.side_speech,
            "Speaking" if self._speaking else "Ready",
            theme.ACCENT if self._speaking else theme.GREEN,
        )

    def _on_speech_failed(self, message: str) -> None:
        self.show_toast(message, "error", 5000)
        self._set_side(self.side_speech, "Unavailable", theme.RED)

    def test_voice(self) -> None:
        self._speak("Hello! I am SignSpeak. Your voice is working.")
        if self.speech is None:
            self.show_toast("Speech is unavailable.", "error")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def start_recording(self, label: str, duration: float, delay: float = 0.0) -> None:
        if self.worker is None or not self.worker.isRunning() or self._paused:
            self.show_toast("Start the camera before recording samples.", "info")
            self.train_page.set_recording_finished(label, 0)
            return
        self.train_page.set_recording_started(label, delay)
        self.worker.start_recording(label, duration, delay)

    def cancel_recording(self) -> None:
        if self.worker is not None:
            self.worker.cancel_recording()
        self.train_page.cancel_recording_ui()

    def on_recording_finished(self, label: str, samples) -> None:
        if self._closing:
            return
        added = 0
        try:
            if samples is not None and len(samples) > 0:
                added = self.store.add_samples(label, samples)
        except Exception as exc:
            self.show_toast(f"Could not save samples: {exc}", "error", 5000)
        self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        self.train_page.set_recording_finished(label, added)
        if added:
            self.show_toast(
                f"Added {added} samples for “{display_name(label)}”.", "success", 2600
            )
        else:
            self.show_toast("No hand detected — try again.", "error", 3000)

    def start_training(self) -> None:
        if self._training_thread is not None and self._training_thread.isRunning():
            return
        X, y, groups = self.store.training_data()
        if len(X) == 0:
            self.train_page.set_training_failed("No samples recorded yet.")
            self.show_toast("Record some signs before training.", "info")
            return
        self.train_page.set_training_started()
        thread = TrainingThread(X, y, groups, self)
        thread.progress.connect(self.train_page.set_training_progress)
        thread.succeeded.connect(self._on_training_success)
        thread.failed.connect(self._on_training_failed)
        thread.finished.connect(self._on_training_thread_finished)
        self._training_thread = thread
        thread.start()

    def _on_training_success(self, result: TrainingResult) -> None:
        try:
            classifier = load_user_classifier()
        except ModelLoadError as exc:
            self._on_training_failed(str(exc))
            return
        self.classifier = classifier
        self.settings.active_model = "user"
        self._save_settings()
        if self.worker is not None:
            self.worker.set_engine(RecognitionEngine(classifier, self.settings))
        self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        self.train_page.set_training_result(result)
        self._update_model_ui()
        self.show_toast(
            f"Personal model ready — {result.balanced_accuracy * 100:.1f}% balanced on {result.validation_method}.",
            "success",
            5200,
        )

    def _on_training_failed(self, message: str) -> None:
        self.train_page.set_training_failed(message)
        self.show_toast(message, "error", 6000)

    def _on_training_thread_finished(self) -> None:
        if self._training_thread is not None:
            self._training_thread.deleteLater()
            self._training_thread = None

    def activate_model(self, kind: str) -> None:
        if kind == "user" and not USER_MODEL.exists():
            self.show_toast("Train a personal model first.", "info")
            return
        if kind == self.classifier.info.kind:
            self._update_model_ui()
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            classifier = load_classifier(kind)
        except ModelLoadError as exc:
            self.show_toast(str(exc), "error", 5000)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.classifier = classifier
        self.settings.active_model = classifier.info.kind
        self._save_settings()
        if self.worker is not None:
            self.worker.set_engine(RecognitionEngine(classifier, self.settings))
        self._update_model_ui()
        self.show_toast(f"Now using {classifier.info.name}.", "success", 2600)

    def save_model_copy(self) -> None:
        if not USER_MODEL.exists():
            self.show_toast("Train a personal model before saving a copy.", "info")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save personal model", "SignSpeak-model.joblib",
            "SignSpeak model (*.joblib)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".joblib":
            target = target.with_suffix(".joblib")
        try:
            shutil.copy2(USER_MODEL, target)
        except OSError as exc:
            self.show_toast(f"Could not save model: {exc}", "error", 5000)
            return
        self.show_toast(f"Model saved to {target.name}.", "success", 3000)

    def load_model_copy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load personal model", "", "SignSpeak model (*.joblib);;All files (*)"
        )
        if not path:
            return
        source = Path(path)
        try:
            # Validate before replacing the active model so a bad file cannot
            # make the app fail on its next launch.
            load_user_classifier(source)
            USER_MODEL.parent.mkdir(parents=True, exist_ok=True)
            temporary = USER_MODEL.with_suffix(".import.joblib")
            shutil.copy2(source, temporary)
            temporary.replace(USER_MODEL)
            classifier = load_user_classifier()
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except (UnboundLocalError, OSError):
                pass
            self.show_toast(f"Could not load model: {exc}", "error", 6000)
            return
        self.classifier = classifier
        self.settings.active_model = "user"
        self._save_settings()
        if self.worker is not None:
            self.worker.set_engine(RecognitionEngine(classifier, self.settings))
        self._update_model_ui()
        self.show_toast(f"Loaded {source.name} and made it the active model.", "success", 4000)

    def _update_model_ui(self) -> None:
        info = self.classifier.info
        has_personal = USER_MODEL.exists()
        self.translate_page.set_model_info(info)
        self.train_page.set_model_info(info, has_personal, info.kind)
        self.settings_page.set_model_info(info, info.kind)
        if info.is_personal:
            accuracy = f" {info.balanced_accuracy * 100:.0f}%" if info.has_take_validation else ""
            self._set_side(self.side_model, f"Personal{accuracy}", theme.GREEN)
        else:
            self._set_side(self.side_model, "Bundled", theme.VIOLET)

    def clear_label_data(self, label: str) -> None:
        count = self.store.counts().get(label, 0)
        if not count:
            self.show_toast("No samples recorded for that sign.", "info", 2000)
            return
        answer = QMessageBox.question(
            self,
            "Clear sign data",
            f"Delete all {count} recorded samples for “{display_name(label)}”?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.clear_label(label)
        self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        self.show_toast(f"Cleared samples for “{display_name(label)}”.", "info")

    def remove_custom_word(self, label: str) -> None:
        if not is_custom_word_label(label):
            return
        count = self.store.counts().get(label, 0)
        word = display_name(label)
        answer = QMessageBox.question(
            self,
            "Remove custom word",
            f'Remove "{word}" and its {count} recorded sample(s) from the training dataset?\n\n'
            "The currently trained model keeps this word until you train it again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.clear_label(label)
        self.train_page.remove_custom_word(label)
        self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        self.show_toast(
            f'"{word}" removed from the dataset. Retrain to update the active model.',
            "success",
            4200,
        )

    def clear_dataset(self) -> None:
        if self.store.total() == 0:
            self.show_toast("The dataset is already empty.", "info", 2000)
            return
        answer = QMessageBox.question(
            self,
            "Clear dataset",
            "Delete every recorded sample? A previously trained personal model stays "
            "until you train again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.clear()
        self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        self.show_toast("Dataset cleared.", "info")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def update_setting(self, key: str, value) -> None:
        if key in ("camera_index", "camera_source"):
            if isinstance(value, str) and (value.startswith("rtsp://") or value.startswith("http://")):
                if self.settings.camera_source == value:
                    return
                self.settings.camera_source = value
            else:
                val = int(value)
                if not self.settings.camera_source and self.settings.camera_index == val:
                    return
                self.settings.camera_index = val
                self.settings.camera_source = ""
            self._save_settings()
            self._sync_camera_combos()
            self.restart_camera()
            return
        setattr(self.settings, key, value)
        self.settings.sanitized()
        self._schedule_save()

        if key in ("mirror",):
            self.translate_page.set_mirror(self.settings.mirror)
            self.settings_page.set_values(self.settings)
        if key in ("auto_word", "auto_sentence"):
            self.translate_page.set_auto(self.settings.auto_word, self.settings.auto_sentence)
            self.settings_page.set_values(self.settings)
        if key in ("tts_voice", "tts_rate", "tts_volume") and self.speech is not None:
            self.speech.configure(
                voice_id=self.settings.tts_voice,
                rate=self.settings.tts_rate,
                volume=self.settings.tts_volume,
            )
        if key == "show_fps" and not self.settings.show_fps:
            self.translate_page.set_fps(0)
        if key == "draw_overlay" and not self.settings.draw_overlay:
            pass
        if self.worker is not None:
            self.worker.apply_settings(self.settings)

    def reset_defaults(self) -> None:
        previous_camera = (self.settings.camera_source, self.settings.camera_index)
        preferred_model = "legacy"
        if USER_MODEL.exists():
            try:
                load_user_classifier()
            except ModelLoadError:
                pass
            else:
                preferred_model = "user"
        self.settings = Settings(active_model=preferred_model)
        camera_changed = previous_camera != (
            self.settings.camera_source,
            self.settings.camera_index,
        )
        self._save_settings()
        self.settings_page.set_values(self.settings)
        self.train_page.set_capture_delay(self.settings.capture_delay)
        self.translate_page.set_auto(self.settings.auto_word, self.settings.auto_sentence)
        self.translate_page.set_mirror(self.settings.mirror)
        self._sync_camera_combos()
        if self.speech is not None:
            self.speech.configure(
                voice_id=self.settings.tts_voice,
                rate=self.settings.tts_rate,
                volume=self.settings.tts_volume,
            )
        if self.worker is not None and not camera_changed:
            self.worker.apply_settings(self.settings)
        if self.classifier.info.kind != preferred_model:
            self.activate_model(preferred_model)
        else:
            self._update_model_ui()
        if camera_changed and self._start_camera_when_ready:
            self.restart_camera()
        self.show_toast("Optimal settings restored.", "success", 2600)

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _save_settings(self) -> None:
        save_settings(self.settings)

    # ------------------------------------------------------------------
    # Navigation / chrome
    # ------------------------------------------------------------------
    def set_page(self, index: int) -> None:
        if index < 0 or index >= self.stack.count():
            return
        self.stack.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.blockSignals(True)
            button.setChecked(position == index)
            button.blockSignals(False)
            button.setIcon(
                make_icon(
                    ("hand", "wand", "book", "settings")[position],
                    theme.ACCENT if position == index else theme.MUTED,
                    18,
                )
            )
        if self.worker is not None:
            self.worker.set_recognition_enabled(index == 0)
            if index != 1:
                self.worker.cancel_recording()
                self.train_page.cancel_recording_ui()
        if index == 1:
            self.train_page.set_counts(self.store.counts(), self.store.take_counts())
        if index == 0:
            self.translate_page.set_fps(self._last_fps if self.settings.show_fps else 0)

    def _set_side(self, label: QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {color};")

    def show_toast(self, text: str, tone: str = "info", duration: int = 3200) -> None:
        self.toast.show_message(text, tone, duration)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "toast"):
            self.toast.reposition()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_dark_title_done", False):
            self._dark_title_done = True
            self._set_light_titlebar()

    def _set_light_titlebar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            value = ctypes.c_int(0)
            for attribute in (20, 19):
                result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
                )
                if result == 0:
                    break
        except Exception:
            pass

    def closeEvent(self, event):  # noqa: N802
        self._closing = True
        self._save_settings()
        worker_stopped = self.stop_camera()
        probe_running = self._probe is not None and self._probe.isRunning()
        training_running = (
            self._training_thread is not None and self._training_thread.isRunning()
        )
        if not worker_stopped or probe_running or training_running:
            event.ignore()
            QTimer.singleShot(150, self._retry_close_when_threads_stop)
            return
        if self.speech is not None:
            self.speech.shutdown()
        shutdown_tracker()
        super().closeEvent(event)

    def _retry_close_when_threads_stop(self) -> None:
        if self._closing:
            self.close()
