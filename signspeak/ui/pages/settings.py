from __future__ import annotations

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...config import Settings
from ...paths import DATA_DIR
from ...recognition import ModelInfo
from .. import theme
from ..icons import make_icon
from ..widgets import (
    Card,
    HLine,
    PageHeader,
    Pill,
    ToggleRow,
    ValueSlider,
    row_widget,
)


class SettingsPage(QWidget):
    changed = Signal(str, object)
    test_voice = Signal()
    reset_defaults = Signal()
    toast_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing = False
        self._voices: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        header = PageHeader(
            "Settings",
            "Tune recognition, speech and the camera. Changes are saved automatically.",
        )
        reset_button = QPushButton("Reset defaults")
        reset_button.setObjectName("Ghost")
        reset_button.setIcon(make_icon("refresh", theme.TEXT, 14))
        reset_button.clicked.connect(self.reset_defaults)
        header.add_action(reset_button)
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("Content")
        columns = QHBoxLayout(body)
        columns.setContentsMargins(0, 0, 8, 8)
        columns.setSpacing(16)
        left_col = QVBoxLayout()
        left_col.setSpacing(16)
        right_col = QVBoxLayout()
        right_col.setSpacing(16)
        columns.addLayout(left_col, 1)
        columns.addLayout(right_col, 1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # ---- camera -------------------------------------------------------
        camera_card, camera_body = self._card(
            "Camera", "Pick the webcam SignSpeak should use."
        )
        camera_row = QHBoxLayout()
        camera_caption = QLabel("Device")
        camera_caption.setObjectName("Value")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(170)
        self.camera_combo.addItem("Camera 0", 0)
        self.camera_combo.currentIndexChanged.connect(self._camera_changed)
        camera_row.addWidget(camera_caption)
        camera_row.addStretch(1)
        camera_row.addWidget(self.camera_combo)
        camera_body.addWidget(row_widget(camera_row, 36))
        width_row = QHBoxLayout()
        width_caption = QLabel("Inference width")
        width_caption.setObjectName("Value")
        self.width_combo = QComboBox()
        for width in (480, 640, 800, 960):
            self.width_combo.addItem(f"{width} px", width)
        self.width_combo.currentIndexChanged.connect(self._width_changed)
        width_row.addWidget(width_caption)
        width_row.addStretch(1)
        width_row.addWidget(self.width_combo)
        camera_body.addWidget(row_widget(width_row, 36))
        width_hint = QLabel("Lower is faster; 640 px is a good balance.")
        width_hint.setObjectName("Muted")
        camera_body.addWidget(width_hint)
        self.mirror_toggle = ToggleRow(
            "Mirror preview", "Flip the picture like a mirror. Recognition is unaffected."
        )
        self.mirror_toggle.toggled.connect(lambda v: self._emit("mirror", v))
        camera_body.addWidget(self.mirror_toggle)
        camera_body.addStretch(1)
        left_col.addWidget(camera_card)

        # ---- recognition --------------------------------------------------
        recognition_card, recognition_body = self._card(
            "Recognition", "Stable voting and hold time decide when a sign is typed."
        )
        self.hold_slider = ValueSlider(
            "Hold to type",
            200,
            2000,
            650,
            formatter=lambda v: f"{v} ms",
            subtitle="How long a sign must stay stable before it is typed.",
        )
        self.hold_slider.valueChanged.connect(lambda v: self._emit("hold_ms", v))
        recognition_body.addWidget(self.hold_slider)
        self.smoothing_slider = ValueSlider(
            "Smoothing window",
            4,
            30,
            12,
            formatter=lambda v: f"{v} frames",
            subtitle="More frames = steadier but slightly slower.",
        )
        self.smoothing_slider.valueChanged.connect(lambda v: self._emit("smoothing", v))
        recognition_body.addWidget(self.smoothing_slider)
        self.ratio_slider = ValueSlider(
            "Vote agreement",
            35,
            90,
            60,
            formatter=lambda v: f"{v}%",
            subtitle="Share of recent frames that must agree.",
        )
        self.ratio_slider.valueChanged.connect(
            lambda v: self._emit("min_vote_ratio", v / 100.0)
        )
        recognition_body.addWidget(self.ratio_slider)
        recognition_body.addStretch(1)
        right_col.addWidget(recognition_card)

        # ---- speech -------------------------------------------------------
        speech_card, speech_body = self._card(
            "Speech", "SignSpeak speaks fully offline using your system voices."
        )
        voice_row = QHBoxLayout()
        voice_caption = QLabel("Voice")
        voice_caption.setObjectName("Value")
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(230)
        self.voice_combo.addItem("System default", "")
        self.voice_combo.currentIndexChanged.connect(self._voice_changed)
        voice_row.addWidget(voice_caption)
        voice_row.addStretch(1)
        voice_row.addWidget(self.voice_combo, 1)
        speech_body.addWidget(row_widget(voice_row, 36))
        self.rate_slider = ValueSlider(
            "Speaking rate",
            -10,
            10,
            0,
            formatter=lambda v: "Normal" if v == 0 else (f"Faster +{v}" if v > 0 else f"Slower {v}"),
        )
        self.rate_slider.valueChanged.connect(lambda v: self._emit("tts_rate", v))
        speech_body.addWidget(self.rate_slider)
        self.volume_slider = ValueSlider(
            "Volume", 0, 100, 100, formatter=lambda v: f"{v}%"
        )
        self.volume_slider.valueChanged.connect(lambda v: self._emit("tts_volume", v))
        speech_body.addWidget(self.volume_slider)
        test_row = QHBoxLayout()
        test_button = QPushButton("Test voice")
        test_button.setObjectName("Ghost")
        test_button.setIcon(make_icon("volume", theme.TEXT, 14))
        test_button.clicked.connect(self.test_voice)
        test_row.addWidget(test_button)
        test_row.addStretch(1)
        speech_body.addWidget(row_widget(test_row, 36))
        speech_body.addWidget(HLine())
        self.auto_word_toggle = ToggleRow(
            "Speak words", "Say each word when you finish it with the space sign."
        )
        self.auto_word_toggle.toggled.connect(lambda v: self._emit("auto_word", v))
        speech_body.addWidget(self.auto_word_toggle)
        self.auto_sentence_toggle = ToggleRow(
            "Speak sentences", "Say the whole sentence when you add a full stop."
        )
        self.auto_sentence_toggle.toggled.connect(lambda v: self._emit("auto_sentence", v))
        speech_body.addWidget(self.auto_sentence_toggle)
        self.speak_letters_toggle = ToggleRow(
            "Speak letters", "Also pronounce every letter as it is typed."
        )
        self.speak_letters_toggle.toggled.connect(lambda v: self._emit("speak_letters", v))
        speech_body.addWidget(self.speak_letters_toggle)
        speech_body.addStretch(1)
        left_col.addWidget(speech_card)

        # ---- advanced + display ------------------------------------------
        advanced_card, advanced_body = self._card(
            "Advanced", "Fine-grained timing for the stabiliser."
        )
        self.cooldown_slider = ValueSlider(
            "Minimum gap",
            100,
            2000,
            250,
            formatter=lambda v: f"{v} ms",
            subtitle="Safety delay between two typed signs.",
        )
        self.cooldown_slider.valueChanged.connect(lambda v: self._emit("cooldown_ms", v))
        advanced_body.addWidget(self.cooldown_slider)
        self.release_slider = ValueSlider(
            "Release time",
            100,
            1000,
            320,
            formatter=lambda v: f"{v} ms",
            subtitle="Time without a stable sign before it can repeat.",
        )
        self.release_slider.valueChanged.connect(lambda v: self._emit("release_ms", v))
        advanced_body.addWidget(self.release_slider)
        advanced_body.addWidget(HLine())
        self.overlay_toggle = ToggleRow(
            "Draw hand overlay", "Show the skeleton, corner brackets and label chip."
        )
        self.overlay_toggle.toggled.connect(lambda v: self._emit("draw_overlay", v))
        advanced_body.addWidget(self.overlay_toggle)
        self.fps_toggle = ToggleRow("Show FPS", "Display the live frame rate.")
        self.fps_toggle.toggled.connect(lambda v: self._emit("show_fps", v))
        advanced_body.addWidget(self.fps_toggle)
        advanced_body.addStretch(1)
        right_col.addWidget(advanced_card)

        # ---- model + about -----------------------------------------------
        model_card, model_body = self._card(
            "Model & data", "Everything stays on this computer."
        )
        model_row = QHBoxLayout()
        self.model_pill = Pill("BUNDLED", "violet")
        model_row.addWidget(self.model_pill)
        model_row.addStretch(1)
        model_body.addWidget(row_widget(model_row, 26))
        self.model_name = QLabel("Bundled ASL model")
        self.model_name.setObjectName("H2")
        model_body.addWidget(self.model_name)
        self.model_detail = QLabel("Random Forest · 38 signs")
        self.model_detail.setObjectName("Muted")
        self.model_detail.setWordWrap(True)
        model_body.addWidget(self.model_detail)
        open_button = QPushButton("Open data folder")
        open_button.setObjectName("Ghost")
        open_button.setIcon(make_icon("folder", theme.TEXT, 14))
        open_button.clicked.connect(self._open_data)
        model_body.addWidget(open_button, 0, Qt.AlignmentFlag.AlignLeft)
        model_body.addStretch(1)
        left_col.addWidget(model_card)

        about_card, about_body = self._card(
            "About SignSpeak", "Hand signs → letters → words → speech."
        )
        about = QLabel(
            "Version 1.0 · offline desktop app<br><br>"
            "<b>Vision</b> MediaPipe Tasks Hand Landmarker<br>"
            "<b>Recognition</b> scikit-learn (personal model)<br>"
            "<b>Speech</b> Windows SAPI / pyttsx3<br>"
            "<b>UI</b> Qt for Python (PySide6)<br><br>"
            "Inspired by the open-source Sign Language to Text and Speech project."
        )
        about.setObjectName("Body")
        about.setWordWrap(True)
        about.setTextFormat(Qt.TextFormat.RichText)
        about_body.addWidget(about)
        about_body.addStretch(1)
        right_col.addWidget(about_card)
        left_col.addStretch(1)
        right_col.addStretch(1)

        self._widgets = {
            "mirror": self.mirror_toggle,
            "auto_word": self.auto_word_toggle,
            "auto_sentence": self.auto_sentence_toggle,
            "speak_letters": self.speak_letters_toggle,
            "draw_overlay": self.overlay_toggle,
            "show_fps": self.fps_toggle,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _card(title: str, subtitle: str) -> tuple[Card, QVBoxLayout]:
        card = Card(padding=18, spacing=11)
        heading = QLabel(title.upper())
        heading.setObjectName("Section")
        card.body.addWidget(heading)
        sub = QLabel(subtitle)
        sub.setObjectName("Muted")
        sub.setWordWrap(True)
        sub.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        card.body.addWidget(sub)
        card.body.addWidget(HLine())
        return card, card.body

    def _emit(self, key: str, value) -> None:
        if not self._syncing:
            self.changed.emit(key, value)

    def _camera_changed(self, index: int) -> None:
        data = self.camera_combo.itemData(index)
        if data is not None:
            self._emit("camera_index", int(data))

    def _width_changed(self, index: int) -> None:
        data = self.width_combo.itemData(index)
        if data is not None:
            self._emit("inference_width", int(data))

    def _voice_changed(self, index: int) -> None:
        data = self.voice_combo.itemData(index)
        if data is not None:
            self._emit("tts_voice", str(data))

    def _open_data(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DATA_DIR)))
        self.toast_requested.emit(f"Opened {DATA_DIR}", "info")

    # ---- public API ---------------------------------------------------
    def set_values(self, settings: Settings) -> None:
        self._syncing = True
        try:
            position = self.camera_combo.findData(settings.camera_index)
            if position >= 0:
                self.camera_combo.setCurrentIndex(position)
            position = self.width_combo.findData(settings.inference_width)
            if position >= 0:
                self.width_combo.setCurrentIndex(position)
            self.hold_slider.set_value(settings.hold_ms)
            self.smoothing_slider.set_value(settings.smoothing)
            self.ratio_slider.set_value(round(settings.min_vote_ratio * 100))
            self.cooldown_slider.set_value(settings.cooldown_ms)
            self.release_slider.set_value(settings.release_ms)
            self.rate_slider.set_value(settings.tts_rate)
            self.volume_slider.set_value(settings.tts_volume)
            self._widgets["mirror"].setChecked(settings.mirror)
            self._widgets["auto_word"].setChecked(settings.auto_word)
            self._widgets["auto_sentence"].setChecked(settings.auto_sentence)
            self._widgets["speak_letters"].setChecked(settings.speak_letters)
            self._widgets["draw_overlay"].setChecked(settings.draw_overlay)
            self._widgets["show_fps"].setChecked(settings.show_fps)
            position = self.voice_combo.findData(settings.tts_voice)
            if position >= 0:
                self.voice_combo.setCurrentIndex(position)
        finally:
            self._syncing = False

    def set_cameras(self, cameras: list[tuple[int, str]], current: int) -> None:
        self._syncing = True
        try:
            self.camera_combo.clear()
            for index, name in cameras:
                self.camera_combo.addItem(name, index)
            if not cameras:
                self.camera_combo.addItem(f"Camera {current}", current)
            position = self.camera_combo.findData(current)
            self.camera_combo.setCurrentIndex(position if position >= 0 else 0)
        finally:
            self._syncing = False

    def set_voices(self, voices: list[dict], current: str = "") -> None:
        self._voices = list(voices)
        self._syncing = True
        try:
            self.voice_combo.clear()
            self.voice_combo.addItem("System default", "")
            for voice in voices:
                name = str(voice.get("name", "Voice"))
                if len(name) > 42:
                    name = name[:39] + "…"
                self.voice_combo.addItem(name, str(voice.get("id", "")))
            position = self.voice_combo.findData(current)
            self.voice_combo.setCurrentIndex(position if position >= 0 else 0)
        finally:
            self._syncing = False

    def set_model_info(self, info: ModelInfo, active_kind: str) -> None:
        if active_kind == "user":
            self.model_pill.set_status("PERSONAL", "green")
        else:
            self.model_pill.set_status("BUNDLED", "violet")
        self.model_name.setText(info.name)
        parts = [info.algorithm]
        if info.classes:
            parts.append(f"{info.classes} signs")
        if info.has_take_validation:
            parts.append(f"{info.balanced_accuracy * 100:.1f}% balanced validation")
        elif info.is_personal:
            parts.append("Needs validation on separate recording takes")
        if info.samples:
            parts.append(f"{info.samples} samples")
        self.model_detail.setText(" · ".join(parts))
