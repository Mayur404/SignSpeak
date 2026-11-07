from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...labels import display_name
from ...labels import BACKSPACE
from ...recognition import ModelInfo, RecognitionState
from .. import theme
from ..icons import make_icon
from ..widgets import (
    Card,
    ConfidenceBar,
    IconButton,
    MiniToggle,
    PageHeader,
    Pill,
    SignDial,
    VideoView,
)


class TranslatePage(QWidget):
    manual_commit = Signal()
    speak_toggled = Signal()
    backspace = Signal()
    space = Signal()
    clear_word = Signal()
    delete_word = Signal()
    clear_all = Signal()
    pause_toggled = Signal(bool)
    mirror_toggled = Signal(bool)
    camera_changed = Signal(object)
    retry_camera = Signal()
    auto_word_changed = Signal(bool)
    auto_sentence_changed = Signal(bool)
    toast_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paused = False
        self._speaking = False
        self._camera_live = False
        self._typed_until = 0.0
        self._typed_label = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        self.header = PageHeader(
            "Live translate",
            "Hold an ASL sign steady — SignSpeak types it, builds words and speaks them aloud.",
        )
        self.model_pill = Pill("Bundled model", "violet")
        self.camera_pill = Pill("STARTING", "amber")
        self.header.add_action(self.model_pill)
        self.header.add_action(self.camera_pill)
        root.addWidget(self.header)

        content = QHBoxLayout()
        content.setSpacing(18)
        root.addLayout(content, 1)

        # ------------------------------------------------------------------
        # Left: camera
        # ------------------------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(14)
        content.addLayout(left, 3)

        video_card = Card(padding=0, spacing=0)
        self.video = VideoView()
        video_card.body.addWidget(self.video, 1)
        controls = QWidget()
        controls_row = QHBoxLayout(controls)
        controls_row.setContentsMargins(16, 12, 16, 14)
        controls_row.setSpacing(8)
        self.live_pill = Pill("OFFLINE", "red")
        controls_row.addWidget(self.live_pill)
        self.fps_label = QLabel("— FPS")
        self.fps_label.setObjectName("Muted")
        controls_row.addWidget(self.fps_label)
        self.retry_button = QPushButton("Retry camera")
        self.retry_button.setObjectName("Ghost")
        self.retry_button.setIcon(make_icon("refresh", theme.TEXT, 14))
        self.retry_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retry_button.clicked.connect(self.retry_camera)
        self.retry_button.hide()
        controls_row.addWidget(self.retry_button)
        controls_row.addStretch(1)
        self.mirror_button = IconButton("flip", "Mirror the preview", 36)
        self.mirror_button.setCheckable(True)
        self.mirror_button.setChecked(True)
        self.mirror_button.toggled.connect(self.mirror_toggled)
        controls_row.addWidget(self.mirror_button)
        self.pause_button = IconButton("pause", "Pause camera (Esc)", 36)
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._on_pause)
        controls_row.addWidget(self.pause_button)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(210)
        self.camera_combo.addItem("Camera 0", 0)
        self.camera_combo.currentIndexChanged.connect(self._on_camera)
        controls_row.addWidget(self.camera_combo)
        video_card.body.addWidget(controls)
        left.addWidget(video_card, 1)

        hint_row = QHBoxLayout()
        hint_row.setContentsMargins(4, 0, 4, 0)
        hint_row.setSpacing(8)
        hint_icon = QLabel()
        hint_icon.setPixmap(make_icon("info", theme.FAINT, 14).pixmap(14, 14))
        hint_row.addWidget(hint_icon)
        hint = QLabel(
            "Keep one hand in frame, palm towards the camera. "
            "Thumbs up finishes a word · “I love you” finishes a sentence."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        hint_row.addWidget(hint, 1)
        left.addLayout(hint_row)

        # ------------------------------------------------------------------
        # Right: recognition + text
        # ------------------------------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(14)
        content.addLayout(right, 2)

        detect_card = Card(padding=18, spacing=12)
        detect_row = QHBoxLayout()
        detect_row.setSpacing(18)
        self.dial = SignDial()
        detect_row.addWidget(self.dial, 0, Qt.AlignmentFlag.AlignVCenter)
        detect_col = QVBoxLayout()
        detect_col.setSpacing(7)
        detect_col.addWidget(self._section("Detected sign"))
        self.status_label = QLabel("Starting camera…")
        self.status_label.setObjectName("H2")
        self.status_label.setWordWrap(True)
        detect_col.addWidget(self.status_label)
        self.status_hint = QLabel("Center your hand in the frame.")
        self.status_hint.setObjectName("Muted")
        self.status_hint.setWordWrap(True)
        detect_col.addWidget(self.status_hint)
        detect_col.addSpacing(4)
        conf_row = QHBoxLayout()
        conf_caption = QLabel("Top-class score")
        conf_caption.setObjectName("Muted")
        conf_caption.setToolTip(
            "The classifier's top score is diagnostic, not recognition accuracy. "
            "The app commits signs using stable votes over time."
        )
        self.conf_value = QLabel("—")
        self.conf_value.setObjectName("Accent")
        conf_row.addWidget(conf_caption)
        conf_row.addStretch(1)
        conf_row.addWidget(self.conf_value)
        detect_col.addLayout(conf_row)
        self.conf_bar = ConfidenceBar()
        detect_col.addWidget(self.conf_bar)
        detect_col.addStretch(1)
        detect_row.addLayout(detect_col, 1)
        detect_card.body.addLayout(detect_row)
        right.addWidget(detect_card)

        word_card = Card(padding=18, spacing=10)
        word_header = QHBoxLayout()
        word_header.addWidget(self._section("Current word"))
        word_header.addStretch(1)
        self.letter_count = QLabel("0 letters")
        self.letter_count.setObjectName("Muted")
        word_header.addWidget(self.letter_count)
        word_card.body.addLayout(word_header)
        self.word_field = QLineEdit()
        self.word_field.setObjectName("WordField")
        self.word_field.setReadOnly(True)
        self.word_field.setPlaceholderText("Waiting…")
        palette = self.word_field.palette()
        palette.setColor(palette.ColorRole.PlaceholderText, QColor(theme.FAINT))
        self.word_field.setPalette(palette)
        self.word_field.setMinimumHeight(46)
        word_card.body.addWidget(self.word_field)
        word_actions = QHBoxLayout()
        word_actions.setSpacing(8)
        self.backspace_button = QPushButton("Backspace")
        self.backspace_button.setObjectName("Ghost")
        self.backspace_button.setIcon(make_icon("backspace", theme.TEXT, 15))
        self.backspace_button.clicked.connect(self.backspace)
        self.space_button = QPushButton("Finish word")
        self.space_button.setObjectName("Ghost")
        self.space_button.setIcon(make_icon("space", theme.TEXT, 15))
        self.space_button.clicked.connect(self.space)
        self.clear_word_button = QPushButton("Clear")
        self.clear_word_button.setObjectName("Ghost")
        self.clear_word_button.setIcon(make_icon("trash", theme.MUTED, 14))
        self.clear_word_button.clicked.connect(self.clear_word)
        word_actions.addWidget(self.backspace_button)
        word_actions.addWidget(self.space_button)
        word_actions.addWidget(self.clear_word_button)
        word_actions.addStretch(1)
        self.type_button = QPushButton("Type sign")
        self.type_button.setObjectName("Primary")
        self.type_button.setIcon(make_icon("keyboard", "#06111D", 15))
        self.type_button.setToolTip("Type the sign you are currently holding (Enter)")
        self.type_button.clicked.connect(self.manual_commit)
        word_actions.addWidget(self.type_button)
        word_card.body.addLayout(word_actions)
        right.addWidget(word_card)

        transcript_card = Card(padding=18, spacing=10)
        transcript_header = QHBoxLayout()
        transcript_header.addWidget(self._section("Transcript"))
        transcript_header.addStretch(1)
        self.delete_word_button = QPushButton("Delete word")
        self.delete_word_button.setObjectName("Ghost")
        self.delete_word_button.setToolTip(
            "Remove the current word, or the last completed word"
        )
        self.delete_word_button.clicked.connect(self.delete_word)
        self.copy_button = IconButton("copy", "Copy transcript", 32)
        self.copy_button.clicked.connect(self._copy)
        self.clear_all_button = IconButton("trash", "Clear transcript (Ctrl+R)", 32)
        self.clear_all_button.clicked.connect(self.clear_all)
        transcript_header.addWidget(self.delete_word_button)
        transcript_header.addWidget(self.copy_button)
        transcript_header.addWidget(self.clear_all_button)
        transcript_card.body.addLayout(transcript_header)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Your sentence will appear here…")
        self.transcript.setMinimumHeight(110)
        self.transcript.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        transcript_card.body.addWidget(self.transcript, 1)
        footer = QHBoxLayout()
        footer.setSpacing(10)
        self.speak_button = QPushButton("Speak sentence")
        self.speak_button.setObjectName("Primary")
        self.speak_button.setIcon(make_icon("volume", "#06111D", 15))
        self.speak_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.speak_button.clicked.connect(self.speak_toggled)
        footer.addWidget(self.speak_button)
        footer.addStretch(1)
        self.auto_word_toggle = MiniToggle("Words")
        self.auto_word_toggle.setToolTip("Speak each finished word automatically")
        self.auto_word_toggle.toggled.connect(self.auto_word_changed)
        self.auto_sentence_toggle = MiniToggle("Sentences")
        self.auto_sentence_toggle.setToolTip("Speak each finished sentence automatically")
        self.auto_sentence_toggle.toggled.connect(self.auto_sentence_changed)
        footer.addWidget(self.auto_word_toggle)
        footer.addWidget(self.auto_sentence_toggle)
        transcript_card.body.addLayout(footer)
        right.addWidget(transcript_card, 1)

        shortcut_row = QHBoxLayout()
        shortcut_row.setContentsMargins(4, 0, 4, 0)
        shortcut_icon = QLabel()
        shortcut_icon.setPixmap(make_icon("keyboard", theme.FAINT, 14).pixmap(14, 14))
        shortcut_row.addWidget(shortcut_icon)
        shortcuts = QLabel(
            "Enter · type sign      Backspace · undo      Ctrl+Backspace · delete word      Esc · pause      Ctrl+R · reset"
        )
        shortcuts.setObjectName("Muted")
        shortcut_row.addWidget(shortcuts)
        shortcut_row.addStretch(1)
        right.addLayout(shortcut_row)

        self._refresh_word_meta()

    # ------------------------------------------------------------------
    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("Section")
        return label

    # ---- public API ---------------------------------------------------
    def set_state(self, state: RecognitionState) -> None:
        self.dial.set_state(state)
        now = time.monotonic()
        if state.committed:
            if state.committed == BACKSPACE:
                self._typed_label = ""
                self._typed_until = 0.0
            else:
                self._typed_label = display_name(state.committed)
                self._typed_until = now + 0.7

        if state.committed == BACKSPACE:
            self.status_label.setText("Backspace used")
            self.status_hint.setText("The last character or transcript token was deleted.")
        elif self._typed_until > now:
            self.status_label.setText(f"Typed “{self._typed_label}”")
            self.status_hint.setText("Keep signing, or move your hand away to repeat.")
        elif not state.hand_present:
            self.status_label.setText("Show your hand")
            self.status_hint.setText("Center one hand in the frame, about 50–70 cm away.")
        elif not state.is_stable:
            self.status_label.setText("Reading your hand…")
            self.status_hint.setText("Hold the sign still and keep it inside the frame.")
        elif not state.armed:
            self.status_label.setText("Release to repeat")
            self.status_hint.setText("Lower your hand or change sign to type this one again.")
        else:
            remaining = max(0, int((1.0 - state.hold_progress) * 0.65 * 1000))
            self.status_label.setText("Hold steady…")
            self.status_hint.setText(
                f"Keep the sign still — typing in about {remaining} ms."
                if remaining > 60
                else "Typing…"
            )

        confidence = state.confidence if state.hand_present else 0.0
        self.conf_value.setText(f"{confidence * 100:.0f}%" if state.hand_present else "—")
        self.conf_bar.set_value(confidence)
        if state.top:
            summary = "   ".join(
                f"{display_name(label)} {prob * 100:.0f}%" for label, prob in state.top[:3]
            )
            self.conf_bar.setToolTip(f"Top predictions: {summary}")

    def set_text(self, word: str, transcript: str) -> None:
        if self.word_field.text() != word:
            self.word_field.setText(word)
            self.word_field.setCursorPosition(len(word))
        if self.transcript.toPlainText() != transcript:
            self.transcript.setPlainText(transcript)
            cursor = self.transcript.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.transcript.setTextCursor(cursor)
        self._refresh_word_meta()

    def _refresh_word_meta(self) -> None:
        count = len(self.word_field.text())
        self.letter_count.setText(f"{count} letter{'s' if count != 1 else ''}")

    def set_model_info(self, info: ModelInfo) -> None:
        if info.is_personal:
            badge = (
                f"PERSONAL · {info.balanced_accuracy * 100:.0f}% BALANCED"
                if info.has_take_validation
                else "PERSONAL MODEL"
            )
            self.model_pill.set_status(
                badge, "green",
            )
            self.model_pill.setToolTip(
                f"{info.name} — {info.algorithm}\n{info.samples or 0} samples"
            )
        else:
            self.model_pill.set_status("BUNDLED MODEL", "violet")
            self.model_pill.setToolTip(
                "Pre-trained on the reference dataset. Train a personal model in "
                "“Train” for much higher accuracy with your own hand."
            )

    def set_camera_status(self, text: str, tone: str, live: bool) -> None:
        self._camera_live = live
        self.live_pill.set_status(text, tone)
        self.camera_pill.set_status(text, tone)
        if not live:
            if tone == "amber":
                self.video.set_placeholder(
                    "Starting camera…", "The first frame usually arrives within a second."
                )
                self.status_label.setText("Starting camera…")
                self.status_hint.setText("The first frame usually arrives within a second.")
            else:
                self.video.set_placeholder(
                    "Camera offline",
                    "Choose a camera and press Retry, or check that no other app is using it.",
                )
                self.status_label.setText("Camera offline")
                self.status_hint.setText(
                    "Pick a camera below and press Retry, or free the camera from another app."
                )

    def set_fps(self, fps: float) -> None:
        self.fps_label.setText(f"{fps:.0f} FPS" if fps > 0 else "— FPS")

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(paused)
        self.pause_button.blockSignals(False)
        self.pause_button.set_icon("play" if paused else "pause")
        self.pause_button.setToolTip("Resume camera (Esc)" if paused else "Pause camera (Esc)")
        if paused:
            self.live_pill.set_status("PAUSED", "amber")
            self.camera_pill.set_status("PAUSED", "amber")
        elif self._camera_live:
            self.live_pill.set_status("LIVE", "green")
            self.camera_pill.set_status("LIVE", "green")

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        self.speak_button.setText("Stop speaking" if speaking else "Speak sentence")
        self.speak_button.setIcon(
            make_icon("stop" if speaking else "volume", "#06111D", 15)
        )

    def set_auto(self, word: bool, sentence: bool) -> None:
        self.auto_word_toggle.set_checked_silent(word)
        self.auto_sentence_toggle.set_checked_silent(sentence)

    def set_mirror(self, mirror: bool) -> None:
        self.mirror_button.blockSignals(True)
        self.mirror_button.setChecked(mirror)
        self.mirror_button.blockSignals(False)

    def set_cameras(self, cameras: list[tuple[int | str, str]], current: int | str) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for key, name in cameras:
            self.camera_combo.addItem(name, key)
        if not cameras:
            self.camera_combo.addItem(f"Camera {current}", current)
        position = self.camera_combo.findData(current)
        self.camera_combo.setCurrentIndex(position if position >= 0 else 0)
        self.camera_combo.blockSignals(False)

    def set_camera_error(self, message: str) -> None:
        self._camera_live = False
        self.video.set_placeholder("Camera unavailable", message)
        self.live_pill.set_status("OFFLINE", "red")
        self.camera_pill.set_status("OFFLINE", "red")
        self.retry_button.show()

    def set_camera_live(self) -> None:
        self._camera_live = True
        self.retry_button.hide()
        self.video.set_placeholder_visible(False)
        self.live_pill.set_status("LIVE", "green")
        self.camera_pill.set_status("LIVE", "green")

    # ---- events -------------------------------------------------------
    def _on_pause(self, paused: bool) -> None:
        self.set_paused(paused)
        self.pause_toggled.emit(paused)

    def _on_camera(self, index: int) -> None:
        data = self.camera_combo.itemData(index)
        if data is not None:
            self.camera_changed.emit(data)

    def _copy(self) -> None:
        text = self.transcript.toPlainText().strip()
        if not text:
            self.toast_requested.emit("Nothing to copy yet.", "info")
            return
        QApplication.clipboard().setText(text)
        self.toast_requested.emit("Transcript copied to the clipboard.", "success")
