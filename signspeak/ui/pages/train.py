from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...labels import (
    BACKSPACE,
    SIGN_HINTS,
    TRAIN_LABELS,
    custom_word_label,
    display_name,
    is_custom_word_label,
    is_trainable_label,
)
from ...recognition import ModelInfo
from ...trainer import MAX_SAMPLES_PER_TAKE, RECOMMENDED_SAMPLES, TrainingResult, train_model
from .. import theme
from ..icons import make_icon
from ..widgets import (
    Card,
    Inset,
    LetterTile,
    PageHeader,
    Pill,
    ToggleSwitch,
    VideoView,
    row_widget,
)


class TrainingThread(QThread):
    progress = Signal(str, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, X, y, groups, parent=None):
        super().__init__(parent)
        self._X = X
        self._y = y
        self._groups = groups

    def run(self) -> None:
        try:
            result = train_model(
                self._X,
                self._y,
                progress=lambda message, percent: self.progress.emit(message, percent),
                groups=self._groups,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class TrainPage(QWidget):
    record_requested = Signal(str, float)
    cancel_recording = Signal()
    train_requested = Signal()
    activate_model = Signal(str)
    clear_label = Signal(str)
    clear_dataset = Signal()
    toast_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts: dict[str, int] = {}
        self._take_counts: dict[str, int] = {}
        self._selected = "A"
        self._recording = False
        self._training = False
        self._camera_live = False
        self._has_personal = False
        self._active_kind = "legacy"
        self._training_thread: TrainingThread | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        self.header = PageHeader(
            "Teach SignSpeak",
            "Record each sign in separate takes. Keep the handshape, and vary hand position and distance between takes.",
        )
        self.model_pill = Pill("BUNDLED MODEL", "violet")
        self.header.add_action(self.model_pill)
        self.header_train_button = QPushButton("Train model")
        self.header_train_button.setObjectName("Primary")
        self.header_train_button.setIcon(make_icon("wand", "#06111D", 15))
        self.header_train_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header_train_button.clicked.connect(self._request_training)
        self.header.add_action(self.header_train_button)
        root.addWidget(self.header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("Content")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 8, 8)
        body_layout.setSpacing(16)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        body_layout.addLayout(top_row)

        # ---- preview ------------------------------------------------------
        preview_card = Card(padding=16, spacing=10)
        preview_header = QHBoxLayout()
        preview_header.addWidget(self._section("Camera preview"))
        preview_header.addStretch(1)
        self.preview_pill = Pill("STARTING", "amber")
        preview_header.addWidget(self.preview_pill)
        preview_card.body.addLayout(preview_header)
        self.preview = VideoView()
        self.preview.setMinimumSize(380, 200)
        preview_card.body.addWidget(self.preview, 1)
        preview_hint = QLabel(
            "Keep both hands visible for two-hand signs. Use even light, and vary hand position slightly between takes."
        )
        preview_hint.setObjectName("Muted")
        preview_hint.setWordWrap(True)
        preview_card.body.addWidget(preview_hint)
        top_row.addWidget(preview_card, 3)

        # ---- record -------------------------------------------------------
        record_card = Card(padding=18, spacing=10)
        record_card.body.addWidget(self._section("Record a sign"))
        selected_row = QHBoxLayout()
        selected_row.setSpacing(14)
        self.selected_label = QLabel("A")
        self.selected_label.setStyleSheet(
            f"font-size: 48px; font-weight: 800; color: {theme.ACCENT};"
        )
        selected_row.addWidget(self.selected_label)
        selected_col = QVBoxLayout()
        selected_col.setSpacing(3)
        self.selected_name = QLabel("Sign A")
        self.selected_name.setObjectName("H2")
        self.selected_count = QLabel("0 samples")
        self.selected_count.setObjectName("Muted")
        selected_col.addWidget(self.selected_name)
        selected_col.addWidget(self.selected_count)
        selected_row.addLayout(selected_col, 1)
        record_card.body.addWidget(row_widget(selected_row, 64))
        self.selected_hint = QLabel(SIGN_HINTS["A"])
        self.selected_hint.setObjectName("Body")
        self.selected_hint.setWordWrap(True)
        self.selected_hint.setMinimumHeight(38)
        record_card.body.addWidget(self.selected_hint)
        duration_row = QHBoxLayout()
        duration_caption = QLabel("Capture length")
        duration_caption.setObjectName("Muted")
        self.duration_combo = QComboBox()
        self.duration_combo.addItem("2 seconds", 2.0)
        self.duration_combo.addItem("3 seconds", 3.0)
        self.duration_combo.addItem("4 seconds", 4.0)
        self.duration_combo.setCurrentIndex(1)
        duration_row.addWidget(duration_caption)
        duration_row.addStretch(1)
        duration_row.addWidget(self.duration_combo)
        record_card.body.addWidget(row_widget(duration_row, 36))
        self.record_button = QPushButton("Record sign")
        self.record_button.setObjectName("Primary")
        self.record_button.setIcon(make_icon("record", "#06111D", 16))
        self.record_button.setMinimumHeight(40)
        self.record_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_button.clicked.connect(self._start_recording)
        record_card.body.addWidget(self.record_button)
        self.record_progress = QProgressBar()
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(0)
        self.record_progress.hide()
        record_card.body.addWidget(self.record_progress)
        self.record_status = QLabel("Record multiple separate takes for each sign. Vary position between takes.")
        self.record_status.setObjectName("Muted")
        self.record_status.setWordWrap(True)
        record_card.body.addWidget(self.record_status)
        clear_selected = QPushButton("Clear this sign")
        clear_selected.setObjectName("Ghost")
        clear_selected.setIcon(make_icon("trash", theme.MUTED, 14))
        clear_selected.clicked.connect(lambda: self.clear_label.emit(self._selected))
        record_card.body.addWidget(clear_selected, 0, Qt.AlignmentFlag.AlignLeft)
        top_row.addWidget(record_card, 2)

        # ---- dataset ------------------------------------------------------
        dataset_card = Card(padding=18, spacing=12)
        dataset_header = QHBoxLayout()
        dataset_header.addWidget(self._section("Your dataset"))
        self.total_label = QLabel("0 samples · 0 signs")
        self.total_label.setObjectName("Muted")
        dataset_header.addWidget(self.total_label)
        dataset_header.addStretch(1)
        self.clear_all_button = QPushButton("Clear all data")
        self.clear_all_button.setObjectName("Danger")
        self.clear_all_button.setIcon(make_icon("trash", theme.RED, 14))
        self.clear_all_button.clicked.connect(self.clear_dataset)
        dataset_header.addWidget(self.clear_all_button)
        dataset_card.body.addLayout(dataset_header)

        custom_row = QHBoxLayout()
        self.custom_word_input = QLineEdit()
        self.custom_word_input.setPlaceholderText("Add a custom word or short phrase")
        self.custom_word_input.setMaxLength(48)
        self.custom_word_input.returnPressed.connect(self.add_custom_word)
        custom_row.addWidget(self.custom_word_input, 1)
        self.add_custom_word_button = QPushButton("Add word")
        self.add_custom_word_button.setObjectName("Ghost")
        self.add_custom_word_button.clicked.connect(self.add_custom_word)
        custom_row.addWidget(self.add_custom_word_button)
        dataset_card.body.addLayout(custom_row)

        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.tiles: dict[str, LetterTile] = {}
        for index, label in enumerate(TRAIN_LABELS):
            self._add_label_tile(label)
        self.tiles["A"].setChecked(True)
        for column in range(13):
            self.grid.setColumnStretch(column, 1)
        dataset_card.body.addWidget(grid_widget)
        legend = QLabel(
            f"Aim for {RECOMMENDED_SAMPLES}+ samples from at least 3 separate takes per sign."
        )
        legend.setObjectName("Muted")
        dataset_card.body.addWidget(legend)
        body_layout.addWidget(dataset_card)

        # ---- train + model ------------------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)
        body_layout.addLayout(bottom_row)

        train_card = Card(padding=18, spacing=10)
        train_card.body.addWidget(self._section("Train model"))
        train_card.body.addWidget(
            self._paragraph(
                "SignSpeak compares SVM, Extra Trees, Random Forest and K-NN. "
                "With multiple takes per sign, it validates on held-out takes before saving the best model."
            )
        )
        self.train_button = QPushButton("Train personal model")
        self.train_button.setObjectName("Primary")
        self.train_button.setIcon(make_icon("wand", "#06111D", 16))
        self.train_button.setMinimumHeight(40)
        self.train_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.train_button.clicked.connect(self._request_training)
        train_card.body.addWidget(self.train_button)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.hide()
        train_card.body.addWidget(self.train_progress)
        self.train_status = QLabel("Record at least two signs to start training.")
        self.train_status.setObjectName("Muted")
        self.train_status.setWordWrap(True)
        train_card.body.addWidget(self.train_status)
        bottom_row.addWidget(train_card, 1)

        model_card = Card(padding=18, spacing=10)
        model_header = QHBoxLayout()
        model_header.addWidget(self._section("Model in use"))
        model_header.addStretch(1)
        self.active_pill = Pill("BUNDLED", "violet")
        model_header.addWidget(self.active_pill)
        model_card.body.addLayout(model_header)
        self.model_name = QLabel("Bundled ASL model")
        self.model_name.setObjectName("H2")
        model_card.body.addWidget(self.model_name)
        self.model_detail = QLabel("Pre-trained · 38 signs")
        self.model_detail.setObjectName("Muted")
        self.model_detail.setWordWrap(True)
        model_card.body.addWidget(self.model_detail)
        self.model_accuracy = QLabel("")
        self.model_accuracy.setStyleSheet(
            f"font-size: 24px; font-weight: 800; color: {theme.GREEN};"
        )
        model_card.body.addWidget(self.model_accuracy)
        button_row = QHBoxLayout()
        self.use_personal = QPushButton("Use personal")
        self.use_personal.setObjectName("Primary")
        self.use_personal.clicked.connect(lambda: self.activate_model.emit("user"))
        self.use_bundled = QPushButton("Use bundled")
        self.use_bundled.setObjectName("Ghost")
        self.use_bundled.clicked.connect(lambda: self.activate_model.emit("legacy"))
        button_row.addWidget(self.use_personal)
        button_row.addWidget(self.use_bundled)
        model_card.body.addLayout(button_row)
        self.model_note = QLabel(
            "Personal models are far more accurate for your own hand. "
            "The bundled model is a starting point trained by the original author."
        )
        self.model_note.setObjectName("Muted")
        self.model_note.setWordWrap(True)
        model_card.body.addWidget(self.model_note)
        bottom_row.addWidget(model_card, 1)

        self._refresh()
        self.set_model_info(
            ModelInfo("legacy", "Bundled ASL model", "legacy42", "Random Forest"), False, "legacy"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("Section")
        return label

    @staticmethod
    def _paragraph(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Muted")
        label.setWordWrap(True)
        return label

    def _select(self, label: str) -> None:
        self._selected = label
        self._refresh()

    def _add_label_tile(self, label: str) -> None:
        if label in self.tiles:
            return
        if label == BACKSPACE:
            hint = "Custom control handshape: deletes the last character or transcript token."
        elif is_custom_word_label(label):
            hint = "Custom handshape: inserts this word or phrase with word spacing."
        else:
            hint = SIGN_HINTS.get(label, "")
        tile = LetterTile(label, hint)
        self.group.addButton(tile)
        self.tiles[label] = tile
        index = len(self.tiles) - 1
        self.grid.addWidget(tile, index // 13, index % 13)
        tile.clicked.connect(lambda _checked=False, value=label: self._select(value))

    def add_custom_word(self) -> None:
        word = " ".join(self.custom_word_input.text().split()).upper()
        label = custom_word_label(word)
        if not is_trainable_label(label):
            self.toast_requested.emit(
                "Enter a word or short phrase using letters, numbers, spaces, apostrophes or hyphens.",
                "error",
            )
            return
        self._add_label_tile(label)
        self.custom_word_input.clear()
        self.tiles[label].setChecked(True)
        self._select(label)

    def _refresh(self) -> None:
        count = self._counts.get(self._selected, 0)
        name = display_name(self._selected)
        size = 48 if len(name) == 1 else (28 if len(name) <= 3 else 22)
        self.selected_label.setStyleSheet(
            f"font-size: {size}px; font-weight: 800; color: {theme.ACCENT};"
        )
        self.selected_label.setText(name)
        self.selected_name.setText(f"Sign {name}")
        if self._selected == BACKSPACE:
            hint = "Record a distinct handshape for this control. It deletes the last character or transcript token."
        elif is_custom_word_label(self._selected):
            hint = "Record a distinct handshape. When recognized, this word or phrase is inserted with a word boundary."
        else:
            hint = SIGN_HINTS.get(self._selected, "")
        self.selected_hint.setText(hint)
        self.selected_count.setText(
            f"{count} samples · {self._take_counts.get(self._selected, 0)} takes · "
            f"target {RECOMMENDED_SAMPLES}"
        )
        total = sum(self._counts.values())
        signs = sum(1 for value in self._counts.values() if value > 0)
        self.total_label.setText(
            f"{total} sample{'s' if total != 1 else ''} · {signs} sign{'s' if signs != 1 else ''}"
        )
        ready_signs = sum(1 for value in self._counts.values() if value >= 5)
        can_train = ready_signs >= 2 and total >= 20
        self.train_button.setEnabled(can_train and not self._training)
        self.header_train_button.setEnabled(can_train and not self._training)
        if not self._training:
            if can_train:
                self.train_status.setText(
                    f"Ready to train on {ready_signs} signs and {total} samples."
                )
            else:
                self.train_status.setText(
                    "Record at least two signs (5+ samples each) to start training."
                )
        self.record_button.setEnabled(
            self._camera_live and not self._recording and not self._training
        )

    # ---- public API ---------------------------------------------------
    def set_counts(
        self, counts: dict[str, int], take_counts: dict[str, int] | None = None
    ) -> None:
        self._counts = dict(counts)
        self._take_counts = dict(take_counts or {})
        for label in self._counts:
            if is_custom_word_label(label):
                self._add_label_tile(label)
        for label, tile in self.tiles.items():
            tile.set_count(self._counts.get(label, 0))
        self._refresh()

    def set_frame(self, image) -> None:
        if self.isVisible():
            self.preview.set_frame(image)

    def set_camera_status(self, text: str, tone: str, live: bool) -> None:
        self._camera_live = live
        self.preview_pill.set_status(text, tone)
        if not live:
            if tone == "amber":
                self.preview.set_placeholder("Starting camera…", "Hang on a moment.")
            else:
                self.preview.set_placeholder(
                    "Camera offline", "Start the camera from the Translate page."
                )
        else:
            self.preview.set_placeholder_visible(False)
        self._refresh()

    def set_camera_error(self, message: str) -> None:
        self._camera_live = False
        self.preview.set_placeholder("Camera unavailable", message)
        self.preview_pill.set_status("OFFLINE", "red")
        self._refresh()

    def set_recording_started(self, label: str) -> None:
        self._recording = True
        self.record_button.setText(f"Recording {display_name(label)}…")
        self.record_button.setEnabled(False)
        self.record_progress.setValue(0)
        self.record_progress.show()
        self.record_status.setText("Hold this sign steady. Vary hand position between separate takes.")
        self.duration_combo.setEnabled(False)
        self._refresh()

    def set_recording_progress(self, label: str, count: int) -> None:
        self.record_progress.setValue(min(100, int(count / MAX_SAMPLES_PER_TAKE * 100)))
        self.record_status.setText(
            f"Recording {display_name(label)} — {count} frames captured…"
        )

    def set_recording_finished(self, label: str, added: int) -> None:
        self._recording = False
        self.record_button.setText("Record sign")
        self.record_button.setEnabled(self._camera_live and not self._training)
        self.record_progress.hide()
        self.duration_combo.setEnabled(True)
        if added:
            self.record_status.setText(
                f"Added {added} samples for “{display_name(label)}”. "
                "Record more signs, or train when ready."
            )
        else:
            self.record_status.setText(
                "No hand was detected. Check the preview and try again."
            )
        self._refresh()

    def set_training_started(self) -> None:
        self._training = True
        self.train_button.setEnabled(False)
        self.header_train_button.setEnabled(False)
        self.train_progress.setValue(0)
        self.train_progress.show()
        self.train_status.setText("Training…")
        self.record_button.setEnabled(False)

    def set_training_progress(self, message: str, percent: int) -> None:
        self.train_progress.setValue(max(0, min(100, percent)))
        self.train_status.setText(message)

    def set_training_result(self, result: TrainingResult) -> None:
        self._training = False
        self._refresh()
        self.train_progress.setValue(100)
        self.train_status.setText(
            f"Done — {result.algorithm}: {result.balanced_accuracy * 100:.1f}% balanced, "
            f"{result.accuracy * 100:.1f}% raw on {result.validation_method} "
            f"({result.samples} samples)."
        )
        method = "take holdout" if result.validation_method == "separate recording takes" else "sample split"
        self.model_accuracy.setText(
            f"{result.balanced_accuracy * 100:.1f}% balanced · "
            f"{result.accuracy * 100:.1f}% raw · {method}"
        )

    def set_training_failed(self, message: str) -> None:
        self._training = False
        self.train_progress.hide()
        self._refresh()
        self.train_status.setText(message)

    def cancel_recording_ui(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self.record_button.setText("Record sign")
        self.record_progress.hide()
        self.duration_combo.setEnabled(True)
        self.record_status.setText("Recording cancelled.")
        self._refresh()

    def set_model_info(self, info: ModelInfo, has_personal: bool, active_kind: str) -> None:
        self._has_personal = has_personal
        self._active_kind = active_kind
        if active_kind == "user":
            self.model_pill.set_status("PERSONAL MODEL", "green")
            self.active_pill.set_status("PERSONAL", "green")
        else:
            self.model_pill.set_status("BUNDLED MODEL", "violet")
            self.active_pill.set_status("BUNDLED", "violet")
        self.model_name.setText(info.name)
        parts = [info.algorithm]
        if info.classes:
            parts.append(f"{info.classes} signs")
        if info.samples:
            parts.append(f"{info.samples} samples")
        self.model_detail.setText(" · ".join(parts))
        if info.has_take_validation:
            method = "take holdout" if info.validation_method == "separate recording takes" else "sample split"
            self.model_accuracy.setText(f"{info.balanced_accuracy * 100:.1f}% balanced · {method}")
        elif info.is_personal:
            self.model_accuracy.setText("Needs validation on separate recording takes")
        else:
            self.model_accuracy.setText("")
        self.use_personal.setEnabled(has_personal and active_kind != "user")
        self.use_bundled.setEnabled(active_kind != "legacy")
        if has_personal:
            self.use_personal.setObjectName(
                "Primary" if active_kind != "user" else "Ghost"
            )
        else:
            self.use_personal.setObjectName("Ghost")
        self.use_personal.style().unpolish(self.use_personal)
        self.use_personal.style().polish(self.use_personal)

    def set_model_switching(self, switching: bool) -> None:
        self.use_personal.setEnabled(
            not switching and self._has_personal and self._active_kind != "user"
        )
        self.use_bundled.setEnabled(not switching and self._active_kind != "legacy")

    # ---- events -------------------------------------------------------
    def _start_recording(self) -> None:
        if self._recording or not self._camera_live:
            return
        duration = float(self.duration_combo.currentData() or 3.0)
        self.record_requested.emit(self._selected, duration)

    def _request_training(self) -> None:
        if self._training:
            return
        self.train_requested.emit()

    def shutdown(self) -> None:
        if self._training_thread is not None and self._training_thread.isRunning():
            self._training_thread.wait(4000)
