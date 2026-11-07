"""Reusable, hand-painted widgets for the SignSpeak interface."""
from __future__ import annotations

import math

from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..labels import BACKSPACE, display_name, is_custom_word_label
from ..recognition import RecognitionState
from . import theme
from .icons import icon_pixmap, make_icon


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------
class Card(QFrame):
    def __init__(self, parent=None, padding: int = 20, spacing: int = 14):
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)


class Inset(QFrame):
    def __init__(self, parent=None, padding: int = 14, spacing: int = 10):
        super().__init__(parent)
        self.setObjectName("Inset")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)


class HLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        text_col = QVBoxLayout()
        text_col.setSpacing(3)
        self.title = QLabel(title)
        self.title.setObjectName("H1")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("Muted")
        self.subtitle.setWordWrap(True)
        text_col.addWidget(self.title)
        text_col.addWidget(self.subtitle)
        row.addLayout(text_col, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)

    def add_action(self, widget: QWidget) -> None:
        self.actions.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)


class Pill(QLabel):
    DOT_COLORS = {
        "accent": theme.ACCENT,
        "green": theme.GREEN,
        "amber": theme.AMBER,
        "red": theme.RED,
        "violet": "#C4B5FD",
        "muted": theme.MUTED,
    }

    def __init__(self, text: str = "", tone: str = "muted", parent=None):
        super().__init__(parent)
        self.setObjectName("Pill")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.set_tone(tone)
        self.set_status(text, tone, dot=False)

    def set_tone(self, tone: str) -> None:
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_status(self, text: str, tone: str | None = None, dot: bool = True) -> None:
        if tone is not None:
            self.set_tone(tone)
        color = self.DOT_COLORS.get(str(self.property("tone")), theme.MUTED)
        if dot:
            self.setText(
                f'<span style="color:{color};">●</span>&nbsp;&nbsp;{text}'
            )
        else:
            self.setText(text)


class IconButton(QPushButton):
    def __init__(
        self,
        icon_name: str,
        tooltip: str = "",
        size: int = 38,
        color: str = theme.TEXT,
        hover_color: str = theme.ACCENT,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("IconButton")
        self.setFixedSize(size, size)
        self.setIconSize(QSize(18, 18))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._icon_name = icon_name
        self._color = color
        self._hover_color = hover_color
        self._icon_size = 18
        self.setIcon(make_icon(icon_name, color, self._icon_size))
        if tooltip:
            self.setToolTip(tooltip)

    def set_icon(self, icon_name: str, color: str | None = None) -> None:
        self._icon_name = icon_name
        if color:
            self._color = color
        self.setIcon(make_icon(self._icon_name, self._color, self._icon_size))

    def enterEvent(self, event):  # noqa: N802
        self.setIcon(make_icon(self._icon_name, self._hover_color, self._icon_size))
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self.setIcon(make_icon(self._icon_name, self._color, self._icon_size))
        super().leaveEvent(event)


class NavButton(QPushButton):
    def __init__(self, text: str, icon_name: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("Nav")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self.setIconSize(QSize(18, 18))
        self._icon_name = icon_name
        self.setIcon(make_icon(icon_name, theme.MUTED, 18))
        self.toggled.connect(self._sync_icon)

    def _sync_icon(self, checked: bool) -> None:
        self.setIcon(make_icon(self._icon_name, theme.ACCENT if checked else theme.MUTED, 18))


class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(46, 26)
        self._pos = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.toggled.connect(self._animate)

    def _on_anim(self, value) -> None:
        self._pos = float(value)
        self.update()

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def set_checked_silent(self, checked: bool) -> None:
        self.blockSignals(True)
        super().setChecked(bool(checked))
        self.blockSignals(False)
        self._pos = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        if self._pos > 0.001:
            base = QColor("#22344E")
            target = QColor(theme.ACCENT)
            mix = self._pos
            color = QColor(
                int(base.red() + (target.red() - base.red()) * mix),
                int(base.green() + (target.green() - base.green()) * mix),
                int(base.blue() + (target.blue() - base.blue()) * mix),
            )
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 13, 13)
        else:
            painter.setBrush(QColor("#1A2A41"))
            painter.setPen(QPen(QColor("#2A3E59"), 1))
            painter.drawRoundedRect(rect, 13, 13)

        travel = self.width() - 8 - 18
        knob_x = 4 + 9 + self._pos * travel
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFFFFF" if self.isChecked() else "#93A6BF"))
        painter.drawEllipse(QPointF(knob_x, self.height() / 2), 9, 9)
        painter.end()


class ToggleRow(QWidget):
    toggled = Signal(bool)

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("Value")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("Muted")
        self.subtitle.setWordWrap(True)
        text_col.addWidget(self.title)
        if subtitle:
            text_col.addWidget(self.subtitle)
        row.addLayout(text_col, 1)
        self.switch = ToggleSwitch()
        row.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignVCenter)
        self.switch.toggled.connect(self.toggled)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def isChecked(self) -> bool:  # noqa: N802
        return self.switch.isChecked()

    def setChecked(self, checked: bool, silent: bool = True) -> None:  # noqa: N802
        if silent:
            self.switch.set_checked_silent(checked)
        else:
            self.switch.setChecked(checked)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            local = self.switch.mapFrom(self, event.position().toPoint())
            if not self.switch.rect().contains(local):
                self.switch.toggle()
        super().mouseReleaseEvent(event)


class MiniToggle(QWidget):
    toggled = Signal(bool)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        self.label = QLabel(text)
        self.label.setObjectName("Muted")
        self.switch = ToggleSwitch()
        self.switch.setFixedSize(38, 22)
        row.addWidget(self.label)
        row.addWidget(self.switch)
        self.switch.toggled.connect(self.toggled)

    def set_checked_silent(self, checked: bool) -> None:
        self.switch.blockSignals(True)
        self.switch.setChecked(bool(checked))
        self.switch.blockSignals(False)
        self.switch._pos = 1.0 if checked else 0.0
        self.switch.update()


class ValueSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        formatter=None,
        subtitle: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._formatter = formatter or (lambda v: str(v))
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("Value")
        self.value_label = QLabel(self._formatter(value))
        self.value_label.setObjectName("Accent")
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(self.value_label)
        col.addLayout(header)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("Muted")
            sub.setWordWrap(True)
            col.addWidget(sub)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._on_change)
        col.addWidget(self.slider)

    def _on_change(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))
        self.valueChanged.emit(value)

    def set_value(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.value_label.setText(self._formatter(int(value)))
        self.slider.blockSignals(False)


# ---------------------------------------------------------------------------
# Media / display widgets
# ---------------------------------------------------------------------------
class VideoView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._title = "Camera preview"
        self._subtitle = "Your webcam will appear here"
        self._icon = "camera"
        self._show_placeholder = True
        self.setMinimumSize(420, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self._show_placeholder = False
        self.update()

    def clear_frame(self) -> None:
        self._image = None
        self.update()

    def set_placeholder_visible(self, visible: bool) -> None:
        self._show_placeholder = bool(visible)
        self.update()

    def set_placeholder(self, title: str, subtitle: str = "", icon: str = "camera-off") -> None:
        self._title = title
        self._subtitle = subtitle
        self._icon = icon
        self._show_placeholder = True
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 17, 17)
        painter.setClipPath(path)

        painter.fillRect(rect, QColor("#0A1421"))

        if self._image is not None and not self._image.isNull():
            image = self._image
            scaled = image.scaled(
                rect.size().toSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = rect.x() + (rect.width() - scaled.width()) / 2
            y = rect.y() + (rect.height() - scaled.height()) / 2
            painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), scaled)
        else:
            if not self._show_placeholder:
                painter.setClipping(False)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(theme.BORDER), 1))
                painter.drawRoundedRect(rect, 17, 17)
                painter.end()
                return
            pixmap = icon_pixmap(self._icon, theme.FAINT, 42)
            painter.drawPixmap(
                int(rect.center().x() - 21), int(rect.center().y() - 58), pixmap
            )
            title_font = QFont(theme.FONT_FAMILY, 12, QFont.Weight.DemiBold)
            painter.setFont(title_font)
            painter.setPen(QColor(theme.MUTED))
            title_rect = QRectF(rect.left(), rect.center().y() - 2, rect.width(), 26)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter, self._title)
            sub_font = QFont(theme.FONT_FAMILY, 9)
            painter.setFont(sub_font)
            painter.setPen(QColor(theme.FAINT))
            sub_rect = QRectF(rect.left() + 24, rect.center().y() + 24, rect.width() - 48, 40)
            painter.drawText(
                sub_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                self._subtitle,
            )

        painter.setClipping(False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawRoundedRect(rect, 17, 17)
        painter.end()


class RoundedImage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: QImage | str) -> None:
        if isinstance(image, str):
            image = QImage(image)
        self._image = image
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        painter.setClipPath(path)
        painter.fillRect(rect, QColor("#0A1421"))
        if self._image is not None and not self._image.isNull():
            scaled = self._image.scaled(
                rect.size().toSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = rect.center().x() - scaled.width() / 2
            y = rect.center().y() - scaled.height() / 2
            painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), scaled)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 14, 14)
        painter.end()


class SignDial(QWidget):
    """Large detected sign with a hold-to-type progress ring."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(204, 204)
        self._label: str | None = None
        self._progress = 0.0
        self._confidence = 0.0
        self._stable = False
        self._hand = False
        self._armed = True
        self._flash = False

    def set_state(self, state: RecognitionState) -> None:
        self._label = state.display_label
        self._progress = state.hold_progress
        self._confidence = state.confidence
        self._stable = state.is_stable
        self._hand = state.hand_present
        self._armed = state.armed
        if state.committed:
            self._flash = True
            QTimer.singleShot(360, self._clear_flash)
        self.update()

    def _clear_flash(self) -> None:
        self._flash = False
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        outer = QRectF(
            (self.width() - side) / 2 + 11,
            (self.height() - side) / 2 + 11,
            side - 22,
            side - 22,
        )
        inner = outer.adjusted(19, 19, -19, -19)

        # Track
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#17263B"), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(outer)

        if self._progress > 0.001:
            ring_color = theme.GREEN if self._flash else theme.ACCENT
            pen = QPen(QColor(ring_color), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(
                outer,
                90 * 16,
                -int(360 * 16 * max(0.0, min(1.0, self._progress))),
            )

        # Inner disc
        painter.setPen(QPen(QColor("#1B2B43"), 1))
        painter.setBrush(QColor("#0A1421"))
        painter.drawEllipse(inner)

        text = display_name(self._label) if self._label else "—"
        if text == "—":
            font = QFont(theme.FONT_FAMILY)
            font.setPixelSize(64)
            painter.setFont(font)
            painter.setPen(QColor(theme.FAINT))
            painter.drawText(inner, Qt.AlignmentFlag.AlignCenter, text)
        else:
            font = QFont(theme.FONT_FAMILY)
            size = 74 if len(text) == 1 else (46 if len(text) <= 3 else 30)
            font.setPixelSize(size)
            font.setWeight(QFont.Weight.Bold)
            metrics = QFontMetrics(font)
            max_width = inner.width() * 0.72
            if metrics.horizontalAdvance(text) > max_width:
                size = max(20, int(size * max_width / max(1, metrics.horizontalAdvance(text))))
                font.setPixelSize(size)
            painter.setFont(font)
            if self._stable and self._hand:
                painter.setPen(QColor(theme.GREEN if self._flash else theme.ACCENT))
            elif self._hand:
                painter.setPen(QColor(theme.MUTED))
            else:
                painter.setPen(QColor(theme.FAINT))
            painter.drawText(
                inner.adjusted(0, -10, 0, -10),
                Qt.AlignmentFlag.AlignCenter,
                text,
            )

        caption_font = QFont(theme.FONT_FAMILY)
        caption_font.setPixelSize(9)
        caption_font.setWeight(QFont.Weight.DemiBold)
        caption_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        painter.setFont(caption_font)
        if self._flash:
            caption, color = "TYPED", QColor(theme.GREEN)
        elif not self._hand:
            caption, color = "SHOW YOUR HAND", QColor(theme.FAINT)
        elif not self._stable:
            caption, color = "READING…", QColor(theme.AMBER)
        elif not self._armed:
            caption, color = "RELEASE TO REPEAT", QColor(theme.MUTED)
        else:
            caption, color = f"HOLD  {int(self._progress * 100)}%", QColor(theme.ACCENT)
        painter.setPen(color)
        painter.drawText(
            QRectF(inner.left(), inner.center().y() + 30, inner.width(), 22),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            caption,
        )
        painter.end()


class ConfidenceBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._threshold = 0.0
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self.update()

    def set_threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#16253A"))
        painter.drawRoundedRect(rect, 4, 4)
        if self._value > 0:
            fill = QRectF(rect.left(), rect.top(), rect.width() * self._value, rect.height())
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawRoundedRect(fill, 4, 4)
        if 0 < self._threshold < 1:
            x = rect.left() + rect.width() * self._threshold
            painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
            painter.drawLine(QPointF(x, rect.top() - 2), QPointF(x, rect.bottom() + 2))
        painter.end()


class LetterTile(QAbstractButton):
    def __init__(self, label: str, hint: str = "", parent=None):
        super().__init__(parent)
        self.label = label
        self.hint = hint
        self.count = 0
        self.target = 20
        self.setCheckable(True)
        self.setFixedSize(54, 54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if hint:
            self.setToolTip(f"{display_name(label)} — {hint}")

    def set_count(self, count: int) -> None:
        self.count = int(count)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        checked = self.isChecked()
        hover = self.underMouse()
        if checked:
            painter.setBrush(QColor(34, 211, 238, 34))
            painter.setPen(QPen(QColor(theme.ACCENT), 1.4))
        elif hover:
            painter.setBrush(QColor("#13243A"))
            painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
        else:
            painter.setBrush(QColor("#0B1626"))
            painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawRoundedRect(rect, 12, 12)

        if self.label == BACKSPACE:
            tile_text = "DEL"
        elif is_custom_word_label(self.label):
            word = display_name(self.label)
            tile_text = word[:4] + ("…" if len(word) > 4 else "")
        else:
            tile_text = {" ": "SPC", ".": "STOP"}.get(self.label, self.label)
        font = QFont(theme.FONT_FAMILY)
        if len(tile_text) == 1:
            font.setPixelSize(17)
        elif len(tile_text) <= 3:
            font.setPixelSize(11)
        else:
            font.setPixelSize(9)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(theme.ACCENT if checked else theme.TEXT))
        painter.drawText(
            rect.adjusted(0, 1, 0, -11),
            Qt.AlignmentFlag.AlignCenter,
            tile_text,
        )

        count_font = QFont(theme.FONT_FAMILY)
        count_font.setPixelSize(8)
        count_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(count_font)
        painter.setPen(QColor(theme.GREEN if self.count >= self.target else theme.FAINT))
        painter.drawText(
            rect.adjusted(0, 0, 0, -5),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            str(self.count),
        )

        if self.count > 0:
            width = rect.width() - 16
            fraction = min(1.0, self.count / self.target)
            bar = QRectF(rect.left() + 8, rect.bottom() - 4, width * fraction, 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.GREEN if self.count >= self.target else theme.ACCENT))
            painter.drawRoundedRect(bar, 1, 1)

        if self.count >= self.target:
            painter.setBrush(QColor(theme.GREEN))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(rect.right() - 7, rect.top() + 7), 3, 3)
        painter.end()


class LogoMark(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(42, 42)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawRoundedRect(rect, 13, 13)
        pixmap = icon_pixmap("hand", "#06111D", 24)
        painter.drawPixmap(
            int(rect.center().x() - 12), int(rect.center().y() - 12), pixmap
        )
        painter.end()


class Toast(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setMaximumWidth(620)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, text: str, tone: str = "info", duration: int = 3200) -> None:
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText(text)
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        self._timer.start(duration)

    def reposition(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        self.adjustSize()
        x = max(16, (parent.width() - self.width()) // 2)
        y = max(16, parent.height() - self.height() - 26)
        self.move(x, y)


def elide(text: str, width: int, font: QFont) -> str:
    metrics = QFontMetrics(font)
    return metrics.elidedText(text, Qt.TextElideMode.ElideRight, width)


def row_widget(layout, height: int = 36) -> QWidget:
    """Wrap a horizontal layout so it cannot grow vertically inside a Card."""
    widget = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(layout)
    widget.setFixedHeight(height)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return widget


def wrap_label(text: str, object_name: str = "Body") -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label
