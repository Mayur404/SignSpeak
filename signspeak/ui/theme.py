"""SignSpeak visual theme."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

BG = "#060A12"
BG_SOFT = "#080E19"
CARD = "#0E1828"
CARD_2 = "#111E31"
CARD_3 = "#14243A"
BORDER = "#1B2B43"
BORDER_STRONG = "#2A405E"
TEXT = "#E9EFFA"
MUTED = "#AAB8CA"
FAINT = "#8193AC"
ACCENT = "#22D3EE"
ACCENT_DIM = "#0E7490"
VIOLET = "#8B5CF6"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#F87171"

FONT_FAMILY = "Segoe UI"
MONO_FAMILY = "Cascadia Mono"


def _font_family() -> str:
    families = set(QFontDatabase.families())
    return FONT_FAMILY if FONT_FAMILY in families else "Arial"


QSS = f"""
* {{
    font-family: "{FONT_FAMILY}";
    color: {TEXT};
    outline: none;
}}
QMainWindow, QWidget#Root {{
    background: {BG};
}}
QWidget#Content {{
    background: {BG};
}}
QFrame#Sidebar {{
    background: {BG_SOFT};
    border-right: 1px solid #131F31;
}}
QFrame#Card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
}}
QFrame#Inset {{
    background: #0B1524;
    border: 1px solid #17263B;
    border-radius: 13px;
}}
QFrame#Divider {{
    background: #17253A;
    border: none;
    max-height: 1px;
}}
QLabel {{
    background: transparent;
}}
QLabel#H1 {{
    font-size: 27px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#H2 {{
    font-size: 18px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#Section {{
    font-size: 11px;
    font-weight: 700;
    color: {FAINT};
}}
QLabel#Muted {{
    color: {MUTED};
    font-size: 12px;
}}
QLabel#Body {{
    color: #B9C8DC;
    font-size: 13px;
}}
QLabel#Value {{
    color: {TEXT};
    font-size: 13px;
    font-weight: 600;
}}
QLabel#BigValue {{
    color: {TEXT};
    font-size: 26px;
    font-weight: 700;
}}
QLabel#Accent {{
    color: {ACCENT};
    font-weight: 600;
}}
QLabel#Pill {{
    background: #122135;
    border: 1px solid #1E3049;
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    color: {MUTED};
}}
QLabel#Pill[tone="accent"] {{
    background: rgba(34, 211, 238, 0.10);
    border: 1px solid rgba(34, 211, 238, 0.35);
    color: {ACCENT};
}}
QLabel#Pill[tone="green"] {{
    background: rgba(52, 211, 153, 0.10);
    border: 1px solid rgba(52, 211, 153, 0.35);
    color: {GREEN};
}}
QLabel#Pill[tone="amber"] {{
    background: rgba(251, 191, 36, 0.10);
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: {AMBER};
}}
QLabel#Pill[tone="red"] {{
    background: rgba(248, 113, 113, 0.10);
    border: 1px solid rgba(248, 113, 113, 0.35);
    color: {RED};
}}
QLabel#Pill[tone="violet"] {{
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.38);
    color: #C4B5FD;
}}
QLabel#Toast {{
    background: #101D30;
    border: 1px solid {BORDER_STRONG};
    border-radius: 12px;
    padding: 11px 18px;
    color: {TEXT};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#Toast[tone="success"] {{
    background: #0C2A22;
    border: 1px solid rgba(52, 211, 153, 0.45);
    color: #A7F3D0;
}}
QLabel#Toast[tone="error"] {{
    background: #2A1216;
    border: 1px solid rgba(248, 113, 113, 0.45);
    color: #FECACA;
}}
QLabel#Toast[tone="info"] {{
    background: #0D2230;
    border: 1px solid rgba(34, 211, 238, 0.40);
    color: #A5F3FC;
}}

QPushButton {{
    background: {CARD_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 8px 14px;
    color: {TEXT};
    font-size: 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: {CARD_3};
    border-color: {BORDER_STRONG};
}}
QPushButton:pressed {{
    background: #0D1929;
}}
QPushButton:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton:disabled {{
    color: #4E6078;
    background: #0B1421;
    border-color: #152236;
}}
QPushButton#Primary {{
    background: {ACCENT};
    border: none;
    color: #06111D;
    font-weight: 800;
    padding: 10px 18px;
}}
QPushButton#Primary:hover {{
    background: #67D8E8;
}}
QPushButton#Primary:focus {{
    border: 2px solid #D9F7FF;
}}
QPushButton#Primary:disabled {{
    background: #1A2637;
    color: #56677F;
}}
QPushButton#Ghost {{
    background: transparent;
    border: 1px solid #24364F;
    color: #C8D6E8;
}}
QPushButton#Ghost:hover {{
    background: #101D2F;
    border-color: {BORDER_STRONG};
}}
QPushButton#Ghost:focus, QPushButton#Danger:focus, QPushButton#IconButton:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton#Danger {{
    background: rgba(248, 113, 113, 0.10);
    border: 1px solid rgba(248, 113, 113, 0.35);
    color: #FCA5A5;
}}
QPushButton#Danger:hover {{
    background: rgba(248, 113, 113, 0.18);
}}
QPushButton#IconButton {{
    background: #121F32;
    border: 1px solid #20324B;
    border-radius: 10px;
    padding: 0px;
}}
QPushButton#IconButton:hover {{
    background: #182A42;
    border-color: #2C4566;
}}
QPushButton#IconButton:checked {{
    background: rgba(34, 211, 238, 0.14);
    border-color: rgba(34, 211, 238, 0.55);
}}
QPushButton#Nav {{
    background: transparent;
    border: none;
    border-radius: 11px;
    padding: 10px 12px;
    text-align: left;
    color: {MUTED};
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#Nav:hover {{
    background: #0D1727;
    color: {TEXT};
}}
QPushButton#Nav:checked {{
    background: #101F33;
    color: {TEXT};
}}
QPushButton#Nav:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox {{
    background: #0D1828;
    border: 1px solid #22334C;
    border-radius: 9px;
    padding: 7px 11px;
    min-height: 18px;
    color: {TEXT};
    font-size: 12px;
}}
QComboBox:hover {{
    border-color: {BORDER_STRONG};
}}
QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background: #0C1727;
    border: 1px solid #22334C;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #17314A;
    selection-color: {TEXT};
    outline: 0;
}}
QComboBox QAbstractItemView::item {{
    min-height: 26px;
    padding-left: 8px;
}}

QLineEdit#WordField {{
    background: transparent;
    border: none;
    color: {TEXT};
    font-size: 31px;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 0px;
}}
QLineEdit#WordField:focus {{
    border: 1px solid {ACCENT};
    border-radius: 6px;
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QTextEdit {{
    background: #0A1421;
    border: 1px solid #17263B;
    border-radius: 12px;
    padding: 12px;
    color: #D7E3F4;
    font-size: 16px;
    selection-background-color: #155E75;
}}
QTextEdit:focus {{
    border-color: #22405C;
}}
QPlainTextEdit {{
    background: #0A1421;
    border: 1px solid #17263B;
    border-radius: 12px;
    padding: 12px;
    color: #D7E3F4;
    font-size: 12px;
}}

QSlider::groove:horizontal {{
    height: 5px;
    background: #16253A;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -5px 0px;
    border-radius: 7px;
    background: #F1F6FF;
    border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT};
}}
QSlider::handle:horizontal:disabled {{
    background: #33455E;
    border-color: #33455E;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #21344E;
    border-radius: 5px;
    min-height: 34px;
}}
QScrollBar::handle:vertical:hover {{
    background: #2C4667;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #21344E;
    border-radius: 5px;
    min-width: 34px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QToolTip {{
    background: #0D1A2A;
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 6px 9px;
    font-size: 11px;
}}

QMessageBox {{
    background: {CARD};
}}
QMessageBox QLabel {{
    color: {TEXT};
}}
QFileDialog {{
    background: {CARD};
}}
QProgressBar {{
    background: #0C1727;
    border: 1px solid #1B2B43;
    border-radius: 6px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    family = _font_family()
    font = QFont(family, 10)
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0A1421"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(CARD))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(CARD_2))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DIM))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    app.setStyleSheet(QSS)
