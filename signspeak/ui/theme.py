"""SignSpeak visual theme."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

BG = "#F3F5F2"
BG_SOFT = "#FFFFFF"
CARD = "#FFFFFF"
CARD_2 = "#F7F9F7"
CARD_3 = "#EDF2EF"
BORDER = "#DDE4DF"
BORDER_STRONG = "#BFCBC3"
TEXT = "#202A25"
MUTED = "#5E6C64"
FAINT = "#7D8982"
ACCENT = "#287A68"
ACCENT_DIM = "#236957"
VIOLET = "#70618F"
GREEN = "#28784B"
AMBER = "#956A18"
RED = "#B44740"

FONT_FAMILY = "Inter"
FONT_FALLBACKS = ("Aptos", "Segoe UI", "Arial")
MONO_FAMILY = "Cascadia Mono"


def _font_family() -> str:
    families = set(QFontDatabase.families())
    for family in (FONT_FAMILY, *FONT_FALLBACKS):
        if family in families:
            return family
    return "Arial"


QSS = f"""
* {{
    font-family: "__FONT_FAMILY__";
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
    border-right: 1px solid {BORDER};
}}
QFrame#Card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 13px;
}}
QFrame#Inset {{
    background: #F8FAF8;
    border: 1px solid {BORDER};
    border-radius: 9px;
}}
QFrame#Divider {{
    background: {BORDER};
    border: none;
    max-height: 1px;
}}
QLabel {{
    background: transparent;
}}
QLabel#H1 {{
    font-size: 29px;
    font-weight: 650;
    color: {TEXT};
}}
QLabel#H2 {{
    font-size: 17px;
    font-weight: 650;
    color: {TEXT};
}}
QLabel#Section {{
    font-size: 10px;
    font-weight: 700;
    color: {FAINT};
    letter-spacing: 0.7px;
}}
QLabel#Muted {{
    color: {MUTED};
    font-size: 12px;
}}
QLabel#Body {{
    color: {MUTED};
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
    background: #F2F5F3;
    border: 1px solid {BORDER};
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    color: {MUTED};
}}
QLabel#Pill[tone="accent"] {{
    background: #EAF4F1;
    border: 1px solid #C9E1D9;
    color: {ACCENT};
}}
QLabel#Pill[tone="green"] {{
    background: #EAF4ED;
    border: 1px solid #CDE4D3;
    color: {GREEN};
}}
QLabel#Pill[tone="amber"] {{
    background: #FAF2E3;
    border: 1px solid #EBD9B5;
    color: {AMBER};
}}
QLabel#Pill[tone="red"] {{
    background: #F9ECEA;
    border: 1px solid #EBCBC7;
    color: {RED};
}}
QLabel#Pill[tone="violet"] {{
    background: #F0EDF5;
    border: 1px solid #D9D1E7;
    color: {VIOLET};
}}
QLabel#Toast {{
    background: #FFFFFF;
    border: 1px solid {BORDER_STRONG};
    border-radius: 12px;
    padding: 11px 18px;
    color: {TEXT};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#Toast[tone="success"] {{
    background: #EAF4ED;
    border: 1px solid #CDE4D3;
    color: #245C39;
}}
QLabel#Toast[tone="error"] {{
    background: #F9ECEA;
    border: 1px solid #EBCBC7;
    color: #843B36;
}}
QLabel#Toast[tone="info"] {{
    background: #EAF4F1;
    border: 1px solid #C9E1D9;
    color: #225C4F;
}}

QPushButton {{
    background: {CARD_2};
    border: 1px solid {BORDER};
    border-radius: 7px;
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
    background: #E8EEEA;
}}
QPushButton:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton:disabled {{
    color: #9AA59E;
    background: #F0F3F1;
    border-color: #E2E8E4;
}}
QPushButton#Primary {{
    background: {ACCENT};
    border: none;
    color: #FFFFFF;
    font-weight: 700;
    padding: 10px 18px;
}}
QPushButton#Primary:hover {{
    background: #226B5B;
}}
QPushButton#Primary:focus {{
    border: 2px solid #B8DCD2;
}}
QPushButton#Primary:disabled {{
    background: #E9EEEB;
    color: #9AA59E;
}}
QPushButton#Ghost {{
    background: transparent;
    border: 1px solid {BORDER};
    color: {TEXT};
}}
QPushButton#Ghost:hover {{
    background: #F1F5F2;
    border-color: {BORDER_STRONG};
}}
QPushButton#Ghost:focus, QPushButton#Danger:focus, QPushButton#IconButton:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton#Danger {{
    background: #F9ECEA;
    border: 1px solid #EBCBC7;
    color: #843B36;
}}
QPushButton#Danger:hover {{
    background: #F5DEDB;
}}
QPushButton#IconButton {{
    background: #F6F8F6;
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0px;
}}
QPushButton#IconButton:hover {{
    background: #EAF0EC;
    border-color: {BORDER_STRONG};
}}
QPushButton#IconButton:checked {{
    background: #EAF4F1;
    border-color: #AFCFC5;
}}
QPushButton#Nav {{
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: left;
    color: {MUTED};
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#Nav:hover {{
    background: #F1F5F2;
    color: {TEXT};
}}
QPushButton#Nav:checked {{
    background: #EAF4F1;
    color: {TEXT};
    border-left: 3px solid {ACCENT};
    padding-left: 9px;
}}
QPushButton#Nav:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
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
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #DDECE6;
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
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px;
    color: {TEXT};
    font-size: 16px;
    selection-background-color: #CFE4DC;
}}
QTextEdit:focus {{
    border-color: {BORDER_STRONG};
}}
QPlainTextEdit {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px;
    color: {TEXT};
    font-size: 12px;
}}

QSlider::groove:horizontal {{
    height: 5px;
    background: #E4EAE6;
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
    background: #FFFFFF;
    border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT};
}}
QSlider::handle:horizontal:disabled {{
    background: #C5CEC8;
    border-color: #C5CEC8;
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
    background: #CDD7D1;
    border-radius: 5px;
    min-height: 34px;
}}
QScrollBar::handle:vertical:hover {{
    background: #B8C5BD;
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
    background: #CDD7D1;
    border-radius: 5px;
    min-width: 34px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QToolTip {{
    background: #FFFFFF;
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
    background: #F0F3F1;
    border: 1px solid {BORDER};
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
    global FONT_FAMILY
    FONT_FAMILY = family
    font = QFont(family, 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(CARD))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(CARD_2))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DIM))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    app.setStyleSheet(QSS.replace("__FONT_FAMILY__", family))
