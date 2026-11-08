"""Crisp inline SVG icons (Lucide-style paths) rendered with QtSvg."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICONS: dict[str, str] = {
    "hand": (
        '<path d="M18 11V6a2 2 0 0 0-4 0v5"/>'
        '<path d="M14 10V4a2 2 0 0 0-4 0v2"/>'
        '<path d="M10 10.5V6a2 2 0 0 0-4 0v8"/>'
        '<path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34'
        'l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>'
    ),
    "camera": (
        '<rect x="2.5" y="6.5" width="13" height="11" rx="2.5"/>'
        '<path d="M15.5 10.2 21 7.2v9.6l-5.5-3"/>'
    ),
    "camera-off": (
        '<path d="M10.7 6.5H13a2.5 2.5 0 0 1 2.5 2.5v1.2l5.5-3v9.6l-2.2-1.2"/>'
        '<path d="M15.5 14.5v.5a2.5 2.5 0 0 1-2.5 2.5H5A2.5 2.5 0 0 1 2.5 15V9'
        'A2.5 2.5 0 0 1 5 6.5h1"/>'
        '<path d="M3 3l18 18"/>'
    ),
    "pause": '<rect x="7" y="5" width="3.4" height="14" rx="1"/><rect x="13.6" y="5" width="3.4" height="14" rx="1"/>',
    "play": '<path d="M8 5.5v13l10-6.5-10-6.5Z"/>',
    "backspace": (
        '<path d="M9.5 5H20a1.5 1.5 0 0 1 1.5 1.5v11A1.5 1.5 0 0 1 20 19H9.5'
        'L3.2 12.7a1 1 0 0 1 0-1.4L9.5 5Z"/>'
        '<path d="M12.5 9.5l5 5M17.5 9.5l-5 5"/>'
    ),
    "space": '<path d="M4 7v10M20 7v10M8 12h8"/><path d="M13 9.5 16 12l-3 2.5"/>',
    "trash": (
        '<path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>'
        '<path d="M6.5 7l.8 12a1.5 1.5 0 0 0 1.5 1.4h6.4a1.5 1.5 0 0 0 1.5-1.4L17.5 7"/>'
        '<path d="M10 11v6M14 11v6"/>'
    ),
    "volume": (
        '<path d="M11 5 6.5 9H3v6h3.5L11 19V5Z"/>'
        '<path d="M15 8.5a5 5 0 0 1 0 7M17.8 6a9 9 0 0 1 0 12"/>'
    ),
    "volume-off": (
        '<path d="M11 5 6.5 9H3v6h3.5L11 19V5Z"/>'
        '<path d="M16 9.5l5 5M21 9.5l-5 5"/>'
    ),
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "copy": (
        '<rect x="8" y="8" width="12" height="12" rx="2"/>'
        '<path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
        'a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.03 1.56V21a2 2 0 1 1-4 0v-.1'
        'a1.7 1.7 0 0 0-1.11-1.56 1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06'
        'A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.56-1.03H3a2 2 0 1 1 0-4h.1'
        'A1.7 1.7 0 0 0 4.6 8.9a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06'
        'A1.7 1.7 0 0 0 9 4.6h.05A1.7 1.7 0 0 0 10.08 3V3a2 2 0 1 1 4 0v.1'
        'A1.7 1.7 0 0 0 15.1 4.6a1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06'
        'A1.7 1.7 0 0 0 19.4 9v.05a1.7 1.7 0 0 0 1.56 1.03H21a2 2 0 1 1 0 4h-.1'
        'A1.7 1.7 0 0 0 19.4 15Z"/>'
    ),
    "book": (
        '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5v-15Z"/>'
        '<path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20v3H6.5A2.5 2.5 0 0 1 4 20.5Z"/>'
    ),
    "activity": '<path d="M3 12h4l2.5-6 5 12L17 12h4"/>',
    "check": '<path d="M4.5 12.5 9 17l10.5-10.5"/>',
    "sparkles": (
        '<path d="M12 3l1.7 4.6L18.5 9l-4.8 1.4L12 15l-1.7-4.6L5.5 9l4.8-1.4L12 3Z"/>'
        '<path d="M18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z"/>'
    ),
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 6.3"/><path d="M20 4v7h-7"/>',
    "record": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="3" fill="currentColor"/>',
    "mic": (
        '<rect x="9" y="3" width="6" height="11" rx="3"/>'
        '<path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M9 21h6"/>'
    ),
    "chevron-right": '<path d="M9 6l6 6-6 6"/>',
    "chevron-down": '<path d="M6 9l6 6 6-6"/>',
    "download": '<path d="M12 3v12M7.5 10.5 12 15l4.5-4.5"/><path d="M4 19h16"/>',
    "upload": '<path d="M12 21V9M7.5 13.5 12 9l4.5 4.5"/><path d="M4 4h16"/>',
    "x": '<path d="M6 6l12 12M18 6 6 18"/>',
    "wand": (
        '<path d="M15 4V2M15 10V8M11 6H9M21 6h-2M18.4 3.6l-1.4 1.4M18.4 8.4l-1.4-1.4'
        'M11.6 3.6l1.4 1.4M4 20l9-9"/>'
    ),
    "arrow-right": '<path d="M4 12h16M14 6l6 6-6 6"/>',
    "keyboard": (
        '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/>'
        '<path d="M6 9.5h.01M9.5 9.5h.01M13 9.5h.01M16.5 9.5h.01M7 14h10"/>'
    ),
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "alert": '<path d="M12 3 2.5 20h19L12 3Z"/><path d="M12 9v5M12 17.5h.01"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "zap": '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"/>',
    "users": (
        '<circle cx="9" cy="8" r="3.5"/>'
        '<path d="M2.5 20a6.5 6.5 0 0 1 13 0"/>'
        '<path d="M16 4.6a3.5 3.5 0 0 1 0 6.8M18.5 20a6.4 6.4 0 0 0-2.5-5.1"/>'
    ),
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
}


@lru_cache(maxsize=256)
def make_icon(name: str, color: str = "#202A25", size: int = 20) -> QIcon:
    body = ICONS.get(name, ICONS["info"]).replace("currentColor", color)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.9" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    dpr = 2
    pixmap = QPixmap(size * dpr, size * dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return QIcon(pixmap)


def icon_pixmap(name: str, color: str = "#202A25", size: int = 20) -> QPixmap:
    return make_icon(name, color, size).pixmap(QSize(size, size))
