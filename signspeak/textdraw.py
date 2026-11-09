"""Unicode-safe text drawing for OpenCV frames.

``cv2.putText`` only knows ASCII, so Kannada / Hindi / Tamil ... custom words
came out as ``?????`` on the video HUD. These helpers keep cv2 for plain ASCII
(fast path) and render anything else with Pillow and bundled Noto fonts.
They mirror ``cv2.getTextSize`` / ``cv2.putText`` so call sites barely change.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .paths import ASSETS_DIR

FONTS_DIR = ASSETS_DIR / "fonts"

# Unicode block -> bundled Noto font (all Bold to match the HUD weight).
_SCRIPTS = (
    (0x0900, 0x097F, "NotoSansDevanagari-Bold.ttf"),
    (0x0980, 0x09FF, "NotoSansBengali-Bold.ttf"),
    (0x0A00, 0x0A7F, "NotoSansGurmukhi-Bold.ttf"),
    (0x0A80, 0x0AFF, "NotoSansGujarati-Bold.ttf"),
    (0x0B00, 0x0B7F, "NotoSansOriya-Bold.ttf"),
    (0x0B80, 0x0BFF, "NotoSansTamil-Bold.ttf"),
    (0x0C00, 0x0C7F, "NotoSansTelugu-Bold.ttf"),
    (0x0C80, 0x0CFF, "NotoSansKannada-Bold.ttf"),
    (0x0D00, 0x0D7F, "NotoSansMalayalam-Bold.ttf"),
)
_LATIN = "NotoSans-Bold.ttf"
_JOINERS = "‌‍"
# Hershey DUPLEX at scale 1.0 has ~22 px caps; Noto caps are ~0.71 em.
_PX_PER_SCALE = 30


def _is_ascii(text: str) -> bool:
    return text.isascii()


def _font_file(ch: str) -> str:
    code = ord(ch)
    for lo, hi, name in _SCRIPTS:
        if lo <= code <= hi:
            return name
    return _LATIN


def _runs(text: str) -> list[tuple[str, str]]:
    """Split text into (font file, run) pieces; joiners and spaces stay
    with the run they sit in so shaping isn't broken."""
    runs: list[tuple[str, str]] = []
    for ch in text:
        if runs and (ch in _JOINERS or ch == " "):
            name = runs[-1][0]
        else:
            name = _font_file(ch)
        if runs and runs[-1][0] == name:
            runs[-1] = (name, runs[-1][1] + ch)
        else:
            runs.append((name, ch))
    return runs


@lru_cache(maxsize=32)
def _font(name: str, px: int):
    from PIL import ImageFont, features

    layout = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    path = FONTS_DIR / name
    if not path.exists():
        path = FONTS_DIR / _LATIN
    return ImageFont.truetype(str(path), px, layout_engine=layout)


@lru_cache(maxsize=256)
def _render(text: str, px: int) -> tuple[np.ndarray, int] | None:
    """Return (alpha mask cropped to the ink, ink rows above the baseline)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    runs = [(_font(name, px), run) for name, run in _runs(text)]
    width = int(sum(font.getlength(run) for font, run in runs)) + px
    height = px * 3
    baseline = px * 2
    image = Image.new("L", (max(1, width), height), 0)
    draw = ImageDraw.Draw(image)
    x = px // 4
    for font, run in runs:
        draw.text((x, baseline), run, font=font, fill=255, anchor="ls")
        x += font.getlength(run)
    box = image.getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    mask = np.asarray(image.crop((left, top, right, bottom)), dtype=np.float32) / 255.0
    return mask, baseline - top


def _px(scale: float) -> int:
    return max(8, int(round(scale * _PX_PER_SCALE)))


def text_size(text: str, font: int, scale: float, thickness: int):
    """Drop-in for ``cv2.getTextSize`` that understands any script."""
    if _is_ascii(text):
        return cv2.getTextSize(text, font, scale, thickness)
    rendered = _render(text, _px(scale))
    if rendered is None:
        return cv2.getTextSize(text, font, scale, thickness)
    mask, above = rendered
    return (mask.shape[1], above), mask.shape[0] - above


def put_text(frame, text: str, org, font: int, scale: float, color, thickness: int = 1,
             line_type: int = cv2.LINE_AA) -> None:
    """Drop-in for ``cv2.putText``; ``org`` is the baseline's left point."""
    if _is_ascii(text):
        cv2.putText(frame, text, org, font, scale, color, thickness, line_type)
        return
    rendered = _render(text, _px(scale))
    if rendered is None:
        cv2.putText(frame, text, org, font, scale, color, thickness, line_type)
        return
    mask, above = rendered
    h, w = mask.shape
    x0, y0 = int(org[0]), int(org[1]) - above
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    alpha = mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0, None]
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    ink = np.asarray(color, dtype=np.float32)[: frame.shape[2]]
    frame[fy0:fy1, fx0:fx1] = (region * (1.0 - alpha) + ink * alpha).astype(frame.dtype)
