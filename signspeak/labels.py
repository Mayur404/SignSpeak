"""Gesture vocabulary and human-readable descriptions."""
from __future__ import annotations

import unicodedata

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = [str(d) for d in range(10)]
SPACE = " "
PERIOD = "."
BACKSPACE = "__BACKSPACE__"
CUSTOM_WORD_PREFIX = "__WORD__:"
MAX_CUSTOM_WORD_LENGTH = 110

# Order used by the bundled RandomForest (classes 0..37).
LEGACY_INDEX_TO_LABEL = {i: ch for i, ch in enumerate(LETTERS + DIGITS)}
LEGACY_INDEX_TO_LABEL[36] = SPACE
LEGACY_INDEX_TO_LABEL[37] = PERIOD

LABELS = LETTERS + DIGITS + [SPACE, PERIOD]

DISPLAY_NAMES = {SPACE: "SPACE", PERIOD: "PERIOD"}
DISPLAY_NAMES[BACKSPACE] = "BACKSPACE"

# Static alphabet + digits, followed by the two custom control gestures.
TRAIN_LABELS = LABELS + [BACKSPACE]


def custom_word_label(word: str) -> str:
    normalized = " ".join(str(word).split())
    return f"{CUSTOM_WORD_PREFIX}{normalized}"


def is_custom_word_label(label: str) -> bool:
    return str(label).startswith(CUSTOM_WORD_PREFIX)


def is_trainable_label(label: str) -> bool:
    if label in TRAIN_LABELS:
        return True
    if not is_custom_word_label(label):
        return False
    word = str(label)[len(CUSTOM_WORD_PREFIX) :]
    return bool(word) and len(word) <= MAX_CUSTOM_WORD_LENGTH and all(
        unicodedata.category(char)[0] != "C" or char in "\u200c\u200d"
        for char in word
    )


def display_name(label: str | None) -> str:
    if not label:
        return "—"
    if is_custom_word_label(label):
        return str(label)[len(CUSTOM_WORD_PREFIX) :]
    return DISPLAY_NAMES.get(label, label)


def spoken_name(label: str) -> str:
    if is_custom_word_label(label):
        return display_name(label).lower()
    if label == BACKSPACE:
        return "backspace"
    if label == SPACE:
        return "space"
    if label == PERIOD:
        return "period"
    return label


SIGN_HINTS: dict[str, str] = {
    "A": "Closed fist, thumb resting beside the index finger.",
    "B": "Flat hand, fingers together, thumb folded across the palm.",
    "C": "Fingers and thumb curved into a C shape.",
    "D": "Index finger up; thumb touches the other fingertips.",
    "E": "Fingers bent down, thumb tucked across the fingertips.",
    "F": "Thumb and index touch; other three fingers point up.",
    "G": "Index and thumb point sideways, parallel.",
    "H": "Index and middle fingers together, pointing sideways.",
    "I": "Pinky finger up, other fingers in a fist.",
    "J": "Pinky up and trace a J in the air (dynamic sign).",
    "K": "Index and middle up in a V; thumb between them.",
    "L": "Index up, thumb out to make an L.",
    "M": "Thumb tucked under three fingers.",
    "N": "Thumb tucked under two fingers.",
    "O": "Fingertips and thumb meet to form an O.",
    "P": "Like K, but pointing down.",
    "Q": "Like G, but pointing down.",
    "R": "Index and middle fingers crossed.",
    "S": "Fist with the thumb across the front of the fingers.",
    "T": "Thumb tucked between index and middle fingers.",
    "U": "Index and middle fingers together, pointing up.",
    "V": "Index and middle fingers apart in a V.",
    "W": "Index, middle and ring fingers up.",
    "X": "Index finger hooked like a small hook.",
    "Y": "Thumb and pinky extended, other fingers curled.",
    "Z": "Index finger traces a Z in the air (dynamic sign).",
    "0": "Fingers and thumb form an O (same handshape as O).",
    "1": "Index finger up.",
    "2": "Index and middle fingers up in a V.",
    "3": "Thumb, index and middle fingers extended.",
    "4": "Four fingers up, thumb folded.",
    "5": "Open hand, all five fingers extended.",
    "6": "Thumb and pinky touch, other fingers up.",
    "7": "Thumb and ring finger touch, other fingers up.",
    "8": "Thumb and middle finger touch, other fingers up.",
    "9": "Thumb and index finger touch, other fingers up.",
    SPACE: "Custom gesture: thumbs up. Finishes the current word.",
    PERIOD: "Custom gesture: 'I love you' (thumb, index, pinky). Finishes the sentence.",
}
