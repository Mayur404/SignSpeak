"""Landmark feature extraction.

Two feature layouts are supported:

* ``legacy42`` – the exact translation-only normalisation used by the bundled
  RandomForest model (21 landmarks x 2 coordinates).
* ``rich63`` – translation, scale *and* rotation normalisation using x/y/z.
  This is what SignSpeak's own trainer uses; it is considerably more robust
  across different hand sizes, camera distances and wrist angles.
"""
from __future__ import annotations

import math

import numpy as np

NUM_LANDMARKS = 21
LEGACY_DIM = NUM_LANDMARKS * 2
RICH_DIM = NUM_LANDMARKS * 3

FEATURE_DIMS = {"legacy42": LEGACY_DIM, "rich63": RICH_DIM}


def as_points(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks (objects or array) to a (21, 3) float array."""
    if isinstance(landmarks, np.ndarray):
        points = np.asarray(landmarks, dtype=np.float32)
    else:
        points = np.asarray(
            [[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in landmarks],
            dtype=np.float32,
        )
    if points.ndim != 2 or points.shape[0] != NUM_LANDMARKS:
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {points.shape}")
    if points.shape[1] == 2:
        points = np.concatenate([points, np.zeros((NUM_LANDMARKS, 1), np.float32)], axis=1)
    return points[:, :3]


def legacy_features(points: np.ndarray) -> np.ndarray:
    """Translation-invariant normalisation matching the reference model."""
    points = as_points(points)
    x = points[:, 0]
    y = points[:, 1]
    data = np.empty(LEGACY_DIM, dtype=np.float32)
    data[0::2] = x - x.min()
    data[1::2] = y - y.min()
    return data


def rich_features(points: np.ndarray) -> np.ndarray:
    """Translation, scale and rotation normalised 63-dimensional features."""
    points = as_points(points).astype(np.float32, copy=True)

    # Move the wrist to the origin.
    points -= points[0]

    # Scale by the largest wrist-to-landmark distance in the image plane.
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    scale = float(xy_norm.max())
    if not math.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    points /= scale

    # Rotate so the wrist -> middle-finger MCP vector points straight up.
    vx, vy = float(points[9, 0]), float(points[9, 1])
    if abs(vx) > 1e-6 or abs(vy) > 1e-6:
        angle = math.atan2(vy, vx)
        phi = -math.pi / 2.0 - angle
        c, s = math.cos(phi), math.sin(phi)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        points[:, :2] = points[:, :2] @ rot.T

    # Keep values in a friendly range for distance-based classifiers.
    return np.clip(points.reshape(-1), -4.0, 4.0).astype(np.float32)


def extract(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "legacy42":
        return legacy_features(points)
    if mode == "rich63":
        return rich_features(points)
    raise ValueError(f"Unknown feature mode: {mode}")
