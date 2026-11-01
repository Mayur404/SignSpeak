"""Camera capture, MediaPipe hand tracking, overlay drawing and recording."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .config import Settings
from .features import as_points, rich_features
from .labels import display_name
from .paths import LANDMARKER_MODEL
from .recognition import RecognitionEngine, RecognitionState
from .trainer import MAX_SAMPLES_PER_TAKE

# MediaPipe hand topology (wrist, thumb, index, middle, ring, pinky).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

ACCENT = (238, 211, 34)      # #22D3EE in BGR
VIOLET = (246, 92, 139)      # #8B5CF6 in BGR
WHITE = (245, 248, 252)
DARK = (8, 12, 20)
GREEN = (153, 211, 52)       # #34D399 in BGR


_MEDIAPIPE_LOG_LOCK = threading.RLock()
_KNOWN_MEDIAPIPE_DIAGNOSTICS = (
    b"Created TensorFlow Lite XNNPACK delegate for CPU.",
    b"Logging before InitGoogle() is written to STDERR",
    b"inference_feedback_manager.cc:",
    b"landmark_projection_calculator.cc:",
)


@contextmanager
def _quiet_known_mediapipe_diagnostics():
    """Hide known benign native diagnostics while preserving other stderr."""
    # MediaPipe/TFLite write directly to the process stderr file descriptor.
    # Limit the process-wide redirection to model initialization and the first
    # inference, and replay every captured line except the known diagnostics.
    with _MEDIAPIPE_LOG_LOCK:
        sys.stderr.flush()
        original_stderr = os.dup(2)
        with tempfile.TemporaryFile() as captured:
            try:
                os.dup2(captured.fileno(), 2)
                yield
            finally:
                sys.stderr.flush()
                os.dup2(original_stderr, 2)
                os.close(original_stderr)
                captured.seek(0)
                for line in captured:
                    if not any(token in line for token in _KNOWN_MEDIAPIPE_DIAGNOSTICS):
                        os.write(2, line)


@dataclass
class Observation:
    points: np.ndarray
    handedness: str
    score: float


def _rounded_rect(image, p1, p2, color, radius=12, thickness=-1, alpha=1.0):
    x1, y1 = p1
    x2, y2 = p2
    if alpha >= 0.999:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        return
    overlay = image.copy()
    _rounded_rect(overlay, p1, p2, color, radius, thickness, 1.0)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def _draw_overlay(frame, points, state: RecognitionState, mirror: bool, paused: bool = False):
    h, w = frame.shape[:2]

    def px(x, y):
        x = 1.0 - x if mirror else x
        return int(x * w), int(y * h)

    xs = [1.0 - p[0] if mirror else p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 0.025
    x1 = int(max(0.0, min(xs) - pad) * w)
    x2 = int(min(1.0, max(xs) + pad) * w)
    y1 = int(max(0.0, min(ys) - pad) * h)
    y2 = int(min(1.0, max(ys) + pad) * h)
    color = GREEN if state.stable_label else ACCENT

    # Skeleton with a soft dark outline for contrast.
    for a, b in HAND_CONNECTIONS:
        pa, pb = px(points[a][0], points[a][1]), px(points[b][0], points[b][1])
        cv2.line(frame, pa, pb, DARK, 7, cv2.LINE_AA)
        cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)

    for index, point in enumerate(points):
        center = px(point[0], point[1])
        radius = 6 if index in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, center, radius + 2, DARK, -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, WHITE if index in (4, 8, 12, 16, 20) else color, -1, cv2.LINE_AA)

    # Modern corner brackets instead of a heavy rectangle.
    length = max(18, int(min(x2 - x1, y2 - y1) * 0.18))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, 3, cv2.LINE_AA)

    # Label chip above the hand.
    label = display_name(state.display_label) if state.display_label else "…"
    confidence = int(round(state.confidence * 100))
    text = f"{label}  {confidence}%" if state.hand_present else label
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.75
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    chip_w, chip_h = tw + 28, th + 22
    cx = min(max(x1, 12), w - chip_w - 12)
    cy = max(12, y1 - chip_h - 12)
    _rounded_rect(frame, (cx, cy), (cx + chip_w, cy + chip_h), DARK, 12, -1, 0.72)
    _rounded_rect(frame, (cx, cy), (cx + chip_w, cy + chip_h), color, 12, 1, 0.9)
    cv2.putText(frame, text, (cx + 14, cy + chip_h - 12), font, scale, WHITE, 2, cv2.LINE_AA)


def _draw_paused(frame) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), DARK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    text = "PAUSED"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
    cv2.putText(
        frame, text, ((w - tw) // 2, (h + th) // 2),
        cv2.FONT_HERSHEY_DUPLEX, 1.4, WHITE, 3, cv2.LINE_AA,
    )


class HandTracker:
    """Latest MediaPipe Tasks hand landmarker, VIDEO mode."""

    def __init__(self, detection: float = 0.5, tracking: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Hand landmarker model missing: {LANDMARKER_MODEL}\n"
                "Re-download hand_landmarker.task into assets/models."
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection,
            min_hand_presence_confidence=detection,
            min_tracking_confidence=tracking,
        )
        self._mp = mp
        with _quiet_known_mediapipe_diagnostics():
            self._detector = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._last_timestamp = 0
        self._first_inference = True

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Observation | None:
        with self._lock:
            if self._detector is None:
                return None
            # The tracker is shared between camera sessions, so keep the
            # VIDEO-mode timestamps strictly increasing even after a restart.
            timestamp_ms = max(int(timestamp_ms), self._last_timestamp + 1)
            self._last_timestamp = timestamp_ms
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            if self._first_inference:
                self._first_inference = False
                with _quiet_known_mediapipe_diagnostics():
                    result = self._detector.detect_for_video(image, timestamp_ms)
            else:
                result = self._detector.detect_for_video(image, timestamp_ms)
            if not result.hand_landmarks:
                return None
            landmarks = result.hand_landmarks[0]
            handedness = ""
            score = 0.0
            if result.handedness:
                category = result.handedness[0][0]
                handedness = category.category_name or ""
                score = float(category.score or 0.0)
            return Observation(as_points(landmarks), handedness, score)

    def close(self) -> None:
        with self._lock:
            detector, self._detector = self._detector, None
        try:
            if detector is not None:
                detector.close()
        except Exception:
            pass


_TRACKER_LOCK = threading.Lock()
_TRACKER: HandTracker | None = None
_TRACKER_READY = threading.Event()
_TRACKER_ERROR: BaseException | None = None
_TRACKER_STARTED = False


def _warm_tracker_worker() -> None:
    global _TRACKER, _TRACKER_ERROR
    try:
        tracker = HandTracker()
    except BaseException as exc:  # keep the UI thread free to report it
        with _TRACKER_LOCK:
            _TRACKER_ERROR = exc
    else:
        with _TRACKER_LOCK:
            _TRACKER = tracker
    finally:
        _TRACKER_READY.set()


def warm_tracker() -> None:
    """Start loading MediaPipe in the background, before the camera needs it."""
    global _TRACKER_STARTED
    with _TRACKER_LOCK:
        if _TRACKER_STARTED or _TRACKER is not None:
            return
        _TRACKER_STARTED = True
    threading.Thread(
        target=_warm_tracker_worker, name="HandTrackerWarmup", daemon=True
    ).start()


def get_tracker() -> HandTracker:
    warm_tracker()
    _TRACKER_READY.wait()
    with _TRACKER_LOCK:
        if _TRACKER is None:
            raise RuntimeError(f"Hand tracker failed to initialise: {_TRACKER_ERROR}")
        return _TRACKER


def shutdown_tracker() -> None:
    global _TRACKER, _TRACKER_STARTED, _TRACKER_ERROR
    with _TRACKER_LOCK:
        tracker = _TRACKER
        _TRACKER = None
        _TRACKER_STARTED = False
        _TRACKER_ERROR = None
        _TRACKER_READY.clear()
    if tracker is not None:
        tracker.close()


class CameraWorker(QThread):
    frameReady = Signal(QImage)
    stateReady = Signal(object)
    committed = Signal(str)
    fpsReady = Signal(float)
    statusChanged = Signal(str)
    error = Signal(str)
    recordingProgress = Signal(str, int)
    recordingFinished = Signal(str, object)

    def __init__(self, settings: Settings, engine: RecognitionEngine, parent=None):
        super().__init__(parent)
        self.settings = settings.copy()
        self.engine = engine
        self._stop = threading.Event()
        self._paused = False
        self._recognition_enabled = True
        self._recording_lock = threading.Lock()
        self._rec_label: str | None = None
        self._rec_until = 0.0
        self._rec_samples: list[np.ndarray] = []
        self._rec_last = 0.0
        self._last_timestamp = 0
        self._manual_commit = False

    # ---- thread-safe control surface -------------------------------------
    def stop(self) -> None:
        self._stop.set()
        self.wait(2500)

    def request_manual_commit(self) -> None:
        self._manual_commit = True

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def set_recognition_enabled(self, enabled: bool) -> None:
        self._recognition_enabled = bool(enabled)
        if not enabled:
            self.engine.reset()

    def apply_settings(self, settings: Settings) -> None:
        self.settings = settings.copy()
        self.engine.apply_settings(self.settings)

    def set_engine(self, engine: RecognitionEngine) -> None:
        self.engine = engine

    def start_recording(self, label: str, duration: float = 3.0) -> None:
        with self._recording_lock:
            self._rec_label = label
            self._rec_until = time.monotonic() + max(0.5, duration)
            self._rec_samples = []
            self._rec_last = 0.0

    def cancel_recording(self) -> None:
        with self._recording_lock:
            self._rec_label = None
            self._rec_samples = []

    def _finish_recording(self) -> None:
        with self._recording_lock:
            label = self._rec_label
            samples = self._rec_samples
            self._rec_label = None
            self._rec_samples = []
        if label:
            array = np.asarray(samples, dtype=np.float32) if samples else np.empty((0, 63), np.float32)
            self.recordingFinished.emit(label, array)

    # ---- main loop -------------------------------------------------------
    def run(self) -> None:  # noqa: C901 - a capture loop is naturally linear
        cap = self._open_camera()
        if cap is None:
            return
        tracker = None
        try:
            tracker = get_tracker()
        except Exception as exc:
            self.error.emit(str(exc))
            cap.release()
            return

        self.statusChanged.emit("live")
        start = time.monotonic()
        last_frame = start
        fps = 0.0
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 60:
                        self.error.emit("Lost the camera feed. Reconnect it and press Retry.")
                        break
                    time.sleep(0.03)
                    continue
                failures = 0

                now = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame.shape[:2]
                infer_width = min(int(self.settings.inference_width), width)
                if infer_width < width:
                    infer = cv2.resize(
                        rgb,
                        (infer_width, int(height * infer_width / width)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    infer = rgb

                timestamp = max(self._last_timestamp + 1, int((now - start) * 1000))
                self._last_timestamp = timestamp
                try:
                    observation = tracker.detect(infer, timestamp)
                except Exception as exc:
                    observation = None
                    self.error.emit(f"Hand tracking error: {exc}")

                points = observation.points if observation else None
                state = self.engine.update(points, timestamp)
                if state.committed and self._recognition_enabled and not self._paused:
                    self.committed.emit(state.committed)

                if self._manual_commit:
                    self._manual_commit = False
                    label = self.engine.commit_now(timestamp)
                    if label and self._recognition_enabled and not self._paused:
                        self.committed.emit(label)

                self._collect_recording(points, now)

                display = cv2.flip(frame, 1) if self.settings.mirror else frame
                if self.settings.draw_overlay and points is not None:
                    _draw_overlay(display, points, state, self.settings.mirror, False)
                if self._paused:
                    _draw_paused(display)

                self.stateReady.emit(state)
                self.frameReady.emit(self._to_qimage(display))

                elapsed = now - last_frame
                last_frame = now
                if elapsed > 0:
                    instant = 1.0 / elapsed
                    fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
                    if int((now - start) * 10) % 5 == 0:
                        self.fpsReady.emit(fps)
        finally:
            self._finish_recording()
            try:
                cap.release()
            except Exception:
                pass
            self.statusChanged.emit("stopped")

    def _collect_recording(self, points: np.ndarray | None, now: float) -> None:
        progress = None
        finished = False
        with self._recording_lock:
            label = self._rec_label
            if not label:
                return
            if now >= self._rec_until or len(self._rec_samples) >= MAX_SAMPLES_PER_TAKE:
                finished = True
            elif points is not None and now - self._rec_last >= 0.18:
                self._rec_samples.append(rich_features(points))
                self._rec_last = now
                progress = len(self._rec_samples)
        if progress is not None:
            self.recordingProgress.emit(label, progress)
        if finished:
            self._finish_recording()

    def _open_camera(self):
        index = int(self.settings.camera_index)
        cap = None
        if sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.error.emit(
                f"Could not open camera {index}. Close other apps using it, "
                "or choose another camera in Settings."
            )
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        for _ in range(3):
            cap.read()
        return cap

    @staticmethod
    def _to_qimage(bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()


class CameraProbe(QThread):
    """Enumerate working camera indices without blocking the UI.

    The camera that is already in use is skipped so probing cannot fight the
    capture thread for the same device.
    """

    finishedList = Signal(list)

    def __init__(self, skip_index: int | None = None, parent=None):
        super().__init__(parent)
        self.skip_index = skip_index

    def run(self) -> None:
        found: list[tuple[int, str]] = []
        for index in range(5):
            if self.skip_index is not None and index == int(self.skip_index):
                continue
            cap = None
            try:
                if sys.platform == "win32":
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    found.append((index, f"Camera {index}"))
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        self.finishedList.emit(found)
