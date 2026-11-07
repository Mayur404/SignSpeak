"""Camera capture, MediaPipe hand tracking, overlay drawing and recording."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

# Configure FFmpeg before cv2 loads its video backends. The stream reader
# discards stale decoded frames, so TCP can remain reliable without a backlog.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .config import Settings
from .features import as_points, dual_hand_features
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


class _LatestFrameReader:
    """Keep RTSP decoding ahead of inference and expose only the newest frame."""

    def __init__(self, capture):
        self.capture = capture
        self._lock = threading.Lock()
        self._frame = None
        self._sequence = 0
        self._failed = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped.is_set():
            try:
                ok, frame = self.capture.read()
            except Exception:
                # OpenCV can throw when release() interrupts a blocking RTSP
                # read during shutdown. Treat it as an ordinary end of feed.
                if not self._stopped.is_set():
                    self._failed = True
                return
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._sequence += 1

    def read(self):
        with self._lock:
            if self._sequence == 0 or self._failed:
                return False, None
            self._sequence = 0
            return True, self._frame

    def close(self):
        self._stopped.set()
        try:
            self.capture.release()
        except Exception:
            pass
        self._thread.join(timeout=1.0)

    def release(self):
        """Match VideoCapture's cleanup method for shared worker error paths."""
        self.close()

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
    hands = np.asarray(points)
    if hands.ndim == 2:
        hands = hands[None, ...]

    def px(x, y):
        x = 1.0 - x if mirror else x
        return int(x * w), int(y * h)

    xs = [1.0 - p[0] if mirror else p[0] for hand in hands for p in hand]
    ys = [p[1] for hand in hands for p in hand]
    pad = 0.025
    x1 = int(max(0.0, min(xs) - pad) * w)
    x2 = int(min(1.0, max(xs) + pad) * w)
    y1 = int(max(0.0, min(ys) - pad) * h)
    y2 = int(min(1.0, max(ys) + pad) * h)
    color = GREEN if state.stable_label else ACCENT

    # Draw each detected hand in its own accent so both skeletons are clear.
    hand_colors = (color, VIOLET)
    for hand_index, hand in enumerate(hands):
        hand_color = hand_colors[hand_index % len(hand_colors)]
        for a, b in HAND_CONNECTIONS:
            pa, pb = px(hand[a][0], hand[a][1]), px(hand[b][0], hand[b][1])
            cv2.line(frame, pa, pb, DARK, 7, cv2.LINE_AA)
            cv2.line(frame, pa, pb, hand_color, 3, cv2.LINE_AA)

        for index, point in enumerate(hand):
            center = px(point[0], point[1])
            radius = 6 if index in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, center, radius + 2, DARK, -1, cv2.LINE_AA)
            cv2.circle(
                frame, center, radius,
                WHITE if index in (4, 8, 12, 16, 20) else hand_color,
                -1, cv2.LINE_AA,
            )

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
            num_hands=2,
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
            hands = np.stack([as_points(hand) for hand in result.hand_landmarks])
            categories = [row[0] for row in result.handedness if row] if result.handedness else []
            handedness = "+".join(category.category_name or "" for category in categories)
            scores = [float(category.score or 0.0) for category in categories]
            score = float(np.mean(scores)) if scores else 0.0
            return Observation(hands, handedness, score)

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
    def stop(self, timeout_ms: int = 8000) -> bool:
        self._stop.set()
        return self.wait(timeout_ms)

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
            array = np.asarray(samples, dtype=np.float32) if samples else np.empty((0, 126), np.float32)
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
        last_preview = 0.0
        fps = 0.0
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    # RTSP may take several seconds to deliver the first frame,
                    # especially while opening the 4K main stream.
                    if failures > 300:
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
                # Limit Qt preview updates so a high resolution stream cannot
                # queue old images behind the camera's live feed.
                if now - last_preview >= 1.0 / 15.0:
                    last_preview = now
                    display_width = 1280
                    if display.shape[1] > display_width:
                        display = cv2.resize(
                            display,
                            (
                                display_width,
                                int(display.shape[0] * display_width / display.shape[1]),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )
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
                if isinstance(cap, _LatestFrameReader):
                    cap.close()
                else:
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
                self._rec_samples.append(dual_hand_features(points))
                self._rec_last = now
                progress = len(self._rec_samples)
        if progress is not None:
            self.recordingProgress.emit(label, progress)
        if finished:
            self._finish_recording()

    def _open_camera(self):
        source = (getattr(self.settings, "camera_source", "") or "").strip()
        if source:
            # Handle RTSP / IP camera stream
            import os
            # Set this before opening each RTSP capture. Avoid FFmpeg's
            # no-buffer/low-delay decoder flags, which can reject fragmented
            # H.264 packets from this camera.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            candidates = [source]
            if "192.168.128.10" in source or "stream1" in source or "stream0" in source:
                sp_sub = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
                sp_main = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=0.sdp"
                # Honor the stream selected in the UI, then use the other
                # Sparsh stream only if the selected one cannot be opened.
                if "/stream=1." in source:
                    alternate = sp_main
                elif "/stream=0." in source:
                    alternate = sp_sub
                else:
                    alternate = sp_sub
                candidates = [source]
                if alternate not in candidates:
                    candidates.append(alternate)

            cap = None
            for url in candidates:
                try:
                    params = [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
                    ]
                    try:
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
                    except (TypeError, cv2.error):
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        break
                    cap.release()
                    cap = None
                except Exception:
                    if cap is not None:
                        cap.release()
                    cap = None

            if cap is None or not cap.isOpened():
                self.error.emit(
                    f"Could not connect to camera stream:\n{source}\n\n"
                    "Please check the Ethernet link to 192.168.128.10."
                )
                if cap is not None:
                    cap.release()
                return None

            return _LatestFrameReader(cap)

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
    """Enumerate working cameras and RTSP streams without blocking the UI."""

    finishedList = Signal(list)

    def __init__(self, skip_source: int | str | None = None, parent=None):
        super().__init__(parent)
        self.skip_source = skip_source

    def run(self) -> None:
        import socket

        found: list[tuple[int | str, str]] = []

        # Check if Sparsh CCTV camera is reachable on Ethernet link
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.4)
            if sock.connect_ex(("192.168.128.10", 554)) == 0:
                found.append((
                    "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp",
                    "Sparsh CCTV (720p Substream - Fast)",
                ))
                found.append((
                    "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=0.sdp",
                    "Sparsh CCTV (4K Mainstream - higher latency)",
                ))
            sock.close()
        except Exception:
            pass

        # If current source is a custom RTSP stream, retain it in list
        if isinstance(self.skip_source, str) and self.skip_source.startswith("rtsp://"):
            existing = [item[0] for item in found]
            if self.skip_source not in existing:
                found.append((self.skip_source, "Custom RTSP Stream"))

        for index in range(5):
            if self.skip_source == index:
                continue
            cap = None
            try:
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    found.append((index, f"Webcam {index}"))
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        self.finishedList.emit(found)
