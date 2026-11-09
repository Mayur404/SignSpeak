"""Threaded Web Video Streamer and Recognition Engine for SignSpeak.

Provides real-time camera capture, MediaPipe dual-hand landmark detection,
extra-trees / random-forest sign classification, stabilization, word/sentence
composition, and multipart MJPEG stream encoding for web browsers.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Any, Generator

import cv2
import numpy as np

from ..composer import TextComposer
from ..config import Settings, load_settings, save_settings
from ..features import dual_hand_features
from ..labels import (
    BACKSPACE,
    PERIOD,
    SPACE,
    SIGN_HINTS,
    TRAIN_LABELS,
    custom_word_label,
    display_name,
    is_custom_word_label,
    is_trainable_label,
)
from ..paths import APP_ROOT, USER_MODEL
from ..recognition import (
    ModelLoadError,
    RecognitionEngine,
    RecognitionState,
    load_user_classifier,
)
from ..trainer import (
    DatasetStore,
    TrainingError,
    train_model,
)
from ..textdraw import put_text, text_size
from ..vision import (
    HAND_CONNECTIONS,
    HandTracker,
    Observation,
    _LatestFrameReader,
    _rounded_rect,
    _camera_candidates,
    _safe_camera_source,
    get_tracker,
    warm_tracker,
)

# Color constants (BGR)
ACCENT_CYAN = (238, 211, 34)       # #22D3EE
ACCENT_VIOLET = (246, 92, 139)     # #8B5CF6
ACCENT_GREEN = (153, 211, 52)      # #34D399
ACCENT_CORAL = (94, 63, 244)       # #F43F5E
DARK_BG = (16, 13, 9)              # #090D10
CARD_BG = (26, 20, 15)             # #0F141A
WHITE = (250, 250, 250)
GRAY = (140, 140, 140)


def _draw_rounded_rect(image, p1, p2, color, radius=10, thickness=-1, alpha=1.0) -> None:
    """Rounded rectangle for the web HUD (shared clean implementation)."""
    _rounded_rect(image, p1, p2, color, radius, thickness, alpha)


class WebStreamEngine:
    """Core video streamer and recognition engine for web clients."""

    def __init__(self, settings: Settings | None = None):
        self.settings = (settings or load_settings()).copy()
        self.composer = TextComposer()

        # The personal classifier is the only active model.
        self.classifier = None
        self.model_kind = "user"
        self._model_error: str | None = None
        try:
            self.classifier = load_user_classifier()
        except ModelLoadError as exc:
            self._model_error = str(exc)

        if self.classifier:
            self.engine = RecognitionEngine(self.classifier, self.settings)
        else:
            self.engine = None

        # Camera & capture state
        self._camera_source = self._resolve_initial_source()
        self._active_camera_source: str | None = None
        self._cap = None
        self._tracker: HandTracker | None = None
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._frame_seq = 0

        # Output buffers
        self._latest_jpeg: bytes | None = None
        self._latest_state = RecognitionState()
        self._latest_fps = 0.0
        self._last_error: str | None = None
        self._camera_connected = False
        self._camera_dims = (1280, 720)
        self._recent_events: list[dict[str, Any]] = []

        # Dataset and recording
        self.store = DatasetStore()
        self._custom_words: set[str] = {lbl for lbl in self.store.counts() if is_custom_word_label(lbl)}
        self._recording_lock = threading.Lock()
        self._rec_label: str | None = None
        self._rec_start_at = 0.0
        self._rec_until = 0.0
        self._rec_samples: list[np.ndarray] = []
        self._rec_last = 0.0
        self._rec_active = False
        self._rec_countdown = 0
        self._rec_finished_notice: str | None = None

        # Word-level debounce: a brief hand flicker re-arms the recogniser,
        # which would commit (and speak) the same word a second time.
        self._last_word_label: str | None = None
        self._last_word_at = 0.0

        # Model training state
        self._training_lock = threading.Lock()
        self._is_training = False
        self._train_progress = 0
        self._train_message = "Ready"
        self._train_error: str | None = None
        self._train_result: dict[str, Any] | None = None

        # Start warm-up
        warm_tracker()

    _WORD_REPEAT_WINDOW_S = 3.0
    # Train in a child process so recognition keeps running (tests disable it).
    use_training_subprocess = True

    def _is_repeat_word(self, label: str) -> bool:
        """True if this whole-word sign was just committed (flicker repeat).

        Single letters are exempt so double letters (e.g. HELLO) still work.
        """
        now = time.monotonic()
        is_word = len(label) > 1 and label not in (SPACE, PERIOD, BACKSPACE)
        if not is_word:
            self._last_word_label = None
            return False
        if label == self._last_word_label and now - self._last_word_at < self._WORD_REPEAT_WINDOW_S:
            self._last_word_at = now
            return True
        self._last_word_label = label
        self._last_word_at = now
        return False

    def _resolve_initial_source(self) -> str:
        """Pick the camera source: CAMERA_SOURCE env > saved choice > 5G RTSP.

        A fresh clone (e.g. on the edge VM) has no webcam, so the default is
        the Sparsh RTSP stream on CAMERA_HOST. A saved webcam index is only
        honoured when that device actually exists on this machine.
        """
        env_source = os.environ.get("CAMERA_SOURCE", "").strip()
        if env_source:
            return env_source
        saved = (self.settings.camera_source or "").strip()
        if saved and not saved.isdigit():
            return saved
        if saved.isdigit() and (
            sys.platform == "win32" or os.path.exists(f"/dev/video{int(saved)}")
        ):
            return saved
        env_host = os.environ.get("CAMERA_HOST", "").strip() or "10.101.0.74"
        return f"rtsp://admin:admin123@{env_host}:554/avstream/channel=1/stream=1.sdp"

    def start(self) -> None:
        """Start the background capture and inference worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop, name="SignSpeak-WebStreamer", daemon=True
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker."""
        with self._lock:
            self._running = False
            self._frame_cond.notify_all()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._close_camera()

    def _close_camera(self) -> None:
        """Safely release the active capture object."""
        if self._cap is not None:
            try:
                if isinstance(self._cap, _LatestFrameReader):
                    self._cap.close()
                else:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._active_camera_source = None
        self._camera_connected = False

    def _open_camera(self, source: str):
        """Open camera from integer webcam index, RTSP URL, or file path."""
        source = str(source).strip()
        if source.lower() in ("webcam", "laptop", "cam0", "0", ""):
            source = "0"

        # 1. Check if integer index (local webcam)
        if source.isdigit():
            index = int(source)
            cap = None
            if sys.platform == "win32":
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(index)
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
            return None

        # 2. RTSP or Video File
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        candidates = _camera_candidates(source)
        for url in candidates:
            try:
                params = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2500,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500,
                ]
                try:
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
                except (TypeError, cv2.error):
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    return _LatestFrameReader(cap)
                cap.release()
            except Exception:
                pass
        return None

    def _worker_loop(self) -> None:
        """Main camera acquisition, hand detection, recognition & HUD loop."""
        consecutive_failures = 0
        last_frame_time = time.monotonic()
        last_good_frame = time.monotonic()
        last_tracker_retry = 0.0

        while self._running:
            # 1. Initialize tracker if not ready
            if self._tracker is None:
                now = time.monotonic()
                if now - last_tracker_retry > 2.0:
                    last_tracker_retry = now
                    try:
                        self._tracker = get_tracker()
                    except Exception as exc:
                        self._last_error = f"MediaPipe HandTracker: {exc}"

            # 2. Reopen the capture when the requested source changes. Keep all
            # capture reads/releases on this worker thread; releasing a capture
            # from the request thread while read() is in progress can hang or
            # leave OpenCV stuck on the old camera.
            with self._lock:
                requested_source = self._camera_source
                active_source = self._active_camera_source

            if self._cap is not None and active_source != requested_source:
                self._close_camera()

            # Open a snapshot of the requested source. Discard it if another
            # switch arrived while OpenCV was connecting.
            if self._cap is None:
                cap = self._open_camera(requested_source)
                if cap is None:
                    self._camera_connected = False
                    self._last_error = f"Cannot open camera: {_safe_camera_source(requested_source)}"
                    # Generate animated standby placeholder frame so browser stream stays active
                    self._render_placeholder_frame("Connecting to camera...", requested_source)
                    time.sleep(1.0)
                    continue
                with self._lock:
                    source_is_current = self._camera_source == requested_source
                if not source_is_current:
                    try:
                        cap.close() if isinstance(cap, _LatestFrameReader) else cap.release()
                    except Exception:
                        pass
                    continue
                self._cap = cap
                self._active_camera_source = requested_source
                self._camera_connected = True
                last_good_frame = time.monotonic()
                self._last_error = None
                consecutive_failures = 0

            # 3. Read frame
            cap = self._cap
            if cap is None:
                continue
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None:
                consecutive_failures += 1
                # The RTSP reader returns "no new frame" instantly while the
                # stream catches up, so judge by time, not by count: only
                # reconnect after 6 s (webcam: 3 s) without any frame.
                stall_limit = 6.0 if isinstance(cap, _LatestFrameReader) else 3.0
                if time.monotonic() - last_good_frame > stall_limit:
                    consecutive_failures = 0
                    last_good_frame = time.monotonic()
                    self._camera_connected = False
                    self._close_camera()
                    time.sleep(0.5)
                else:
                    time.sleep(0.02)
                continue

            consecutive_failures = 0
            now = time.monotonic()
            last_good_frame = now
            h, w = frame.shape[:2]
            self._camera_dims = (w, h)

            # Calculate FPS
            elapsed = now - last_frame_time
            last_frame_time = now
            if elapsed > 0:
                inst_fps = 1.0 / elapsed
                self._latest_fps = inst_fps if self._latest_fps == 0 else self._latest_fps * 0.85 + inst_fps * 0.15

            # 4. Hand Tracking & Inference
            points = None
            observation = None
            if self._tracker is not None:
                infer_w = min(int(self.settings.inference_width), w)
                if infer_w < w:
                    infer_bgr = cv2.resize(
                        frame,
                        (infer_w, int(h * infer_w / w)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    infer_bgr = frame
                infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(now * 1000)
                try:
                    observation = self._tracker.detect(infer_rgb, timestamp_ms)
                    if observation:
                        points = observation.points
                except Exception as exc:
                    self._last_error = f"Hand detection error: {exc}"

            # 4b. Sample recording collection
            with self._recording_lock:
                if self._rec_label:
                    rem_delay = max(0, math.ceil(self._rec_start_at - now))
                    self._rec_countdown = rem_delay
                    if now < self._rec_start_at:
                        pass
                    elif now >= self._rec_until or len(self._rec_samples) >= 45:
                        label = self._rec_label
                        samples = list(self._rec_samples)
                        self._rec_label = None
                        self._rec_active = False
                        self._rec_countdown = 0
                        if samples:
                            try:
                                arr = np.asarray(samples, dtype=np.float32)
                                added = self.store.add_samples(label, arr)
                                total = int(self.store.counts().get(label, 0))
                                self._rec_finished_notice = (
                                    f"Saved {added} samples for {display_name(label)} "
                                    f"({total} total). Train the model to use them."
                                )
                            except Exception as exc:
                                self._rec_finished_notice = (
                                    f"Could not save samples for {display_name(label)}: {exc}"
                                )
                        else:
                            self._rec_finished_notice = (
                                f"No hand detected - nothing was recorded for {display_name(label)}. "
                                "Keep your hand clearly in the camera frame and try again."
                            )
                    elif points is not None and now - self._rec_last >= 0.18:
                        self._rec_active = True
                        feat = dual_hand_features(points)
                        self._rec_samples.append(feat)
                        self._rec_last = now

            # 5. Recognition Engine update
            timestamp_ms = int(now * 1000)
            if self.engine:
                state = self.engine.update(points, timestamp_ms)
            else:
                state = RecognitionState(hand_present=points is not None)

            # 6. Check for committed sign -> Composer
            if state.committed and self._is_repeat_word(state.committed):
                state.committed = None
            if state.committed and not self._paused:
                res = self.composer.commit(state.committed)
                event = {
                    "timestamp": time.time(),
                    "sign": state.committed,
                    "kind": res.kind,
                    "word": self.composer.word,
                    "transcript": self.composer.transcript,
                    "speak_word": res.speak_word,
                    "speak_sentence": res.speak_sentence,
                }
                with self._lock:
                    self._recent_events.append(event)
                    if len(self._recent_events) > 50:
                        self._recent_events.pop(0)

            # 7. Render HUD overlay on display frame
            display = cv2.flip(frame, 1) if self.settings.mirror else frame
            if self.settings.draw_overlay:
                self._render_hud(display, points, state, self.settings.mirror, self._paused)

            # 8. Compress to JPEG
            # Cap display resolution at 1280px wide for optimal network performance
            if display.shape[1] > 1280:
                new_h = int(display.shape[0] * 1280 / display.shape[1])
                display = cv2.resize(display, (1280, new_h), interpolation=cv2.INTER_AREA)

            ret, jpeg_bytes = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if ret:
                with self._lock:
                    self._latest_jpeg = jpeg_bytes.tobytes()
                    self._latest_state = state
                    self._frame_seq += 1
                    self._frame_cond.notify_all()

            time.sleep(0.005)

    def _render_hud(
        self,
        frame: np.ndarray,
        points: np.ndarray | None,
        state: RecognitionState,
        mirror: bool,
        paused: bool,
    ) -> None:
        """Render high-tech cyberpunk HUD overlay on frame."""
        h, w = frame.shape[:2]

        def px(x, y):
            x = 1.0 - x if mirror else x
            return int(x * w), int(y * h)

        if paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            text = "RECOGNITION PAUSED"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
            cv2.putText(
                frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, WHITE, 2, cv2.LINE_AA,
            )
            return

        # 1. Draw Hand Skeleton
        if points is not None:
            hands = np.asarray(points)
            if hands.ndim == 2:
                hands = hands[None, ...]

            xs = [1.0 - p[0] if mirror else p[0] for hand in hands for p in hand]
            ys = [p[1] for hand in hands for p in hand]
            pad = 0.03
            x1 = int(max(0.0, min(xs) - pad) * w)
            x2 = int(min(1.0, max(xs) + pad) * w)
            y1 = int(max(0.0, min(ys) - pad) * h)
            y2 = int(min(1.0, max(ys) + pad) * h)

            hand_color = ACCENT_GREEN if state.stable_label else ACCENT_CYAN
            colors = (hand_color, ACCENT_VIOLET)

            for h_idx, hand in enumerate(hands):
                c = colors[h_idx % len(colors)]
                # Skeleton bones
                for a, b in HAND_CONNECTIONS:
                    pa, pb = px(hand[a][0], hand[a][1]), px(hand[b][0], hand[b][1])
                    cv2.line(frame, pa, pb, DARK_BG, 6, cv2.LINE_AA)
                    cv2.line(frame, pa, pb, c, 3, cv2.LINE_AA)

                # Joints
                for idx, pt in enumerate(hand):
                    center = px(pt[0], pt[1])
                    is_tip = idx in (4, 8, 12, 16, 20)
                    r = 7 if is_tip else 4
                    cv2.circle(frame, center, r + 2, DARK_BG, -1, cv2.LINE_AA)
                    cv2.circle(frame, center, r, WHITE if is_tip else c, -1, cv2.LINE_AA)

            # Modern corner brackets around hand
            bracket_len = max(20, int(min(x2 - x1, y2 - y1) * 0.18))
            for cx, cy, dx, dy in (
                (x1, y1, 1, 1),
                (x2, y1, -1, 1),
                (x1, y2, 1, -1),
                (x2, y2, -1, -1),
            ):
                cv2.line(frame, (cx, cy), (cx + dx * bracket_len, cy), hand_color, 3, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx, cy + dy * bracket_len), hand_color, 3, cv2.LINE_AA)

            # Glassmorphic HUD label chip above hand
            label_text = display_name(state.display_label) if state.display_label else "…"
            conf_val = int(round(state.confidence * 100))
            badge_str = f"{label_text}  {conf_val}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.75
            (tw, th), _ = text_size(badge_str, font, scale, 2)
            chip_w, chip_h = tw + 32, th + 24
            cx = min(max(x1, 14), w - chip_w - 14)
            cy = max(14, y1 - chip_h - 14)

            _draw_rounded_rect(frame, (cx, cy), (cx + chip_w, cy + chip_h), DARK_BG, 12, -1, 0.75)
            _draw_rounded_rect(frame, (cx, cy), (cx + chip_w, cy + chip_h), hand_color, 12, 2, 0.9)
            put_text(
                frame, badge_str, (cx + 16, cy + chip_h - 12), font, scale, WHITE, 2, cv2.LINE_AA
            )

            # Circular hold progress ring next to chip if holding
            if state.hold_progress > 0.05:
                ring_radius = 16
                ring_cx = cx + chip_w + ring_radius + 10
                ring_cy = cy + chip_h // 2
                if ring_cx + ring_radius < w - 10:
                    cv2.circle(frame, (ring_cx, ring_cy), ring_radius, DARK_BG, -1, cv2.LINE_AA)
                    cv2.circle(frame, (ring_cx, ring_cy), ring_radius, GRAY, 2, cv2.LINE_AA)
                    angle = int(360 * min(1.0, state.hold_progress))
                    cv2.ellipse(
                        frame,
                        (ring_cx, ring_cy),
                        (ring_radius, ring_radius),
                        -90,
                        0,
                        angle,
                        ACCENT_GREEN,
                        3,
                        cv2.LINE_AA,
                    )

        # 2. Bottom Stream Status Bar (Source, Live FPS, Resolution)
        # Check recording overlay
        with self._recording_lock:
            rec_countdown = self._rec_countdown
            rec_active = self._rec_active
            rec_lbl = self._rec_label
            rec_samples_len = len(self._rec_samples)

        if rec_countdown > 0:
            count_str = str(rec_countdown)
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(count_str, font, 4.0, 6)
            rcx, rcy = w // 2, h // 2
            cv2.circle(frame, (rcx, rcy), 80, DARK_BG, -1, cv2.LINE_AA)
            cv2.circle(frame, (rcx, rcy), 80, ACCENT_GREEN, 4, cv2.LINE_AA)
            cv2.putText(frame, count_str, (rcx - tw // 2, rcy + th // 2), font, 4.0, WHITE, 6, cv2.LINE_AA)
        elif rec_active and rec_lbl:
            badge_text = f"REC: {display_name(rec_lbl)} ({rec_samples_len} samples)"
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = text_size(badge_text, font, 0.7, 2)
            _draw_rounded_rect(frame, (w // 2 - tw // 2 - 16, 16), (w // 2 + tw // 2 + 16, 16 + th + 18), (0, 0, 180), 8, -1, 0.85)
            put_text(frame, badge_text, (w // 2 - tw // 2, 16 + th + 7), font, 0.7, WHITE, 2, cv2.LINE_AA)

        bar_h = 36
        _draw_rounded_rect(frame, (12, h - bar_h - 12), (w - 12, h - 12), DARK_BG, 10, -1, 0.75)
        _draw_rounded_rect(frame, (12, h - bar_h - 12), (w - 12, h - 12), (50, 50, 50), 10, 1, 0.5)

        # Left: Camera source badge
        safe_src = _safe_camera_source(self._camera_source)
        src_label = f"CAM: {safe_src}"
        cv2.putText(
            frame, src_label, (24, h - 23),
            cv2.FONT_HERSHEY_DUPLEX, 0.45, ACCENT_CYAN, 1, cv2.LINE_AA
        )

        # Right: FPS and Resolution
        fps_label = f"{self._latest_fps:.1f} FPS  |  {w}x{h}"
        (fw, _), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(
            frame, fps_label, (w - fw - 24, h - 23),
            cv2.FONT_HERSHEY_DUPLEX, 0.45, WHITE, 1, cv2.LINE_AA
        )

    def _render_placeholder_frame(self, message: str, source: str) -> None:
        """Create a futuristic standby placeholder frame when camera is offline."""
        w, h = 960, 540
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = DARK_BG

        # Grid lines for tech look
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (24, 20, 15), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (24, 20, 15), 1)

        # Glowing central card
        cx1, cy1, cx2, cy2 = w // 2 - 240, h // 2 - 110, w // 2 + 240, h // 2 + 110
        _draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), CARD_BG, 16, -1, 0.9)
        _draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), ACCENT_CYAN, 16, 2, 0.8)

        # Pulsing radar dot
        radar_cx, radar_cy = w // 2, h // 2 - 40
        pulse = int(12 + 6 * math.sin(time.time() * 4))
        cv2.circle(frame, (radar_cx, radar_cy), pulse + 8, (60, 45, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, (radar_cx, radar_cy), pulse, ACCENT_CYAN, -1, cv2.LINE_AA)

        # Text
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, _), _ = cv2.getTextSize(message, font, 0.8, 2)
        cv2.putText(frame, message, (w // 2 - tw // 2, h // 2 + 25), font, 0.8, WHITE, 2, cv2.LINE_AA)

        src_text = f"Target: {_safe_camera_source(source)}"
        (sw, _), _ = cv2.getTextSize(src_text, font, 0.5, 1)
        cv2.putText(frame, src_text, (w // 2 - sw // 2, h // 2 + 60), font, 0.5, GRAY, 1, cv2.LINE_AA)

        ret, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            with self._lock:
                self._latest_jpeg = jpeg_bytes.tobytes()
                self._frame_seq += 1
                self._frame_cond.notify_all()

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG stream to HTTP browser client."""
        # Send each frame once; re-send the last one only as a 1 s keep-alive
        # so a stalled camera doesn't flood the (5G) link with duplicates.
        sent_seq = -1
        last_sent = 0.0
        while self._running:
            with self._frame_cond:
                if self._frame_seq == sent_seq:
                    self._frame_cond.wait(timeout=1.0)
                frame_data = self._latest_jpeg
                seq = self._frame_seq
            now = time.monotonic()
            if seq == sent_seq and now - last_sent < 1.0:
                continue
            sent_seq, last_sent = seq, now
            if frame_data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )

    def get_state(self) -> dict[str, Any]:
        """Return serializable system and recognition metrics for /api/status."""
        with self._lock:
            state = self._latest_state
            evs = list(self._recent_events)
            self._recent_events.clear()

        top_pairs = []
        if state.top:
            top_pairs = [
                {"label": display_name(lbl), "raw_label": lbl, "confidence": round(float(prob), 3)}
                for lbl, prob in state.top[:5]
            ]

        model_info = {}
        if self.classifier:
            info = self.classifier.info
            model_info = {
                "kind": info.kind,
                "name": info.name,
                "algorithm": info.algorithm,
                "accuracy": round(info.accuracy * 100, 1) if info.accuracy is not None else None,
                "classes_count": len(self.classifier.classes),
                "feature_mode": info.feature_mode,
                "samples": info.samples,
                "validation_method": info.validation_method,
                "balanced_accuracy": (
                    round(info.balanced_accuracy * 100, 1)
                    if info.balanced_accuracy is not None else None
                ),
            }
        else:
            model_info = {
                "kind": "user",
                "error": self._model_error or "Your personal model is unavailable.",
            }

        raw_src = str(self._camera_source).strip()
        env_host = os.environ.get("CAMERA_HOST", "10.101.0.74").strip()
        if raw_src in ("0", "webcam", "laptop") or raw_src.isdigit():
            cam_mode = "laptop"
        elif "192.168.128.10" in raw_src:
            cam_mode = "ethernet"
        else:
            cam_mode = "5g"

        return {
            "camera": {
                "connected": self._camera_connected,
                "source": _safe_camera_source(self._camera_source),
                "raw_source": self._camera_source,
                "mode": cam_mode,
                "host_5g": env_host,
                "fps": round(self._latest_fps, 1),
                "width": self._camera_dims[0],
                "height": self._camera_dims[1],
                "error": self._last_error,
            },
            "recognition": {
                "hand_present": state.hand_present,
                "label": state.label,
                "display_name": display_name(state.display_label),
                "confidence": round(float(state.confidence), 3),
                "stable_label": state.stable_label,
                "hold_progress": round(float(state.hold_progress), 3),
                "is_stable": state.is_stable,
                "paused": self._paused,
                "top": top_pairs,
            },
            "composer": {
                "word": self.composer.word,
                "transcript": self.composer.transcript,
                "sentence": self.composer.sentence,
                "speak_text": self.composer.speak_text,
                "tokens": self.composer.tokens,
            },
            "model": model_info,
            "settings": self.settings.to_dict(),
            "events": evs,
        }

    # ---- Actions ---------------------------------------------------------
    def switch_camera(self, new_source: str) -> None:
        """Switch camera stream source dynamically."""
        new_source = str(new_source).strip()
        if not new_source:
            new_source = "0"
        with self._lock:
            self._camera_source = new_source
            self.settings.camera_source = new_source
            if new_source.isdigit():
                self.settings.camera_index = int(new_source)
            save_settings(self.settings)
            self._frame_cond.notify_all()

    def switch_model(self, kind: str) -> bool:
        """Load and activate the personal classifier."""
        if kind != "user":
            self._model_error = "SignSpeak only uses your personal model."
            return False
        try:
            new_classifier = load_user_classifier()
            self.classifier = new_classifier
            self.model_kind = kind
            self.settings.active_model = kind
            save_settings(self.settings)
            if self.engine:
                self.engine.set_classifier(new_classifier)
            else:
                self.engine = RecognitionEngine(new_classifier, self.settings)
            self._model_error = None
            return True
        except (ModelLoadError, OSError, ValueError, TypeError) as exc:
            self._model_error = f"Cannot load your personal model: {exc}"
            return False

    def toggle_pause(self) -> bool:
        """Toggle recognition pause state."""
        self._paused = not self._paused
        return self._paused

    def composer_action(self, action: str, param: str = "") -> dict[str, Any]:
        """Execute a text composition action."""
        action = action.lower()
        if action == "space":
            res = self.composer.space()
        elif action == "period":
            res = self.composer.period()
        elif action == "backspace":
            self.composer.backspace()
            res = None
        elif action in ("clear", "clear_all"):
            self.composer.clear_all()
            res = None
        elif action == "clear_word":
            self.composer.clear_word()
            res = None
        elif action == "delete_word":
            self.composer.delete_word()
            res = None
        elif action == "type" and param:
            res = self.composer.commit(param.upper())
        elif action == "commit_now":
            if self.engine:
                label = self.engine.commit_now(int(time.monotonic() * 1000))
                if label:
                    res = self.composer.commit(label)
                else:
                    res = None
            else:
                res = None
        else:
            res = None

        return {
            "word": self.composer.word,
            "transcript": self.composer.transcript,
            "speak_word": res.speak_word if res else None,
            "speak_sentence": res.speak_sentence if res else None,
        }

    def update_settings(self, values: dict[str, Any]) -> None:
        """Update settings live."""
        self.settings.update(values)
        save_settings(self.settings)
        if self.engine:
            self.engine.apply_settings(self.settings)

    # ---- Dataset & Training Controls -------------------------------------
    def start_recording(self, label: str, duration: float = 3.0, delay: float = 2.0) -> None:
        if not isinstance(label, str) or not is_trainable_label(label):
            raise ValueError("Choose a valid sign label before recording.")
        duration = float(duration)
        delay = float(delay)
        if not math.isfinite(duration) or not 0.5 <= duration <= 30:
            raise ValueError("Recording duration must be between 0.5 and 30 seconds.")
        if not math.isfinite(delay) or not 0 <= delay <= 15:
            raise ValueError("Recording delay must be between 0 and 15 seconds.")
        with self._recording_lock:
            self._rec_label = label
            self._rec_start_at = time.monotonic() + max(0.0, delay)
            self._rec_until = self._rec_start_at + max(0.5, duration)
            self._rec_countdown = math.ceil(max(0.0, delay))
            self._rec_samples = []
            self._rec_last = 0.0
            self._rec_active = False
            self._rec_finished_notice = None

    def cancel_recording(self) -> None:
        with self._recording_lock:
            self._rec_label = None
            self._rec_samples = []
            self._rec_active = False
            self._rec_countdown = 0

    def get_recording_state(self) -> dict[str, Any]:
        with self._recording_lock:
            notice = self._rec_finished_notice
            self._rec_finished_notice = None
            return {
                "recording": self._rec_label is not None,
                "label": self._rec_label,
                "display": display_name(self._rec_label) if self._rec_label else "",
                "active": self._rec_active,
                "countdown": self._rec_countdown,
                "samples_count": len(self._rec_samples),
                "finished_notice": notice,
            }

    def add_custom_word(self, word: str) -> str | None:
        if not isinstance(word, str):
            return None
        word = word.strip()
        if not word or len(word) > 110:
            return None
        lbl = custom_word_label(word)
        self._custom_words.add(lbl)
        return lbl

    def remove_custom_word(self, label: str) -> bool:
        if not is_custom_word_label(label):
            return False
        self._custom_words.discard(label)
        self.store.clear_label(label)
        return True

    def clear_label(self, label: str) -> None:
        self.store.clear_label(label)

    def clear_dataset(self) -> None:
        self._custom_words.clear()
        self.store.clear()

    def get_dataset_info(self) -> dict[str, Any]:
        counts = self.store.counts()
        take_counts = self.store.take_counts()
        total = self.store.total()

        # Build list of all labels: standard TRAIN_LABELS + any custom labels from dataset or added
        labels_list = list(TRAIN_LABELS)
        for lbl in list(self._custom_words) + list(counts.keys()):
            if lbl not in labels_list:
                labels_list.append(lbl)

        items = []
        for lbl in labels_list:
            items.append({
                "label": lbl,
                "display": display_name(lbl),
                "count": counts.get(lbl, 0),
                "takes": take_counts.get(lbl, 0),
                "is_custom": is_custom_word_label(lbl),
                "hint": SIGN_HINTS.get(lbl, "Custom word or gesture" if is_custom_word_label(lbl) else "")
            })

        return {
            "total_samples": total,
            "total_signs": len([c for c in counts.values() if c > 0]),
            "labels": items,
            "has_model": self.store.has_model(),
        }

    def train_personal_model_async(self) -> bool:
        with self._training_lock:
            if self._is_training:
                return False
            self._is_training = True
            self._train_progress = 0
            self._train_message = "Preparing dataset…"
            self._train_error = None
            self._train_result = None

        def _train_worker():
            try:
                X, y, groups = self.store.training_data()
                if len(X) == 0:
                    raise TrainingError("No samples available to train on.")

                def progress_cb(msg: str, pct: int):
                    with self._training_lock:
                        self._train_message = msg
                        self._train_progress = pct

                result = self._run_training(X, y, groups, progress_cb)
                with self._training_lock:
                    self._train_progress = 100
                    self._train_message = (
                        f"Trained successfully ({result['algorithm']}, {result['accuracy'] * 100:.1f}%)"
                    )
                    self._train_result = {
                        "accuracy": result["accuracy"],
                        "algorithm": result["algorithm"],
                        "samples": len(X),
                        "classes": len(np.unique(y)),
                    }
                # Reload the newly trained classifier
                if not self.switch_model("user"):
                    raise TrainingError(self._model_error or "Could not activate the trained model.")
            except Exception as exc:
                with self._training_lock:
                    self._train_error = str(exc)
                    self._train_message = f"Training failed: {exc}"
            finally:
                with self._training_lock:
                    self._is_training = False

        threading.Thread(target=_train_worker, daemon=True, name="ModelTrainer").start()
        return True

    def _run_training(self, X, y, groups, progress_cb) -> dict[str, Any]:
        """Train in a separate process so recognition never stalls.

        Falls back to in-process training if a child process can't start.
        """
        import multiprocessing as mp
        import queue as queue_mod

        from ..trainer import train_in_subprocess

        if not self.use_training_subprocess:
            res = train_model(X, y, progress=progress_cb, groups=groups)
            return {"accuracy": res.accuracy, "algorithm": res.algorithm}

        try:
            ctx = mp.get_context("spawn")
            msgs = ctx.Queue()
            proc = ctx.Process(
                target=train_in_subprocess,
                args=(X, y, groups, msgs),
                name="SignSpeak-Trainer",
                daemon=True,
            )
            proc.start()
        except Exception:
            res = train_model(X, y, progress=progress_cb, groups=groups)
            return {"accuracy": res.accuracy, "algorithm": res.algorithm}

        try:
            while True:
                try:
                    msg = msgs.get(timeout=0.5)
                except queue_mod.Empty:
                    if not proc.is_alive():
                        # Drain anything sent just before exit.
                        try:
                            msg = msgs.get(timeout=1.0)
                        except queue_mod.Empty:
                            raise TrainingError(
                                f"Training process stopped unexpectedly (exit code {proc.exitcode})."
                            )
                    else:
                        continue
                kind = msg[0]
                if kind == "progress":
                    progress_cb(msg[1], msg[2])
                elif kind == "done":
                    return msg[1]
                elif kind == "error":
                    raise TrainingError(msg[1])
        finally:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def get_training_state(self) -> dict[str, Any]:
        with self._training_lock:
            return {
                "training": self._is_training,
                "progress": self._train_progress,
                "message": self._train_message,
                "error": self._train_error,
                "result": self._train_result,
            }
