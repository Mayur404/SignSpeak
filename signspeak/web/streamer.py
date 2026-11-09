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
from urllib.parse import urlsplit

import cv2
import numpy as np

from ..composer import TextComposer
from ..config import Settings, load_settings, save_settings
from ..labels import (
    BACKSPACE,
    CUSTOM_WORD_PREFIX,
    DISPLAY_NAMES,
    PERIOD,
    SPACE,
    display_name,
    spoken_name,
)
from ..paths import APP_ROOT, LEGACY_MODEL, USER_MODEL
from ..recognition import (
    ModelLoadError,
    RecognitionEngine,
    RecognitionState,
    load_classifier,
)
from ..vision import (
    HAND_CONNECTIONS,
    HandTracker,
    Observation,
    _LatestFrameReader,
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


def _draw_rounded_rect(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
    alpha: float = 1.0,
) -> None:
    """Draw a smooth rounded rectangle with optional alpha transparency."""
    x1, y1 = p1
    x2, y2 = p2
    w = x2 - x1
    h = y2 - y1
    radius = min(radius, w // 2, h // 2)

    if alpha < 0.999:
        overlay = image.copy()
        _draw_rounded_rect(overlay, p1, p2, color, radius, thickness, 1.0)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        return

    # Interior rectangles
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    # 4 rounded corners
    for cx, cy in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(image, (cx, cy), radius, color, thickness, cv2.LINE_AA)


class WebStreamEngine:
    """Core video streamer and recognition engine for web clients."""

    def __init__(self, settings: Settings | None = None):
        self.settings = (settings or load_settings()).copy()
        self.composer = TextComposer()

        # Load active classifier (default to 'user' if present, fallback to 'legacy')
        self.classifier = None
        self.model_kind = self.settings.active_model
        try:
            self.classifier = load_classifier(self.model_kind)
        except Exception:
            try:
                self.model_kind = "user" if self.model_kind == "legacy" else "legacy"
                self.classifier = load_classifier(self.model_kind)
            except Exception:
                self.classifier = None

        if self.classifier:
            self.engine = RecognitionEngine(self.classifier, self.settings)
        else:
            self.engine = None

        # Camera & capture state
        self._camera_source = self._resolve_initial_source()
        self._cap = None
        self._tracker: HandTracker | None = None
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)

        # Output buffers
        self._latest_jpeg: bytes | None = None
        self._latest_state = RecognitionState()
        self._latest_fps = 0.0
        self._last_error: str | None = None
        self._camera_connected = False
        self._camera_dims = (1280, 720)
        self._recent_events: list[dict[str, Any]] = []

        # Start warm-up
        warm_tracker()

    def _resolve_initial_source(self) -> str:
        """Pick the smartest camera source from env, settings, or default."""
        env_host = os.environ.get("CAMERA_HOST", "").strip()
        if self.settings.camera_source:
            return self.settings.camera_source
        if env_host:
            return f"rtsp://admin:admin123@{env_host}:554/avstream/channel=1/stream=1.sdp"
        return "0"

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
        self._camera_connected = False

    def _open_camera(self, source: str):
        """Open camera from integer webcam index, RTSP URL, or file path."""
        source = source.strip()
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
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
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

            # 2. Open camera if not opened
            if self._cap is None:
                cap = self._open_camera(self._camera_source)
                if cap is None:
                    self._camera_connected = False
                    self._last_error = f"Cannot open camera: {_safe_camera_source(self._camera_source)}"
                    # Generate animated standby placeholder frame so browser stream stays active
                    self._render_placeholder_frame("Connecting to camera...", self._camera_source)
                    time.sleep(1.0)
                    continue
                self._cap = cap
                self._camera_connected = True
                self._last_error = None
                consecutive_failures = 0

            # 3. Read frame
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    self._camera_connected = False
                    self._close_camera()
                    time.sleep(0.5)
                else:
                    time.sleep(0.02)
                continue

            consecutive_failures = 0
            now = time.monotonic()
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

            # 5. Recognition Engine update
            timestamp_ms = int(now * 1000)
            if self.engine:
                state = self.engine.update(points, timestamp_ms)
            else:
                state = RecognitionState(hand_present=points is not None)

            # 6. Check for committed sign -> Composer
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
            (tw, th), _ = cv2.getTextSize(badge_str, font, scale, 2)
            chip_w, chip_h = tw + 32, th + 24
            cx = min(max(x1, 14), w - chip_w - 14)
            cy = max(14, y1 - chip_h - 14)

            _draw_rounded_rect(frame, (cx, cy), (cx + chip_w, cy + chip_h), DARK_BG, 12, -1, 0.75)
            _draw_rounded_rect(frame, (cx, cy), (cx + chip_w, cy + chip_h), hand_color, 12, 2, 0.9)
            cv2.putText(
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
                self._frame_cond.notify_all()

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG stream to HTTP browser client."""
        while self._running:
            with self._frame_cond:
                self._frame_cond.wait(timeout=1.0)
                frame_data = self._latest_jpeg
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
                "accuracy": round(info.accuracy * 100, 1) if info.accuracy else None,
                "classes_count": len(self.classifier.classes),
                "feature_mode": info.feature_mode,
            }

        return {
            "camera": {
                "connected": self._camera_connected,
                "source": _safe_camera_source(self._camera_source),
                "raw_source": self._camera_source,
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
        new_source = new_source.strip()
        if not new_source:
            return
        with self._lock:
            self._camera_source = new_source
            self.settings.camera_source = new_source
            save_settings(self.settings)
            self._close_camera()

    def switch_model(self, kind: str) -> bool:
        """Switch between 'user' trained model and 'legacy' bundled model."""
        if kind not in ("user", "legacy"):
            return False
        try:
            new_classifier = load_classifier(kind)
            self.classifier = new_classifier
            self.model_kind = kind
            self.settings.active_model = kind
            save_settings(self.settings)
            if self.engine:
                self.engine.set_classifier(new_classifier)
            else:
                self.engine = RecognitionEngine(new_classifier, self.settings)
            return True
        except Exception as exc:
            self._last_error = f"Cannot switch model to {kind}: {exc}"
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
        elif action == "clear":
            self.composer.clear_all()
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
