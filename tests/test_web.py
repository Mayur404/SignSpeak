"""Unit tests for SignSpeak Web Application and REST API."""
from __future__ import annotations

import os
import unittest
import hashlib
import io
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app as app_entry
import numpy as np
from signspeak.config import Settings
from signspeak.paths import USER_MODEL
from signspeak.web.server import create_app
from signspeak.web.streamer import WebStreamEngine


_ISOLATION: dict = {}


def setUpModule():
    # Keep the tests away from the real project files: camera presets rewrite
    # CAMERA_HOST in .env and every settings change is saved to
    # data/settings.json.
    _ISOLATION["tmp"] = tempfile.TemporaryDirectory()
    _ISOLATION["camera_host"] = os.environ.get("CAMERA_HOST")
    _ISOLATION["patchers"] = [
        patch("signspeak.web.server.APP_ROOT", Path(_ISOLATION["tmp"].name)),
        patch("signspeak.web.streamer.save_settings"),
    ]
    for patcher in _ISOLATION["patchers"]:
        patcher.start()


def tearDownModule():
    for patcher in reversed(_ISOLATION["patchers"]):
        patcher.stop()
    if _ISOLATION["camera_host"] is None:
        os.environ.pop("CAMERA_HOST", None)
    else:
        os.environ["CAMERA_HOST"] = _ISOLATION["camera_host"]
    _ISOLATION["tmp"].cleanup()


class WebAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create engine without auto-opening real camera
        settings = Settings(camera_source="test_dummy_source")
        cls.engine = WebStreamEngine(settings)
        cls.app = create_app(cls.engine)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.engine.stop()

    def test_index_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"SignSpeak", res.data)
        self.assertIn(b"Live translate", res.data)
        self.assertIn(b"Model active", res.data)
        self.assertNotIn(b"Use bundled", res.data)

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("camera", data)
        self.assertIn("recognition", data)
        self.assertIn("composer", data)
        self.assertIn("model", data)
        self.assertEqual(data["model"]["kind"], "user")

    def test_api_dictionary(self):
        res = self.client.get("/api/dictionary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("vocabulary", data)
        self.assertGreater(len(data["vocabulary"]), 25)

    def test_api_system(self):
        res = self.client.get("/api/system")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["default_port"], int(os.environ.get("WEB_PORT", 18473)))
        self.assertEqual(data["modem_direct_ip"], "192.168.128.1")
        self.assertEqual(data["camera_login_url"], "http://192.168.128.10:8080")

    def test_composer_actions(self):
        # Type letters
        self.client.post("/api/action", json={"action": "type", "param": "H"})
        self.client.post("/api/action", json={"action": "type", "param": "I"})
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["composer"]["word"], "HI")

        # Space
        self.client.post("/api/action", json={"action": "space"})
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["composer"]["word"], "")
        self.assertEqual(status["composer"]["transcript"], "HI")

        # Period
        self.client.post("/api/action", json={"action": "period"})
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["composer"]["transcript"], "HI.")

        # Clear
        self.client.post("/api/action", json={"action": "clear"})
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["composer"]["transcript"], "")

    def test_camera_preset_switching(self):
        # Mode 1: Laptop webcam
        res = self.client.post("/api/camera", json={"preset": "webcam"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["source"], "0")
        self.assertEqual(data["mode"], "laptop")

        res2 = self.client.post("/api/camera", json={"mode": "laptop"})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["source"], "0")

        # Mode 2: Direct Ethernet (192.168.128.10)
        res_eth = self.client.post("/api/camera", json={"mode": "ethernet"})
        self.assertEqual(res_eth.status_code, 200)
        eth_data = res_eth.get_json()
        self.assertEqual(eth_data["mode"], "ethernet")
        self.assertIn("192.168.128.10", eth_data["source"])
        self.assertTrue(eth_data["source"].endswith("/avstream/channel=1/stream=1.sdp"))

        # Mode 3: 5G Modem (dynamic IP, e.g. 10.101.0.73)
        res_5g = self.client.post("/api/camera", json={"mode": "5g", "host": "10.101.0.73"})
        self.assertEqual(res_5g.status_code, 200)
        g_data = res_5g.get_json()
        self.assertEqual(g_data["mode"], "5g")
        self.assertEqual(g_data["host_5g"], "10.101.0.73")
        self.assertIn("10.101.0.73", g_data["source"])

        # Switch back to Laptop webcam 0 for local testing
        res_back = self.client.post("/api/camera", json={"mode": "laptop"})
        self.assertEqual(res_back.status_code, 200)
        self.assertEqual(res_back.get_json()["source"], "0")

    def test_camera_cli_override_reaches_web_engine(self):
        camera = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
        settings = Settings(camera_source="0")
        with patch.object(app_entry, "load_settings", return_value=settings), patch(
            "signspeak.web.streamer.WebStreamEngine"
        ) as engine_type, patch("signspeak.web.server.run_server") as run_server:
            self.assertEqual(
                app_entry.run_web_app(camera=camera, open_browser=False), 0
            )
        engine_type.assert_called_once_with(settings)
        run_server.assert_called_once_with(
            host="0.0.0.0", port=18473, engine=engine_type.return_value, https=False
        )
        self.assertEqual(settings.camera_source, camera)

    def test_training_completion_activates_and_reports_result(self):
        features = np.zeros((10, 126), dtype=np.float32)
        labels = np.array(["A"] * 5 + ["B"] * 5)
        groups = np.repeat(np.arange(2), 5)
        self.engine.use_training_subprocess = False  # keep the mocked trainer in-process
        with patch.object(
            self.engine.store, "training_data", return_value=(features, labels, groups)
        ), patch(
            "signspeak.web.streamer.train_model",
            return_value=SimpleNamespace(algorithm="Test model", accuracy=0.9),
        ), patch.object(self.engine, "switch_model", return_value=True) as switch_model:
            self.assertTrue(self.engine.train_personal_model_async())
            deadline = time.monotonic() + 3
            while self.engine.get_training_state()["training"] and time.monotonic() < deadline:
                time.sleep(0.01)

        state = self.engine.get_training_state()
        self.assertFalse(state["training"])
        self.assertIsNone(state["error"])
        self.assertEqual(state["result"]["algorithm"], "Test model")
        switch_model.assert_called_once_with("user")

    def test_invalid_inputs_return_client_errors(self):
        self.assertEqual(self.client.post("/api/action", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/api/model", json={"kind": "legacy"}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/camera", json={"mode": "5g", "host": "invalid-host"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/record/start", json={"label": "A", "duration": "bad"}).status_code,
            400,
        )
        self.assertEqual(self.client.post("/api/dataset/add_word", json={"word": 12}).status_code, 400)

    def test_invalid_model_upload_keeps_existing_personal_model(self):
        before = hashlib.sha256(USER_MODEL.read_bytes()).digest() if USER_MODEL.exists() else None
        response = self.client.post(
            "/api/model/upload",
            data={"file": (io.BytesIO(b"not a joblib model"), "broken.joblib")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        if before is not None:
            self.assertEqual(hashlib.sha256(USER_MODEL.read_bytes()).digest(), before)

    def test_settings_update(self):
        res = self.client.post("/api/settings", json={"hold_ms": 650})
        self.assertEqual(res.status_code, 200)
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["settings"]["hold_ms"], 650)

    def test_invalid_settings_are_rejected_and_leave_settings_intact(self):
        before = self.engine.settings.to_dict()
        res = self.client.post("/api/settings", json={"hold_ms": 900, "inference_width": "abc"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.engine.settings.to_dict(), before)

    def test_status_events_are_not_consumed_by_other_readers(self):
        with self.engine._lock:
            self.engine._event_seq += 1
            seq = self.engine._event_seq
            self.engine._recent_events.append({"seq": seq, "sign": "HELLO", "kind": "word",
                                               "speak_word": "HELLO", "speak_sentence": None})
        since = seq - 1
        # A plain status read (e.g. opening Settings) returns no events...
        self.assertEqual(self.client.get("/api/status").get_json()["events"], [])
        # ...and two viewers both receive the same event.
        for _ in range(2):
            data = self.client.get(f"/api/status?since={since}").get_json()
            self.assertEqual([ev["sign"] for ev in data["events"]], ["HELLO"])
            self.assertGreaterEqual(data["event_seq"], seq)
        self.assertEqual(self.client.get(f"/api/status?since={seq}").get_json()["events"], [])

    def test_custom_stream_url_is_used_exactly(self):
        host = os.environ.get("CAMERA_HOST", "").strip() or "10.101.0.74"
        main = f"rtsp://admin:admin123@{host}:554/avstream/channel=1/stream=0.sdp"
        res = self.client.post("/api/camera", json={"source": main})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["source"], main)
        self.assertEqual(res.get_json()["mode"], "custom")
        self.client.post("/api/camera", json={"mode": "laptop"})

    def test_empty_camera_host_falls_back_to_default(self):
        with patch.dict(os.environ, {"CAMERA_HOST": ""}):
            res = self.client.post("/api/camera", json={"mode": "5g"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("@10.101.0.74:554", res.get_json()["source"])
        self.client.post("/api/camera", json={"mode": "laptop"})

    def test_concurrent_text_actions_do_not_lose_letters(self):
        import threading

        self.client.post("/api/action", json={"action": "clear"})

        def type_letters():
            for _ in range(200):
                self.engine.composer_action("type", "A")

        threads = [threading.Thread(target=type_letters) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(self.engine.composer.word), 800)
        self.client.post("/api/action", json={"action": "clear"})

    def test_dataset_and_custom_words(self):
        res = self.client.get("/api/dataset")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_samples", data)
        self.assertIn("labels", data)

        # Add custom word
        res = self.client.post("/api/dataset/add_word", json={"word": "THANKYOU"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["display"], "THANKYOU")
        self.assertTrue(data["label"].startswith("__WORD__:"))

        # Check record status
        res = self.client.get("/api/record/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("recording", data)

        # Check train status
        res = self.client.get("/api/train/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("training", data)

    def test_sarvam_tts_languages(self):
        res = self.client.get("/api/tts/languages")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("languages", data)
        codes = [l["code"] for l in data["languages"]]
        self.assertIn("hi-IN", codes)
        self.assertIn("ta-IN", codes)
        self.assertIn("te-IN", codes)

    def test_tts_endpoint_validation(self):
        # Empty text should return 400
        res = self.client.get("/api/tts")
        self.assertEqual(res.status_code, 400)
        res = self.client.post("/api/tts", json={"text": ""})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()



class CameraHostTestCase(unittest.TestCase):
    def test_saved_5g_url_follows_current_camera_host(self):
        from signspeak.web.streamer import _with_current_5g_host

        old = "rtsp://admin:admin123@10.101.0.79:554/avstream/channel=1/stream=1.sdp"
        self.assertEqual(
            _with_current_5g_host(old, "10.101.0.82"),
            "rtsp://admin:admin123@10.101.0.82:554/avstream/channel=1/stream=1.sdp",
        )
        # Direct Ethernet and custom streams are never rewritten.
        eth = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
        self.assertEqual(_with_current_5g_host(eth, "10.101.0.82"), eth)
        custom = "rtsp://user:pw@10.101.0.79:554/live/other"
        self.assertEqual(_with_current_5g_host(custom, "10.101.0.82"), custom)

    def test_open_abandons_stale_source_after_switch(self):
        engine = WebStreamEngine(Settings(camera_source="0"))
        old = "rtsp://admin:admin123@10.101.0.79:554/avstream/channel=1/stream=1.sdp"
        engine._camera_source = "rtsp://admin:admin123@10.101.0.82:554/avstream/channel=1/stream=1.sdp"
        with patch("signspeak.web.streamer.cv2.VideoCapture") as capture:
            self.assertIsNone(engine._open_camera(old))
        capture.assert_not_called()


class BrowserCameraTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = WebStreamEngine(Settings(camera_source="0"))
        self.client = create_app(self.engine).test_client()
        import cv2
        ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 120, dtype=np.uint8))
        self.jpeg = jpeg.tobytes()

    def test_laptop_mode_from_remote_browser_uses_browser_camera(self):
        res = self.client.post(
            "/api/camera", json={"mode": "laptop"}, environ_base={"REMOTE_ADDR": "10.20.30.40"}
        )
        self.assertEqual(res.get_json()["source"], "browser")
        self.assertEqual(res.get_json()["mode"], "laptop")
        # Local browser still opens the server's own webcam.
        res_local = self.client.post("/api/camera", json={"mode": "laptop"})
        self.assertEqual(res_local.get_json()["source"], "0")

    def test_browser_frames_accepted_only_when_active(self):
        res = self.client.post("/api/browser_frame", data=self.jpeg, content_type="image/jpeg")
        self.assertEqual(res.status_code, 409)

        self.engine.switch_camera("browser")
        source = self.engine._open_camera("browser")
        res = self.client.post("/api/browser_frame", data=self.jpeg, content_type="image/jpeg")
        self.assertEqual(res.status_code, 204)
        ok, frame = source.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(source.read()[0], False)  # newest frame is returned once

        bad = self.client.post("/api/browser_frame", data=b"not a jpeg", content_type="image/jpeg")
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.client.post("/api/browser_frame").status_code, 400)

    def test_status_reports_browser_source_as_laptop(self):
        self.engine.switch_camera("browser")
        cam = self.client.get("/api/status").get_json()["camera"]
        self.assertEqual(cam["raw_source"], "browser")
        self.assertEqual(cam["mode"], "laptop")


class InterruptibleRtspOpenTestCase(unittest.TestCase):
    def test_switching_camera_abandons_slow_rtsp_connect(self):
        import threading

        engine = WebStreamEngine(Settings(camera_source="0"))
        url = "rtsp://admin:admin123@10.101.0.85:554/avstream/channel=1/stream=1.sdp"
        engine._camera_source = url
        engine._running = True
        release_open = threading.Event()

        class SlowCapture:
            def __init__(self, *args, **kwargs):
                release_open.wait(5)

            def isOpened(self):
                return True

            def release(self):
                pass

        def switch_soon():
            time.sleep(0.3)
            engine.switch_camera("browser")

        threading.Thread(target=switch_soon, daemon=True).start()
        started = time.monotonic()
        with patch("signspeak.web.streamer.cv2.VideoCapture", SlowCapture):
            cap, switched = engine._open_rtsp_interruptible(url, url)
        release_open.set()
        engine._running = False
        self.assertTrue(switched)
        self.assertIsNone(cap)
        self.assertLess(time.monotonic() - started, 2.0)


class RtspReaderTestCase(unittest.TestCase):
    def test_close_never_releases_capture_during_a_blocked_read(self):
        import threading
        from signspeak.web.streamer import _RtspReader

        in_read, finish_read = threading.Event(), threading.Event()
        events = []

        class BlockingCapture:
            def read(self):
                in_read.set()
                finish_read.wait(5)
                events.append("read-returned")
                return False, None

            def release(self):
                events.append(("release", threading.current_thread().name))

        reader = _RtspReader(BlockingCapture())
        self.assertTrue(in_read.wait(2))
        started = time.monotonic()
        reader.close()  # read still blocked
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(events, [])  # not released mid-read
        finish_read.set()
        reader._thread.join(2)
        self.assertEqual(events[0], "read-returned")
        self.assertEqual(events[1], ("release", "camera-reader"))

    def test_read_error_is_reported_as_failed(self):
        from signspeak.web.streamer import _RtspReader

        class BrokenCapture:
            def read(self):
                raise RuntimeError("stream gone")

            def release(self):
                pass

        reader = _RtspReader(BrokenCapture())
        reader._thread.join(2)
        self.assertTrue(reader.failed)
        self.assertEqual(reader.read(), (False, None))
