"""Unit tests for SignSpeak Web Application and REST API."""
from __future__ import annotations

import unittest
import hashlib
import io
import time
from types import SimpleNamespace
from unittest.mock import patch

import app as app_entry
import numpy as np
from signspeak.config import Settings
from signspeak.paths import USER_MODEL
from signspeak.web.server import create_app
from signspeak.web.streamer import WebStreamEngine


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
        self.assertIn(b"Personal model active", res.data)
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
        self.assertEqual(data["default_port"], 8080)
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
            host="0.0.0.0", port=8080, engine=engine_type.return_value
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

