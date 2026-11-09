"""Unit tests for SignSpeak Web Application and REST API."""
from __future__ import annotations

import unittest

from signspeak.config import Settings
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
        self.assertIn(b"5G MEC EDGE", res.data)

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("camera", data)
        self.assertIn("recognition", data)
        self.assertIn("composer", data)
        self.assertIn("model", data)

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
        res = self.client.post("/api/camera", json={"preset": "webcam"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["source"], "0")

    def test_settings_update(self):
        res = self.client.post("/api/settings", json={"hold_ms": 650})
        self.assertEqual(res.status_code, 200)
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["settings"]["hold_ms"], 650)


if __name__ == "__main__":
    unittest.main()

