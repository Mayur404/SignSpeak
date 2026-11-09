"""Flask Web Server for SignSpeak Dashboard."""
from __future__ import annotations

import os
import socket
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from ..labels import SIGN_HINTS, TRAIN_LABELS, display_name
from .streamer import WebStreamEngine


def get_network_ips() -> list[str]:
    """Discover host IP addresses for display on banner and UI."""
    ips = ["127.0.0.1"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        s.close()
        if primary not in ips:
            ips.append(primary)
    except Exception:
        pass
    return ips


def create_app(engine: WebStreamEngine | None = None) -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    if engine is None:
        engine = WebStreamEngine()
    engine.start()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        """MJPEG video stream endpoint for browsers and mobile devices."""
        return Response(
            engine.generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        """Live status of camera, recognition, text composer, and model."""
        return jsonify(engine.get_state())

    @app.route("/api/action", methods=["POST"])
    def api_action():
        """Trigger text composition actions (space, period, backspace, clear, type)."""
        data = request.get_json(silent=True) or {}
        action = data.get("action", "")
        param = data.get("param", "")
        result = engine.composer_action(action, param)
        return jsonify({"status": "ok", "action": action, "result": result})

    @app.route("/api/camera", methods=["POST"])
    def api_camera():
        """Switch camera stream source or select a 5G / Ethernet / Webcam preset."""
        data = request.get_json(silent=True) or {}
        preset = data.get("preset", "").lower()
        source = data.get("source", "").strip()

        env_host = os.environ.get("CAMERA_HOST", "10.101.0.70").strip()

        if preset == "5g":
            source = f"rtsp://admin:admin123@{env_host}:554/avstream/channel=1/stream=1.sdp"
        elif preset == "ethernet":
            source = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
        elif preset == "webcam":
            source = "0"

        if source:
            engine.switch_camera(source)
            return jsonify({"status": "ok", "source": source})
        return jsonify({"status": "error", "message": "No valid camera source specified"}), 400

    @app.route("/api/settings", methods=["POST"])
    def api_settings():
        """Update recognition or display settings live."""
        data = request.get_json(silent=True) or {}
        engine.update_settings(data)
        return jsonify({"status": "ok", "settings": engine.settings.to_dict()})

    @app.route("/api/model", methods=["POST"])
    def api_model():
        """Switch active model between 'user' and 'legacy'."""
        data = request.get_json(silent=True) or {}
        kind = data.get("kind", "user")
        success = engine.switch_model(kind)
        if success:
            return jsonify({"status": "ok", "model": kind})
        return jsonify({"status": "error", "message": f"Failed to switch model to {kind}"}), 400

    @app.route("/api/pause", methods=["POST"])
    def api_pause():
        """Toggle recognition pause."""
        paused = engine.toggle_pause()
        return jsonify({"status": "ok", "paused": paused})

    @app.route("/api/dictionary")
    def api_dictionary():
        """Return full ASL & custom sign dictionary with descriptions."""
        vocabulary = []
        for label in TRAIN_LABELS:
            vocabulary.append({
                "label": label,
                "display": display_name(label),
                "hint": SIGN_HINTS.get(label, "Sign gesture for " + display_name(label)),
            })
        return jsonify({"vocabulary": vocabulary})

    @app.route("/api/system")
    def api_system():
        """Return system and 5G network configuration guide."""
        env_host = os.environ.get("CAMERA_HOST", "10.101.0.70").strip()
        return jsonify({
            "ips": get_network_ips(),
            "camera_env_host": env_host,
            "modem_direct_ip": "192.168.128.1",
            "camera_direct_ip": "192.168.128.10",
            "default_port": 8080,
            "edge_server_ip": "172.16.0.10",
        })

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the SignSpeak Web Server."""
    engine = WebStreamEngine()
    app = create_app(engine)

    ips = get_network_ips()
    primary_ip = ips[-1] if len(ips) > 1 else "127.0.0.1"

    print("=" * 70)
    print("  SignSpeak — 5G Edge Web UI")
    print("=" * 70)
    print(f"  * Local browser:    http://localhost:{port}")
    print(f"  * Wi-Fi / LAN:      http://{primary_ip}:{port}")
    print(f"  * 5G Edge VM:       http://172.16.0.10:{port} (if on edge VM)")
    print(f"  * Camera:           {engine.settings.camera_source or 'Default webcam / 5G'}")
    print("=" * 70)
    print("  Press Ctrl+C to stop.\n")

    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    finally:
        engine.stop()
