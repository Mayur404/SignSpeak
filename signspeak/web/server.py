"""Flask Web Server for SignSpeak Dashboard."""
from __future__ import annotations

import base64
import ipaddress
import json
import math
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from ..labels import SIGN_HINTS, TRAIN_LABELS, display_name, is_trainable_label
from ..paths import APP_ROOT, USER_MODEL
from ..recognition import ModelLoadError, load_user_classifier
from ..speech_input import transcribe_audio
from ..sarvam_env import SARVAM_LANGUAGES, _load_project_env
from .streamer import WebStreamEngine


def get_network_ips() -> list[str]:
    """Discover host IP addresses for display on banner and UI."""
    ips = ["127.0.0.1"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary = s.getsockname()[0]
        if primary not in ips:
            ips.append(primary)
    except OSError:
        return ips
    return ips


def update_env_camera_host(new_host: str) -> None:
    """Persist updated 5G dynamic IP into .env and runtime environment."""
    new_host = new_host.strip()
    if not new_host:
        raise ValueError("A camera IP address is required.")
    env_path = APP_ROOT / ".env"
    temp_path = env_path.with_name(env_path.name + ".tmp")
    try:
        lines = []
        found = False
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("CAMERA_HOST="):
                    lines.append(f"CAMERA_HOST={new_host}")
                    found = True
                else:
                    lines.append(line)
        if not found:
            lines.append(f"CAMERA_HOST={new_host}")
        temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temp_path, env_path)
        os.environ["CAMERA_HOST"] = new_host
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(f"Could not save the 5G camera IP to .env: {exc}") from exc


def create_app(engine: WebStreamEngine | None = None) -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": exc.description}), exc.code
        return exc

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        app.logger.exception("Unhandled SignSpeak request error")
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": "The request failed unexpectedly."}), 500
        return "The page could not be loaded. Check the server log for details.", 500

    if engine is None:
        engine = WebStreamEngine()
    engine.start()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(os.path.join(APP_ROOT, "assets"), filename)

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
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Action must be a JSON object."}), 400
        action = data.get("action", "")
        param = data.get("param", "")
        if not isinstance(action, str) or not isinstance(param, str):
            return jsonify({"status": "error", "message": "Action and parameter must be text."}), 400
        if action.lower() not in {"space", "period", "backspace", "clear", "clear_all", "clear_word", "delete_word", "type", "commit_now"}:
            return jsonify({"status": "error", "message": "Unsupported text action."}), 400
        result = engine.composer_action(action, param)
        return jsonify({"status": "ok", "action": action, "result": result})

    @app.route("/api/camera", methods=["POST"])
    def api_camera():
        """Switch camera between 3 modes: Laptop Webcam (0), Direct Ethernet, or 5G Modem (dynamic IP)."""
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Camera selection must be a JSON object."}), 400
        raw_mode = data.get("mode", data.get("preset", "")) or ""
        raw_source = data.get("source", "") or ""
        raw_host = data.get("host", data.get("ip", "")) or ""
        if not isinstance(raw_mode, str) or not isinstance(raw_host, str) or not isinstance(raw_source, (str, int)):
            return jsonify({"status": "error", "message": "Camera mode, source, and host must be text."}), 400
        mode = raw_mode.lower().strip()
        source = str(raw_source).strip()
        new_5g_ip = raw_host.strip()

        _load_project_env()
        if new_5g_ip and mode in ("5g", "sparsh", "wan", "wifi", "modem"):
            try:
                if ipaddress.ip_address(new_5g_ip).version != 4:
                    raise ValueError
            except ValueError:
                return jsonify({"status": "error", "message": "Enter a valid IPv4 camera address."}), 400
            try:
                update_env_camera_host(new_5g_ip)
            except OSError as exc:
                app.logger.error("Could not save camera host configuration: %s", exc)
                return jsonify({"status": "error", "message": str(exc)}), 500
            env_host = new_5g_ip
        else:
            env_host = os.environ.get("CAMERA_HOST", "10.101.0.74").strip()

        if mode in ("webcam", "laptop", "0", "cam0") or source in ("0", "webcam", "laptop"):
            resolved_source = "0"
            resolved_mode = "laptop"
        elif mode in ("ethernet", "eth", "192.168.128.10") or "192.168.128.10" in source:
            resolved_source = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
            resolved_mode = "ethernet"
        elif mode in ("5g", "sparsh", "wan", "wifi", "modem") or env_host in source or (not source and mode == "5g"):
            resolved_source = f"rtsp://admin:admin123@{env_host}:554/avstream/channel=1/stream=1.sdp"
            resolved_mode = "5g"
        elif source:
            resolved_source = source
            resolved_mode = "custom"
        elif mode:
            return jsonify({"status": "error", "message": f"Unsupported camera mode: {mode}"}), 400
        else:
            resolved_source = "0"
            resolved_mode = "laptop"

        engine.switch_camera(resolved_source)
        return jsonify({
            "status": "ok",
            "source": resolved_source,
            "mode": resolved_mode,
            "host_5g": env_host,
        })

    @app.route("/api/settings", methods=["POST"])
    def api_settings():
        """Update recognition or display settings live."""
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Settings must be a JSON object."}), 400
        try:
            engine.update_settings(data)
        except (TypeError, ValueError, OverflowError) as exc:
            return jsonify({"status": "error", "message": f"Invalid settings: {exc}"}), 400
        return jsonify({"status": "ok", "settings": engine.settings.to_dict()})

    @app.route("/api/model", methods=["POST"])
    def api_model():
        """Activate the personal model."""
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Model selection must be a JSON object."}), 400
        kind = data.get("kind", "user")
        if kind != "user":
            return jsonify({"status": "error", "message": "Only the personal model is supported."}), 400
        success = engine.switch_model(kind)
        if success:
            return jsonify({"status": "ok", "model": kind})
        return jsonify({"status": "error", "message": engine._model_error or "Could not load the personal model."}), 400

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
        _load_project_env()
        env_host = os.environ.get("CAMERA_HOST", "10.101.0.74").strip()
        return jsonify({
            "ips": get_network_ips(),
            "camera_5g_host": env_host,
            "camera_env_host": env_host,
            "modem_direct_ip": "192.168.128.1",
            "camera_direct_ip": "192.168.128.10",
            "camera_login_url": "http://192.168.128.10:8080",
            "default_port": 8080,
            "edge_server_ip": "172.16.0.11",
            "edge_hostname": "niralos-video",
        })

    @app.route("/api/dataset")
    def api_dataset():
        """Return counts, take_counts, custom words and dataset info."""
        return jsonify(engine.get_dataset_info())

    @app.route("/api/dataset/add_word", methods=["POST"])
    def api_dataset_add_word():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Word must be sent in a JSON object."}), 400
        word = data.get("word", "")
        if not isinstance(word, str):
            return jsonify({"status": "error", "message": "Word must be text."}), 400
        word = word.strip()
        lbl = engine.add_custom_word(word)
        if lbl:
            return jsonify({"status": "ok", "label": lbl, "display": display_name(lbl)})
        return jsonify({"status": "error", "message": "Invalid custom word"}), 400

    @app.route("/api/dataset/remove_word", methods=["POST"])
    def api_dataset_remove_word():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Word selection must be a JSON object."}), 400
        label = data.get("label", "")
        if not isinstance(label, str):
            return jsonify({"status": "error", "message": "Word label must be text."}), 400
        success = engine.remove_custom_word(label)
        if not success:
            return jsonify({"status": "error", "message": "That custom word does not exist."}), 404
        return jsonify({"status": "ok"})

    @app.route("/api/dataset/clear_label", methods=["POST"])
    def api_dataset_clear_label():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Label selection must be a JSON object."}), 400
        label = data.get("label", "")
        if not isinstance(label, str):
            return jsonify({"status": "error", "message": "Sign label must be text."}), 400
        if not is_trainable_label(label):
            return jsonify({"status": "error", "message": "Choose a valid sign label."}), 400
        engine.clear_label(label)
        return jsonify({"status": "ok"})

    @app.route("/api/dataset/clear_all", methods=["POST"])
    def api_dataset_clear_all():
        engine.clear_dataset()
        return jsonify({"status": "ok"})

    @app.route("/api/record/start", methods=["POST"])
    def api_record_start():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Recording settings must be a JSON object."}), 400
        label = data.get("label", "A")
        if not isinstance(label, str) or not is_trainable_label(label):
            return jsonify({"status": "error", "message": "Choose a valid sign before recording."}), 400
        try:
            duration = float(data.get("duration", 3.0))
            delay = float(data.get("delay", 2.0))
        except (TypeError, ValueError, OverflowError):
            return jsonify({"status": "error", "message": "Recording duration and delay must be numbers."}), 400
        if not math.isfinite(duration) or not 0.5 <= duration <= 30:
            return jsonify({"status": "error", "message": "Recording duration must be between 0.5 and 30 seconds."}), 400
        if not math.isfinite(delay) or not 0 <= delay <= 15:
            return jsonify({"status": "error", "message": "Recording delay must be between 0 and 15 seconds."}), 400
        engine.start_recording(label, duration, delay)
        return jsonify({"status": "ok", "label": label})

    @app.route("/api/record/cancel", methods=["POST"])
    def api_record_cancel():
        engine.cancel_recording()
        return jsonify({"status": "ok"})

    @app.route("/api/record/status")
    def api_record_status():
        return jsonify(engine.get_recording_state())

    @app.route("/api/train", methods=["POST"])
    def api_train():
        started = engine.train_personal_model_async()
        if started:
            return jsonify({"status": "started"})
        return jsonify({"status": "already_running", "message": "Training is already running."}), 409

    @app.route("/api/train/status")
    def api_train_status():
        return jsonify(engine.get_training_state())

    @app.route("/api/model/download")
    def api_model_download():
        target = USER_MODEL if USER_MODEL.exists() else (APP_ROOT / "signmodel.joblib")
        if target.exists():
            return send_file(target, as_attachment=True, download_name="user_model.joblib")
        return jsonify({"status": "error", "message": "No model file found"}), 404

    @app.route("/api/model/upload", methods=["POST"])
    def api_model_upload():
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        f = request.files["file"]
        if not f.filename or Path(f.filename).suffix.lower() != ".joblib":
            return jsonify({"status": "error", "message": "Choose a .joblib personal model file."}), 400

        temp_path: Path | None = None
        backup_path: Path | None = None
        had_model = USER_MODEL.exists()
        USER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".model-upload-", suffix=".joblib", dir=USER_MODEL.parent, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
            f.save(temp_path)
            # Validate the complete model bundle before replacing the active copy.
            load_user_classifier(temp_path)
            if had_model:
                with tempfile.NamedTemporaryFile(
                    prefix=".model-backup-", suffix=".joblib", dir=USER_MODEL.parent, delete=False
                ) as backup_file:
                    backup_path = Path(backup_file.name)
                shutil.copy2(USER_MODEL, backup_path)
            os.replace(temp_path, USER_MODEL)
            temp_path = None
            if not engine.switch_model("user"):
                if had_model and backup_path is not None:
                    os.replace(backup_path, USER_MODEL)
                    backup_path = None
                elif USER_MODEL.exists():
                    USER_MODEL.unlink()
                return jsonify({
                    "status": "error",
                    "message": engine._model_error or "The model could not be activated.",
                }), 400
            return jsonify({"status": "ok", "message": "Personal model uploaded and activated."})
        except (OSError, ModelLoadError, ValueError, TypeError) as exc:
            return jsonify({"status": "error", "message": f"Could not load personal model: {exc}"}), 400
        finally:
            for leftover in (temp_path, backup_path):
                if leftover is not None:
                    try:
                        leftover.unlink(missing_ok=True)
                    except OSError:
                        app.logger.warning("Could not remove temporary model file %s", leftover)

    @app.route("/api/tts", methods=["GET", "POST"])
    def api_tts():
        """Sarvam Bulbul TTS audio synthesis endpoint."""
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                return jsonify({"status": "error", "message": "Speech request must be a JSON object."}), 400
            raw_text = data.get("text", "")
            if not isinstance(raw_text, str):
                return jsonify({"status": "error", "message": "Speech text must be text."}), 400
            text = raw_text.strip()
            lang = data.get("language") or engine.settings.tts_language or "hi-IN"
        else:
            text = str(request.args.get("text", "")).strip()
            lang = request.args.get("language") or engine.settings.tts_language or "hi-IN"

        if not text:
            return jsonify({"status": "error", "message": "No text provided"}), 400
        if not isinstance(lang, str) or lang not in SARVAM_LANGUAGES.values():
            return jsonify({"status": "error", "message": "Choose a supported Sarvam language."}), 400

        _load_project_env()
        api_key = os.environ.get("SARVAM_API_KEY", "").strip()
        if not api_key:
            return jsonify({"status": "error", "message": "SARVAM_API_KEY not configured in .env"}), 400

        pace = max(0.5, min(2.0, 1.0 + engine.settings.tts_rate * 0.035))
        payload = json.dumps({
            "text": text[:2500],
            "language_code": lang,
            "speaker": "shubh",
            "model": "bulbul:v3",
            "pace": pace,
            "output_audio_codec": "wav",
        }).encode("utf-8")
        req = Request(
            "https://api.sarvam.ai/text-to-speech",
            data=payload,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
            audio_bytes = base64.b64decode(res_data["audios"][0])
            return Response(audio_bytes, mimetype="audio/wav")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            return jsonify({"status": "error", "message": f"Sarvam error ({exc.code}): {detail}"}), 502
        except (URLError, TimeoutError) as exc:
            return jsonify({"status": "error", "message": f"Sarvam service is unavailable: {exc}"}), 502
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/stt/transcribe", methods=["POST"])
    def api_stt_transcribe():
        """Sarvam Saaras STT voice phrase transcription endpoint."""
        filename = "phrase.wav"
        if "file" in request.files:
            f = request.files["file"]
            audio_bytes = f.read()
            filename = f.filename or "phrase.wav"
        elif "audio" in request.files:
            f = request.files["audio"]
            audio_bytes = f.read()
            filename = f.filename or "phrase.wav"
        else:
            audio_bytes = request.data
            filename = request.args.get("filename", "phrase.wav")

        if not audio_bytes:
            return jsonify({"status": "error", "message": "No audio received"}), 400

        try:
            transcript, language = transcribe_audio(audio_bytes, filename=filename)
            return jsonify({"status": "ok", "transcript": transcript, "language_code": language})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/tts/languages")
    def api_tts_languages():
        """Return list of Indian languages supported by Sarvam Bulbul."""
        return jsonify({
            "languages": [{"name": name, "code": code} for name, code in SARVAM_LANGUAGES.items()]
        })

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    engine: WebStreamEngine | None = None,
) -> None:
    """Start the SignSpeak Web Server."""
    engine = engine or WebStreamEngine()
    app = create_app(engine)

    ips = get_network_ips()
    primary_ip = ips[-1] if len(ips) > 1 else "127.0.0.1"

    print("=" * 70)
    print("  SignSpeak — 5G Edge Web UI")
    print("=" * 70)
    print(f"  * Local browser:    http://localhost:{port}")
    print(f"  * Wi-Fi / LAN:      http://{primary_ip}:{port}")
    print(f"  * 5G Edge VM (niralos-video): http://172.16.0.11:{port}")
    from ..vision import _safe_camera_source
    cam_src = str(engine._camera_source)
    cam_label = f"webcam {cam_src}" if cam_src.isdigit() else (
        _safe_camera_source(cam_src) if "://" in cam_src else cam_src)
    print(f"  * Camera:           {cam_label}")
    print("=" * 70)
    print("  Press Ctrl+C to stop.\n")

    try:
        # The project loads .env itself in paths.py; disabling Flask's optional
        # dotenv discovery avoids a misleading python-dotenv startup warning.
        app.run(host=host, port=port, threaded=True, debug=False, load_dotenv=False)
    finally:
        engine.stop()

