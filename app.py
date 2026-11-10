"""SignSpeak — 5G Edge ASL & Gesture Recognition Web UI."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser



def _load_env_files() -> None:
    """Load KEY=VALUE pairs from .env, then .env.example as a fallback.

    Must run before signspeak.vision is imported, because it reads
    CAMERA_HOST at import time. Real environment variables always win, and
    placeholder values from the example file are skipped.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    for name in (".env", ".env.example"):
        try:
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not sep or not key or key in os.environ:
                continue
            if not value or value.startswith("your_"):
                continue
            os.environ[key] = value


_load_env_files()

from signspeak.config import load_settings  # noqa: E402


def run_desktop_app(args: list[str]) -> int:
    """Run the legacy PySide6 desktop GUI."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
    from signspeak.ui.main_window import MainWindow
    from signspeak.ui.theme import apply_theme
    from signspeak.vision import warm_tracker

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SignSpeak.ASL.1")
        except Exception:
            pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(args)
    app.setApplicationName("SignSpeak")
    app.setOrganizationName("SignSpeak")
    app.setApplicationVersion("1.0.0")
    apply_theme(app)

    start_camera = "--no-camera" not in args
    enable_speech = "--no-speech" not in args
    settings = load_settings()

    if start_camera:
        warm_tracker()

    try:
        window = MainWindow(settings, start_camera=start_camera, enable_speech=enable_speech)
    except Exception as exc:
        detail = str(exc)
        normalized = detail.lower()
        if "model" in normalized and any(term in normalized for term in ("missing", "not found", "no such file")):
            detail += "\n\nIf a model file is missing, reinstall the app assets."
        QMessageBox.critical(None, "SignSpeak could not start", detail)
        return 1

    window.show()
    return app.exec()


def run_web_app(host: str = "0.0.0.0", port: int = 18473, camera: str | None = None, open_browser: bool = True, https: bool = False) -> int:
    """Run the SignSpeak Web Server (default port 18473; 8080 is NiralOS)."""
    from signspeak.web.server import run_server
    from signspeak.web.streamer import WebStreamEngine

    settings = load_settings()
    if camera:
        settings.camera_source = camera

    if open_browser:
        def _open():
            time.sleep(1.2)
            try:
                scheme = "https" if https else "http"
                target_url = f"{scheme}://localhost:{port}" if host in ("0.0.0.0", "127.0.0.1") else f"{scheme}://{host}:{port}"
                webbrowser.open(target_url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    engine = WebStreamEngine(settings)
    run_server(host=host, port=port, engine=engine, https=https)
    return 0


DEFAULT_PORT = 18473


def _env_port() -> int:
    """WEB_PORT from the environment / .env, falling back to 18473."""
    raw = os.environ.get("WEB_PORT", "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        print(f"WARNING: WEB_PORT={raw!r} in .env is not a valid port (1-65535); using {DEFAULT_PORT}.")
        return DEFAULT_PORT
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description="SignSpeak — ASL & Gesture Recognition")
    parser.add_argument("--desktop", action="store_true", help="Launch the legacy Qt desktop GUI")
    parser.add_argument("--host", default=os.environ.get("WEB_HOST", "").strip() or "0.0.0.0", help="Host to bind (default: WEB_HOST in .env, else 0.0.0.0)")
    parser.add_argument("--port", type=int, default=_env_port(), help=f"Port to bind (default: WEB_PORT in .env, else {DEFAULT_PORT})")
    parser.add_argument("--camera", type=str, default=None, help="Override camera RTSP URL or device index")
    parser.add_argument(
        "--http",
        action="store_true",
        default=os.environ.get("WEB_HTTPS", "1").strip().lower() in ("0", "false", "no", "off"),
        help="Serve plain HTTP instead of the default HTTPS (camera/mic then only work on localhost)",
    )
    parser.add_argument("--https", action="store_true", help=argparse.SUPPRESS)  # old flag; HTTPS is now the default
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    # If --desktop is specified, forward to desktop app
    if "--desktop" in sys.argv:
        return run_desktop_app(sys.argv)

    args, _ = parser.parse_known_args()
    return run_web_app(
        host=args.host,
        port=args.port,
        camera=args.camera,
        https=args.https or not args.http,
        open_browser=not args.no_browser and "SSH_CONNECTION" not in os.environ and sys.platform == "win32",
    )


if __name__ == "__main__":
    raise SystemExit(main())
