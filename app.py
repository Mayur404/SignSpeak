"""SignSpeak — 5G Edge ASL & Gesture Recognition Web UI."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser

from signspeak.config import load_settings


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


def run_web_app(host: str = "0.0.0.0", port: int = 8080, camera: str | None = None, open_browser: bool = True) -> int:
    """Run the SignSpeak Web Server on port 8080."""
    from signspeak.web.server import run_server
    from signspeak.web.streamer import WebStreamEngine

    settings = load_settings()
    if camera:
        settings.camera_source = camera

    if open_browser:
        def _open():
            time.sleep(1.2)
            try:
                target_url = f"http://localhost:{port}" if host in ("0.0.0.0", "127.0.0.1") else f"http://{host}:{port}"
                webbrowser.open(target_url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    run_server(host=host, port=port)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SignSpeak — ASL & Gesture Recognition")
    parser.add_argument("--desktop", action="store_true", help="Launch the legacy Qt desktop GUI")
    parser.add_argument("--host", default=os.environ.get("WEB_HOST", "0.0.0.0"), help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEB_PORT", 8080)), help="Port to bind (default: 8080)")
    parser.add_argument("--camera", type=str, default=None, help="Override camera RTSP URL or device index")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    # If --desktop is specified, forward to desktop app
    if "--desktop" in sys.argv:
        return run_desktop_app(sys.argv)

    args, _ = parser.parse_known_args()
    return run_web_app(
        host=args.host,
        port=args.port,
        camera=args.camera,
        open_browser=not args.no_browser and "SSH_CONNECTION" not in os.environ and sys.platform == "win32",
    )


if __name__ == "__main__":
    raise SystemExit(main())
