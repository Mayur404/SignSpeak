"""SignSpeak — real-time ASL fingerspelling to text and speech."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from signspeak.config import load_settings
from signspeak.ui.main_window import MainWindow
from signspeak.ui.theme import apply_theme
from signspeak.vision import warm_tracker


def main() -> int:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "SignSpeak.ASL.1"
            )
        except Exception:
            pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("SignSpeak")
    app.setOrganizationName("SignSpeak")
    app.setApplicationVersion("1.0.0")
    apply_theme(app)

    start_camera = "--no-camera" not in sys.argv
    enable_speech = "--no-speech" not in sys.argv
    settings = load_settings()

    # Load the MediaPipe model while Qt builds the window. The camera thread
    # then waits only for the remaining warm-up instead of starting it cold.
    if start_camera:
        warm_tracker()

    try:
        window = MainWindow(settings, start_camera=start_camera, enable_speech=enable_speech)
    except Exception as exc:
        detail = str(exc)
        normalized = detail.lower()
        if "model" in normalized and any(term in normalized for term in ("missing", "not found", "no such file")):
            detail += "\n\nIf a model file is missing, reinstall the app assets."
        QMessageBox.critical(
            None,
            "SignSpeak could not start",
            detail,
        )
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
