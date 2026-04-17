"""main.py — Application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from config_loader import load_config
from main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    try:
        cfg = load_config(Path(__file__).parent / "config.yaml")
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Configuration Error", str(exc))
        sys.exit(1)
    window = MainWindow(cfg)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
