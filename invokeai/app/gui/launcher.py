import sys
from typing import Optional


def ask_ui_mode() -> Optional[str]:
    """Show a minimal PySide6 desktop picker for UI mode.

    Returns ``"local"`` or ``"server"``, or ``None`` if the user closes the
    dialog without choosing.
    """
    try:
        from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
        from PySide6.QtCore import Qt
    except ImportError as exc:
        raise ImportError(
            "PySide6 is required for the local GUI picker. "
            "Install it with: pip install PySide6"
        ) from exc

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = QDialog()
    dialog.setWindowTitle("InvokeAI")
    dialog.setModal(True)
    dialog.setFixedWidth(360)

    label = QLabel("Choose how to start InvokeAI")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    local_btn = QPushButton("Local")
    server_btn = QPushButton("Server")

    button_layout = QHBoxLayout()
    button_layout.addWidget(local_btn)
    button_layout.addWidget(server_btn)

    layout = QVBoxLayout(dialog)
    layout.addWidget(label)
    layout.addLayout(button_layout)

    result: dict[str, Optional[str]] = {"mode": None}

    def choose(mode: str) -> None:
        result["mode"] = mode
        dialog.accept()

    local_btn.clicked.connect(lambda: choose("local"))
    server_btn.clicked.connect(lambda: choose("server"))

    dialog.exec()
    return result["mode"]
