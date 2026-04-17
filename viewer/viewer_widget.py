"""viewer_widget.py — Image display panel and overlay control bar."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap

from contracts import RenderParams


class ViewerWidget(QWidget):
    """Pure image display panel — no controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap: QPixmap | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setScaledContents(False)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._image_label.setStyleSheet("background-color: #1a1a1a;")
        self._image_label.setMinimumSize(1, 1)
        root.addWidget(self._image_label, stretch=1)

    def display(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self._image_label.clear()
            self._original_pixmap = None
            return
        self._original_pixmap = pixmap
        self._scale_to_label()

    def _scale_to_label(self) -> None:
        original = self._original_pixmap
        if original is None or original.isNull():
            return
        label_size = self._image_label.size()
        if label_size.isEmpty():
            return
        scaled = original.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._scale_to_label()


class OverlayControlWidget(QWidget):
    """Horizontal control bar: checkbox + alpha slider for RGB, Depth, Seg.

    Signals
    -------
    params_changed(RenderParams)
        Emitted whenever any control changes.
    """

    params_changed = pyqtSignal(RenderParams)

    def __init__(
        self,
        default_alpha_rgb: float = 1.0,
        default_alpha_depth: float = 0.5,
        default_alpha_seg: float = 0.6,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._build_ui(default_alpha_rgb, default_alpha_depth, default_alpha_seg)
        self._connect_signals()

    def _build_ui(
        self,
        alpha_rgb: float,
        alpha_depth: float,
        alpha_seg: float,
    ) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(24)
        root.addStretch()

        self._check_rgb, self._slider_rgb, _ = self._add_layer(root, "RGB", alpha_rgb)
        self._check_depth, self._slider_depth, _ = self._add_layer(root, "Depth", alpha_depth)
        self._check_seg, self._slider_seg, _ = self._add_layer(root, "Seg", alpha_seg)

        root.addStretch()

    def _add_layer(
        self, parent_layout: QHBoxLayout, label: str, default_alpha: float
    ) -> tuple[QCheckBox, QSlider, QLabel]:
        group = QWidget()
        row = QHBoxLayout(group)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        check = QCheckBox(label)
        check.setChecked(True)
        check.setFixedWidth(64)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(round(default_alpha * 100)))
        slider.setFixedWidth(110)

        readout = QLabel(f"{slider.value() / 100:.2f}")
        readout.setFixedWidth(36)
        readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        slider.valueChanged.connect(
            lambda val, lbl=readout: lbl.setText(f"{val / 100:.2f}")
        )

        row.addWidget(check)
        row.addWidget(slider)
        row.addWidget(readout)
        parent_layout.addWidget(group)

        return check, slider, readout

    def _connect_signals(self) -> None:
        for slider in (self._slider_rgb, self._slider_depth, self._slider_seg):
            slider.valueChanged.connect(self._on_changed)
        for check in (self._check_rgb, self._check_depth, self._check_seg):
            check.toggled.connect(self._on_changed)

    def _on_changed(self) -> None:
        self.params_changed.emit(self.current_params())

    def current_params(self) -> RenderParams:
        return RenderParams(
            alpha_rgb=self._slider_rgb.value() / 100,
            alpha_depth=self._slider_depth.value() / 100,
            alpha_seg=self._slider_seg.value() / 100,
            show_rgb=self._check_rgb.isChecked(),
            show_depth=self._check_depth.isChecked(),
            show_seg=self._check_seg.isChecked(),
        )
