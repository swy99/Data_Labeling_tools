"""main_window.py — Top-level application window.

Wires together BrowserWidget, ViewerWidget, ImageLoader, OverlayRenderer,
and DataIndex into a working image-viewer application.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
)
from PyQt6.QtCore import Qt, QTimer

from contracts import AppConfig, ImageEntry, RenderParams
from data_index import DataIndex
from image_loader import ImageLoader
from overlay_renderer import OverlayRenderer
from browser_widget import BrowserWidget
from viewer_widget import ViewerWidget, LayerControlPanel


class MainWindow(QMainWindow):
    """Main application window.

    Parameters
    ----------
    cfg:
        Fully validated AppConfig loaded from config.yaml.
    """

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Configuration & stateless components
        # ------------------------------------------------------------------ #
        self._cfg = cfg
        self._data_index = DataIndex()
        self._image_loader = ImageLoader(parent=self)
        self._renderer = OverlayRenderer()

        # ------------------------------------------------------------------ #
        # Cache for re-composing on slider change (no disk I/O)
        # ------------------------------------------------------------------ #
        self._last_rgb: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_seg: list[dict] | None = None
        self._current_entry: ImageEntry | None = None  # race condition guard

        # ------------------------------------------------------------------ #
        # Current filter state (updated by signals from BrowserWidget)
        # ------------------------------------------------------------------ #
        self._current_query: str = ""
        self._current_sort: str = cfg.ui.default_sort

        # ------------------------------------------------------------------ #
        # Widgets
        #   Panel 1 (top-left):  composite overlay
        #   Panel 2 (top-right): layer control — RGB/Depth/Seg previews + sliders
        #   Panel 3 (bot-left):  segmentation only
        #   Panel 4 (bot-right): depth only
        # ------------------------------------------------------------------ #
        self._browser = BrowserWidget()

        self._panel_overlay     = ViewerWidget()
        self._panel_layer_ctrl  = LayerControlPanel(
            default_alpha_rgb=cfg.overlay.default_alpha_rgb,
            default_alpha_depth=cfg.overlay.default_alpha_depth,
            default_alpha_seg=cfg.overlay.default_alpha_seg,
        )
        self._panel_seg         = ViewerWidget()
        self._panel_depth       = ViewerWidget()

        # ------------------------------------------------------------------ #
        # Layout: browser (left) + 2×2 panel grid (right)
        #
        #   QSplitter(Horizontal)       ← setCentralWidget
        #     BrowserWidget
        #     QSplitter(Vertical)       ← right_split
        #       QSplitter(Horizontal)   ← top_row   [Overlay | LayerCtrl]
        #       QSplitter(Horizontal)   ← bot_row   [Seg     | Depth    ]
        # ------------------------------------------------------------------ #
        top_row = QSplitter(Qt.Orientation.Horizontal)
        top_row.addWidget(self._panel_overlay)
        top_row.addWidget(self._panel_layer_ctrl)
        top_row.setSizes([500, 500])

        bot_row = QSplitter(Qt.Orientation.Horizontal)
        bot_row.addWidget(self._panel_seg)
        bot_row.addWidget(self._panel_depth)
        bot_row.setSizes([500, 500])

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(top_row)
        right_split.addWidget(bot_row)
        right_split.setSizes([500, 500])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._browser)
        splitter.addWidget(right_split)
        splitter.setSizes([cfg.ui.browser_width, 9999])

        self.setCentralWidget(splitter)

        # ------------------------------------------------------------------ #
        # Status bar
        # ------------------------------------------------------------------ #
        self.statusBar()

        # ------------------------------------------------------------------ #
        # Signal wiring
        # ------------------------------------------------------------------ #
        self._browser.entry_selected.connect(self._on_entry_selected)
        self._browser.search_changed.connect(self._on_search_changed)
        self._browser.sort_changed.connect(self._on_sort_changed)

        self._image_loader.loaded.connect(self._on_loaded)
        self._image_loader.error.connect(self._on_error)

        self._panel_layer_ctrl.params_changed.connect(self._on_params_changed)

        # ------------------------------------------------------------------ #
        # Initial data scan (deferred to avoid blocking window display)
        # ------------------------------------------------------------------ #
        self._all_entries: list[ImageEntry] = []
        QTimer.singleShot(0, self._initial_scan)

        # ------------------------------------------------------------------ #
        # Window properties
        # ------------------------------------------------------------------ #
        self.setWindowTitle(cfg.ui.window_title)
        self.resize(1400, 900)

    # ---------------------------------------------------------------------- #
    # Slots
    # ---------------------------------------------------------------------- #

    def _initial_scan(self) -> None:
        self._all_entries = self._data_index.scan(Path(self._cfg.data.root), self._cfg.data)
        self._browser.set_entries(self._all_entries)

    def _on_entry_selected(self, entry: ImageEntry) -> None:
        self._current_entry = entry
        self.statusBar().showMessage(f"Loading {entry.frame_id} …")
        self._image_loader.load_async(entry)

    def _on_loaded(
        self,
        rgb: np.ndarray,
        depth: object,
        seg: object,
    ) -> None:
        if self._current_entry is None:
            return
        self._last_rgb = rgb
        self._last_depth = depth      # type: ignore[assignment]
        self._last_seg = seg          # type: ignore[assignment]
        self.statusBar().clearMessage()
        self._recompose()

    def _recompose(self, params: RenderParams | None = None) -> None:
        """Re-render all panels using cached data and current slider params."""
        if self._last_rgb is None:
            return
        if params is None:
            params = self._panel_layer_ctrl.current_params()

        self._panel_overlay.display(
            self._renderer.compose(
                self._last_rgb,
                self._last_depth,
                self._last_seg,
                params,
                self._cfg,
            )
        )
        self._panel_seg.display(
            self._renderer.compose_seg(self._last_rgb, self._last_seg, self._cfg)
        )
        self._panel_depth.display(
            self._renderer.compose_depth(self._last_depth, self._cfg)
        )

    def _on_params_changed(self, params: RenderParams) -> None:
        self._recompose(params)

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error: {msg}", 5000)

    def _on_search_changed(self, query: str) -> None:
        self._current_query = query
        self._apply_filter()

    def _on_sort_changed(self, sort_by: str) -> None:
        self._current_sort = sort_by
        self._apply_filter()

    def _apply_filter(self) -> None:
        filtered = self._data_index.filter(
            self._all_entries,
            query=self._current_query,
            sort_by=self._current_sort,
        )
        self._browser.set_entries(filtered)
