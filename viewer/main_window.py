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
    QVBoxLayout,
    QWidget,
    QStatusBar,
)
from PyQt6.QtCore import Qt, QTimer

from contracts import AppConfig, ImageEntry, RenderParams
from data_index import DataIndex
from image_loader import ImageLoader
from overlay_renderer import OverlayRenderer
from browser_widget import BrowserWidget
from viewer_widget import ViewerWidget, OverlayControlWidget


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
        # Widgets — 4-panel grid: Overlay, RGB, Seg, Depth
        # ------------------------------------------------------------------ #
        self._browser = BrowserWidget()

        self._panel_overlay = ViewerWidget()
        self._panel_rgb     = ViewerWidget()
        self._panel_seg     = ViewerWidget()
        self._panel_depth   = ViewerWidget()

        self._controls = OverlayControlWidget(
            default_alpha_rgb=cfg.overlay.default_alpha_rgb,
            default_alpha_depth=cfg.overlay.default_alpha_depth,
            default_alpha_seg=cfg.overlay.default_alpha_seg,
        )

        # ------------------------------------------------------------------ #
        # Layout: browser (left) + right container (grid + control bar)
        #
        #   QSplitter(Horizontal)          ← setCentralWidget
        #     BrowserWidget
        #     QWidget (right_container)
        #       QSplitter(Vertical)        ← right_split
        #         QSplitter(Horizontal)    ← top_row   [Overlay | RGB  ]
        #         QSplitter(Horizontal)    ← bottom_row[Seg     | Depth]
        #       OverlayControlWidget       ← controls (checkbox + sliders)
        # ------------------------------------------------------------------ #
        top_row = QSplitter(Qt.Orientation.Horizontal)
        top_row.addWidget(self._panel_overlay)
        top_row.addWidget(self._panel_rgb)

        bottom_row = QSplitter(Qt.Orientation.Horizontal)
        bottom_row.addWidget(self._panel_seg)
        bottom_row.addWidget(self._panel_depth)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(top_row)
        right_split.addWidget(bottom_row)

        # Wrap the 4-panel grid + control bar in a plain widget so controls
        # sit flush below the grid without consuming splitter space.
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(right_split, stretch=1)
        right_layout.addWidget(self._controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._browser)
        splitter.addWidget(right_container)

        # Give the browser its configured initial width; let the grid take
        # the rest.  Use a large number for the right side so it naturally expands.
        splitter.setSizes([cfg.ui.browser_width, 9999])

        # The splitter is the central widget; it fills the whole window.
        self.setCentralWidget(splitter)

        # ------------------------------------------------------------------ #
        # Status bar (created on demand by QMainWindow, reference it now)
        # ------------------------------------------------------------------ #
        self.statusBar()  # creates and shows it

        # ------------------------------------------------------------------ #
        # Signal wiring
        # ------------------------------------------------------------------ #
        self._browser.entry_selected.connect(self._on_entry_selected)
        self._browser.search_changed.connect(self._on_search_changed)
        self._browser.sort_changed.connect(self._on_sort_changed)

        self._image_loader.loaded.connect(self._on_loaded)
        self._image_loader.error.connect(self._on_error)

        self._controls.params_changed.connect(self._on_params_changed)

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
        """Run DataIndex.scan() after the event loop starts to avoid blocking window display."""
        self._all_entries = self._data_index.scan(Path(self._cfg.data.root), self._cfg.data)
        self._browser.set_entries(self._all_entries)

    def _on_entry_selected(self, entry: ImageEntry) -> None:
        """Kick off async image loading when the user clicks a row."""
        self._current_entry = entry
        self.statusBar().showMessage(f"Loading {entry.frame_id} …")
        self._image_loader.load_async(entry)

    def _on_loaded(
        self,
        rgb: np.ndarray,
        depth: object,  # np.ndarray | None
        seg: object,    # list[dict] | None
    ) -> None:
        """Cache the freshly loaded data and compose the initial overlay.

        NOTE: Full race condition fix requires image_loader to include a
        generation number (or entry reference) in the loaded signal so we can
        confirm the result still matches _current_entry.  Until that layer is
        updated, we guard only against the case where _current_entry was cleared.
        """
        if self._current_entry is None:
            # No entry selected (cleared between load start and finish); discard.
            return
        self._last_rgb = rgb
        self._last_depth = depth      # type: ignore[assignment]
        self._last_seg = seg          # type: ignore[assignment]
        self.statusBar().clearMessage()
        self._recompose()

    def _recompose(self, params: RenderParams | None = None) -> None:
        """Re-render all four panels using cached data and current slider params."""
        if self._last_rgb is None:
            return
        if params is None:
            params = self._controls.current_params()

        self._panel_overlay.display(
            self._renderer.compose(
                self._last_rgb,
                self._last_depth,
                self._last_seg,
                params,
                self._cfg,
            )
        )
        self._panel_rgb.display(
            self._renderer.compose_rgb(self._last_rgb)
        )
        self._panel_seg.display(
            self._renderer.compose_seg(self._last_rgb, self._last_seg, self._cfg)
        )
        self._panel_depth.display(
            self._renderer.compose_depth(self._last_depth, self._cfg)
        )

    def _on_params_changed(self, params: RenderParams) -> None:
        """Re-compose all panels when a slider moves (no disk I/O — uses cached arrays)."""
        self._recompose(params)

    def _on_error(self, msg: str) -> None:
        """Show loader errors in the status bar for 5 seconds."""
        self.statusBar().showMessage(f"Error: {msg}", 5000)

    def _on_search_changed(self, query: str) -> None:
        """Update search state and re-filter the entry list."""
        self._current_query = query
        self._apply_filter()

    def _on_sort_changed(self, sort_by: str) -> None:
        """Update sort state and re-filter the entry list."""
        self._current_sort = sort_by
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Run DataIndex.filter() with the current query/sort and push to browser."""
        filtered = self._data_index.filter(
            self._all_entries,
            query=self._current_query,
            sort_by=self._current_sort,
        )
        self._browser.set_entries(filtered)
