"""image_loader.py — Async image / depth / segmentation loader for the viewer.

Each call to load_async() spins up a fresh QThread so the main thread is
never blocked.  The thread (and its worker) are cleaned up automatically via
the thread.finished → thread.deleteLater connection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from contracts import DepthShapeMismatch, ImageEntry


# ---------------------------------------------------------------------------
# Internal worker — lives inside the QThread
# ---------------------------------------------------------------------------

class _LoadWorker(QObject):
    """Does the actual I/O on the worker thread."""

    # Mirror the outer signals so results bubble up cleanly.
    loaded = pyqtSignal(np.ndarray, object, object)  # rgb, depth|None, seg|None
    error = pyqtSignal(str)

    def __init__(self, entry: ImageEntry, generation: int) -> None:
        super().__init__()
        self._entry = entry
        self._generation = generation
        # Mutable reference to the loader's current generation counter.
        # The worker reads this at emit time to detect stale results.
        self._loader_generation: list[int] = []  # set by ImageLoader before start

    # Slot called by QThread.started
    def run(self) -> None:
        entry = self._entry
        try:
            # ----------------------------------------------------------------
            # 1. RGB — always present
            # ----------------------------------------------------------------
            rgb: np.ndarray = np.array(
                Image.open(entry.path).convert("RGB"), dtype=np.uint8
            )

            # ----------------------------------------------------------------
            # 2. Depth — optional
            # ----------------------------------------------------------------
            depth: np.ndarray | None = None
            if entry.depth_path is not None:
                depth = np.load(entry.depth_path)
                # Bug #5: validate ndim before shape[:2] comparison
                if depth.ndim < 2:
                    raise DepthShapeMismatch(
                        f"Depth array must be at least 2-D, got shape {depth.shape}"
                    )
                if depth.shape[:2] != rgb.shape[:2]:
                    raise DepthShapeMismatch(
                        f"Depth shape {depth.shape[:2]} does not match "
                        f"image shape {rgb.shape[:2]} for '{entry.path.name}'"
                    )

            # ----------------------------------------------------------------
            # 3. Segmentation — optional
            # ----------------------------------------------------------------
            seg: list[dict] | None = None
            if entry.annotation_path is not None:
                text = Path(entry.annotation_path).read_text(encoding="utf-8")
                data = json.loads(text)
                # Support both bare list and {"shapes": [...]} wrappers.
                if isinstance(data, list):
                    seg = data
                elif isinstance(data, dict) and "shapes" in data:
                    seg = data["shapes"]
                else:
                    seg = []

            # Bug #1: discard result if a newer load_async() was called.
            if self._loader_generation and self._loader_generation[0] != self._generation:
                return
            self.loaded.emit(rgb, depth, seg)

        except DepthShapeMismatch as exc:
            if self._loader_generation and self._loader_generation[0] != self._generation:
                return
            self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            if self._loader_generation and self._loader_generation[0] != self._generation:
                return
            self.error.emit(f"Load error for '{entry.path}': {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ImageLoader(QObject):
    """Emits loaded/error signals from a background QThread.

    Keep an instance alive (e.g. as a member of a widget) for the signals to
    work.  A fresh QThread is created per load_async() call; the previous
    thread is allowed to finish on its own.
    """

    loaded = pyqtSignal(np.ndarray, object, object)  # rgb, depth|None, seg|None
    error = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Retain a reference so Python's GC does not collect threads/workers
        # while they are still running.
        self._threads: list[QThread] = []
        # Bug #2: also keep worker references to prevent GC after moveToThread.
        self._workers: list[_LoadWorker] = []
        # Bug #1: monotonically increasing generation counter.
        # Shared as a mutable container so workers can read the live value.
        self._generation: int = 0
        self._generation_box: list[int] = [0]  # [current_generation]

    def load_async(self, entry: ImageEntry) -> None:
        """Start loading *entry* on a background thread.

        Returns immediately.  Results arrive via ``loaded`` or ``error``.
        """
        # Bug #1: increment generation so any in-flight worker sees it's stale.
        self._generation += 1
        self._generation_box[0] = self._generation
        current_gen = self._generation

        thread = QThread()
        worker = _LoadWorker(entry, current_gen)
        # Give the worker a reference to the shared generation box so it can
        # check staleness just before emitting.
        worker._loader_generation = self._generation_box

        # Move the worker onto the thread so its slots execute there.
        worker.moveToThread(thread)

        # Wire signals between worker and loader (cross-thread, queued
        # connections are created automatically because they cross threads).
        worker.loaded.connect(self.loaded)
        worker.error.connect(self.error)

        # Start the work when the thread is ready.
        thread.started.connect(worker.run)

        # Clean-up chain: when work is done, quit the event loop, then
        # schedule both worker and thread for deletion.
        worker.loaded.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # Bug #6: capture thread in default arg to avoid late-binding closure.
        thread.finished.connect(lambda t=thread, w=worker: self._remove_thread(t, w))

        # Keep references while the thread is alive.
        self._threads.append(thread)
        self._workers.append(worker)  # Bug #2: prevent GC of worker

        thread.start()

    def _remove_thread(self, thread: QThread, worker: _LoadWorker) -> None:
        """Drop the references once the thread has finished."""
        try:
            self._threads.remove(thread)
        except ValueError:
            pass
        try:
            self._workers.remove(worker)
        except ValueError:
            pass
