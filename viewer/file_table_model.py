from PyQt6.QtCore import QAbstractTableModel, Qt, QModelIndex
from PyQt6.QtGui import QColor

from contracts import ImageEntry


class FileTableModel(QAbstractTableModel):
    PAGE_SIZE = 200
    COLUMNS = ["Frame ID", "Scenario", "Camera", "Annotated", "Labels", "Depth"]

    # Column index constants — keeps data() / headerData() readable
    _COL_FRAME_ID  = 0
    _COL_SCENARIO  = 1
    _COL_CAMERA    = 2
    _COL_ANNOTATED = 3
    _COL_LABELS    = 4
    _COL_DEPTH     = 5

    _COLOR_YES = QColor(0, 180, 0)    # green  — annotated / depth present
    _COLOR_NO  = QColor(180, 0, 0)    # red    — not annotated

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_entries: list[ImageEntry] = []
        self._loaded: int = 0           # how many rows are currently visible

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_entries(self, entries: list[ImageEntry]) -> None:
        """Replace the full entry list and reset to first PAGE_SIZE rows."""
        self.beginResetModel()
        self._all_entries = entries
        self._loaded = min(self.PAGE_SIZE, len(entries))
        self.endResetModel()

    def entry_at(self, row: int) -> ImageEntry:
        """Return the ImageEntry at *row* (used by BrowserWidget on click)."""
        if not (0 <= row < self._loaded):
            raise IndexError(row)
        return self._all_entries[row]

    # ------------------------------------------------------------------
    # QAbstractTableModel interface
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return self._loaded

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section]
        # Vertical header: suppress row numbers (BrowserWidget hides it too,
        # but returning None here makes it clean even without the hide call)
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row, col = index.row(), index.column()
        if row >= self._loaded or col >= len(self.COLUMNS):
            return None

        entry: ImageEntry = self._all_entries[row]

        # ------ DisplayRole ------
        if role == Qt.ItemDataRole.DisplayRole:
            if col == self._COL_FRAME_ID:
                return entry.frame_id
            if col == self._COL_SCENARIO:
                return entry.scenario
            if col == self._COL_CAMERA:
                return entry.camera
            if col == self._COL_ANNOTATED:
                return "✓" if entry.annotation_path is not None else "✗"
            if col == self._COL_LABELS:
                return str(entry.label_count)
            if col == self._COL_DEPTH:
                return "✓" if entry.depth_path is not None else "–"

        # ------ ForegroundRole (colours for Annotated column only) ------
        if role == Qt.ItemDataRole.ForegroundRole:
            if col == self._COL_ANNOTATED:
                return self._COLOR_YES if entry.annotation_path is not None else self._COLOR_NO

        # ------ TextAlignmentRole — centre the tick / cross columns ------
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (self._COL_ANNOTATED, self._COL_LABELS, self._COL_DEPTH):
                return Qt.AlignmentFlag.AlignCenter

        return None

    # ------------------------------------------------------------------
    # Incremental loading (fetchMore / canFetchMore)
    # ------------------------------------------------------------------

    def canFetchMore(self, parent: QModelIndex = QModelIndex()) -> bool:  # noqa: B008
        if parent.isValid():
            return False
        return self._loaded < len(self._all_entries)

    def fetchMore(self, parent: QModelIndex = QModelIndex()) -> None:  # noqa: B008
        if parent.isValid():
            return
        remaining = len(self._all_entries) - self._loaded
        to_fetch = min(self.PAGE_SIZE, remaining)
        if to_fetch <= 0:
            return
        self.beginInsertRows(QModelIndex(), self._loaded, self._loaded + to_fetch - 1)
        self._loaded += to_fetch
        self.endInsertRows()
