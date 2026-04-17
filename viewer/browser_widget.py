from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QTableView,
    QAbstractItemView,
    QHeaderView,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, QModelIndex, QTimer

from contracts import ImageEntry
from file_table_model import FileTableModel


class BrowserWidget(QWidget):
    """File browser panel.

    Responsibilities
    ----------------
    - Display a filterable, sortable table of ImageEntry rows.
    - Emit ``entry_selected`` when the user clicks a row.
    - Emit ``search_changed`` / ``sort_changed`` so the MainWindow can call
      DataIndex.filter() and feed the result back via set_entries().

    This widget does *not* call DataIndex.scan() or filter() itself.
    """

    entry_selected = pyqtSignal(ImageEntry)
    search_changed = pyqtSignal(str)
    sort_changed   = pyqtSignal(str)   # emits the sort_by key string

    # Maps human-readable combo labels to DataIndex sort_by strings
    _SORT_OPTIONS: list[tuple[str, str]] = [
        ("Name",         "name"),
        ("Annotation",   "annotation"),
        ("Label Count",  "label_count"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(150)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- Top row: search bar + sort combo ----
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search…")
        self._search.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_row.addWidget(self._search)

        self._sort_combo = QComboBox()
        for label, _ in self._SORT_OPTIONS:
            self._sort_combo.addItem(label)
        self._sort_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        top_row.addWidget(self._sort_combo)

        root.addLayout(top_row)

        # ---- Table ----
        self._model = FileTableModel(self)

        self._table = QTableView()
        self._table.setModel(self._model)

        # Selection: whole row at a time, single selection
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Column sizing: stretch the last column (Depth) to fill remaining space
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        # Resize other columns to fit their content initially
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # Override the last column back to stretch (ResizeToContents wins per-column)
        header.setSectionResizeMode(
            len(FileTableModel.COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )

        # Hide the vertical header (row numbers)
        self._table.verticalHeader().setVisible(False)

        # Read-only: no editing
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # Alternating row colours for readability at 40k rows
        self._table.setAlternatingRowColors(True)

        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._table)

    def _connect_signals(self) -> None:
        self._search_timer.timeout.connect(lambda: self.search_changed.emit(self._search.text()))
        self._search.textChanged.connect(lambda: self._search_timer.start())
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self._table.clicked.connect(self._on_row_clicked)
        self._table.selectionModel().currentChanged.connect(self._on_current_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_entries(self, entries: list[ImageEntry]) -> None:
        """Push a new filtered/sorted entry list into the table model."""
        self._model.set_entries(entries)

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_sort_changed(self, index: int) -> None:
        _, key = self._SORT_OPTIONS[index]
        self.sort_changed.emit(key)

    def _on_row_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        entry = self._model.entry_at(index.row())
        self.entry_selected.emit(entry)

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        """Emit entry_selected when keyboard navigation changes the current row."""
        if not current.isValid():
            return
        entry = self._model.entry_at(current.row())
        self.entry_selected.emit(entry)
