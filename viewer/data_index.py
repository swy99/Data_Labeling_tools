"""data_index.py — Filesystem scan and filtering for ImageEntry objects.

No PyQt6 dependency.  Requires only the stdlib + contracts.py.

Dataset layout expected by scan():
    {root}/{scenario}/Camera_{N}/{frame_id}.jpg          ← image
    {root}/{scenario}/Camera_{N}/{frame_id}.json         ← annotation (optional)
    {root}/{scenario}/Camera_{N}/{depth_dir}/{frame_id}{depth_suffix}
                                                         ← depth map (optional)

scan() derives scenario and camera from the two path components directly above
each .jpg file (i.e. parent.name and parent.parent.name respectively — see
comments in the method).  The folder names are not validated against a fixed
pattern so that any scenario / camera naming convention is accepted.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from contracts import DataConfig, ImageEntry


class DataIndex:
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, root: Path, cfg: DataConfig) -> list[ImageEntry]:
        """Walk *root* recursively and build an ImageEntry for every .jpg found.

        The camera directory is the immediate parent of the .jpg file.
        The scenario directory is the parent of the camera directory.

        Parameters
        ----------
        root:
            Dataset root directory.
        cfg:
            DataConfig instance providing depth_dir_name and depth_suffix.

        Returns
        -------
        list[ImageEntry]
            One entry per .jpg discovered, in filesystem traversal order.
        """
        entries: list[ImageEntry] = []

        for jpg_path in sorted(root.rglob("*.jpg")):
            camera_dir = jpg_path.parent
            scenario_dir = camera_dir.parent

            # Skip .jpg files that are not nested at least two levels under root.
            # scenario_dir must be a proper subdirectory of root (not root itself),
            # and camera_dir must likewise differ from root.
            if scenario_dir == root or camera_dir == root:
                continue

            stem = jpg_path.stem
            scenario = scenario_dir.name
            camera = camera_dir.name

            # --- annotation ---
            json_path = camera_dir / f"{stem}.json"
            annotation_path: Path | None
            label_count: int
            labels: list[str]

            if json_path.is_file():
                annotation_path = json_path
                labels = _read_labels(json_path)
                label_count = len(labels)
            else:
                annotation_path = None
                label_count = 0
                labels = []

            # --- depth ---
            depth_candidate = (
                camera_dir / cfg.depth_dir_name / f"{stem}{cfg.depth_suffix}"
            )
            depth_path: Path | None = (
                depth_candidate if depth_candidate.is_file() else None
            )

            entries.append(
                ImageEntry(
                    path=jpg_path,
                    scenario=scenario,
                    camera=camera,
                    frame_id=stem,
                    annotation_path=annotation_path,
                    depth_path=depth_path,
                    label_count=label_count,
                    labels=labels,
                )
            )

        return entries

    def filter(
        self,
        entries: list[ImageEntry],
        query: str = "",
        sort_by: str = "name",
    ) -> list[ImageEntry]:
        """Return a filtered and sorted copy of *entries*.

        Parameters
        ----------
        entries:
            Source list — never mutated.
        query:
            Case-insensitive substring matched against each entry's frame_id.
            Empty string matches everything.
        sort_by:
            Sorting strategy:

            ``"name"``
                Alphabetical ascending by frame_id.
            ``"annotation"``
                Annotated entries first (annotation_path is not None), then
                unannotated; ties broken alphabetically by frame_id.
            ``"label_count"``
                Descending by label_count; ties broken alphabetically by
                frame_id.

        Returns
        -------
        list[ImageEntry]
            New list — input is not mutated.
        """
        q = query.lower()

        # Filter (copy only, do not mutate input)
        filtered = [e for e in entries if q in e.frame_id.lower()] if q else list(entries)

        # Sort
        _VALID_SORT_KEYS = {"name", "annotation", "label_count"}
        if sort_by not in _VALID_SORT_KEYS:
            warnings.warn(
                f"Unknown sort_by value '{sort_by}'. "
                f"Valid values are: {sorted(_VALID_SORT_KEYS)}. "
                "Falling back to 'name'.",
                stacklevel=2,
            )
            sort_by = "name"

        if sort_by == "annotation":
            filtered.sort(key=lambda e: (0 if e.annotation_path is not None else 1, e.frame_id))
        elif sort_by == "label_count":
            filtered.sort(key=lambda e: (-e.label_count, e.frame_id))
        else:
            # Default: "name"
            filtered.sort(key=lambda e: e.frame_id)

        return filtered


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_labels(json_path: Path) -> list[str]:
    """Extract label names from an X-AnyLabeling annotation JSON file.

    Returns an empty list if the file cannot be parsed or has no shapes.
    The label list may contain duplicates (one entry per shape instance).
    """
    try:
        with json_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        print(f"[data_index] WARNING: Cannot open '{json_path}': {exc}", file=sys.stderr)
        return []
    except json.JSONDecodeError as exc:
        print(f"[data_index] WARNING: JSON parse error in '{json_path}': {exc}", file=sys.stderr)
        return []

    try:
        shapes = data.get("shapes", [])
        return [s["label"] for s in shapes if "label" in s]
    except (AttributeError, TypeError) as exc:
        print(
            f"[data_index] WARNING: Unexpected structure in '{json_path}' "
            f"(parsed OK but 'shapes' is not iterable): {exc}",
            file=sys.stderr,
        )
        return []
