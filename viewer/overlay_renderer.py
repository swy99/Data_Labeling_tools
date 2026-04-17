"""overlay_renderer.py — Compose RGB + depth + segmentation into a QPixmap.

Composition order (back to front):
    1.  RGB base
    2.  Depth colormap overlay   (alpha-blended)
    3.  Polygon segmentation     (alpha-blended per-polygon fill)
"""

from __future__ import annotations

import hashlib

import numpy as np
import matplotlib
from PIL import Image, ImageDraw
from PyQt6.QtGui import QImage, QPixmap

from contracts import AppConfig, RenderParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_color(label: str) -> tuple[int, int, int]:
    """Return a deterministic RGB color for *label* via MD5 hash."""
    digest = hashlib.md5(label.encode("utf-8")).digest()  # noqa: S324 — not crypto
    return int(digest[0]), int(digest[1]), int(digest[2])


def _get_label_color(
    label: str,
    label_colors: dict[str, list[int]],
) -> tuple[int, int, int]:
    """Look up *label* in config or fall back to a hash-derived color."""
    if label in label_colors:
        c = label_colors[label]
        return int(c[0]), int(c[1]), int(c[2])
    return _auto_color(label)


def _ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert a uint8 HxWx3 array to a QPixmap."""
    h, w, ch = arr.shape
    if ch != 3:  # Bug #4: assert 대신 명시적 예외 (assert는 -O 플래그 시 무효)
        raise ValueError(f"Expected 3-channel uint8 array, got {ch} channels")
    # QImage requires a contiguous buffer.
    arr_c = np.ascontiguousarray(arr)
    qimage = QImage(
        arr_c.data,
        w,
        h,
        w * 3,
        QImage.Format.Format_RGB888,
    )
    # QImage shares the buffer; keep arr_c alive until after copy.
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """Stateless compositor.  Create once and call compose() as needed."""

    def compose(
        self,
        rgb: np.ndarray,           # HxWx3 uint8
        depth: np.ndarray | None,  # HxW float, None = skip
        seg: list[dict] | None,    # shapes list, None = skip
        params: RenderParams,
        cfg: AppConfig,
    ) -> QPixmap:
        """Return a QPixmap composited from the given layers.

        Parameters
        ----------
        rgb:
            Source image as uint8 HxWx3.
        depth:
            Depth map as float HxW.  Skipped when None.
        seg:
            List of annotation shape dicts with keys ``label``, ``points``,
            and ``shape_type``.  Skipped when None.
        params:
            Per-render alpha values.
        cfg:
            Application config (colormap name, value_range, label_colors, …).
        """
        # ------------------------------------------------------------------ #
        # Step 1 — start with RGB as float [0, 1], blended toward black by alpha_rgb
        # ------------------------------------------------------------------ #
        rgb_alpha = params.alpha_rgb if params.show_rgb else 0.0
        canvas: np.ndarray = rgb.astype(np.float32) / 255.0 * rgb_alpha

        # ------------------------------------------------------------------ #
        # Step 2 — depth overlay
        # ------------------------------------------------------------------ #
        if depth is not None and params.show_depth and params.alpha_depth > 0.0:
            depth_colored = self._colorize_depth(depth, cfg)
            # Alpha-blend: keep (1 - alpha_depth) of canvas, add depth layer.
            canvas = (
                canvas * (1.0 - params.alpha_depth)
                + depth_colored * params.alpha_depth
            )
            canvas = np.clip(canvas, 0.0, 1.0)

        # ------------------------------------------------------------------ #
        # Step 3 — segmentation polygon overlay (via PIL)
        # ------------------------------------------------------------------ #
        if seg and params.show_seg and params.alpha_seg > 0.0:
            canvas = self._draw_polygons(canvas, seg, params.alpha_seg, cfg)

        # ------------------------------------------------------------------ #
        # Step 4 — convert to QPixmap
        # ------------------------------------------------------------------ #
        uint8 = (canvas * 255.0).clip(0, 255).astype(np.uint8)
        return _ndarray_to_qpixmap(uint8)

    def compose_rgb(self, rgb: np.ndarray) -> QPixmap:
        """Return a QPixmap from *rgb* without any overlay.

        Parameters
        ----------
        rgb:
            Source image as uint8 HxWx3.
        """
        return _ndarray_to_qpixmap(rgb)

    def compose_seg(
        self,
        rgb: np.ndarray,
        seg: list[dict] | None,
        cfg: AppConfig,
    ) -> QPixmap:
        """Return a QPixmap showing segmentation polygons on a black background.

        Parameters
        ----------
        rgb:
            Source image as uint8 HxWx3.  Used as fallback when *seg* is None.
        seg:
            List of annotation shape dicts.  When None the original *rgb* is
            returned unchanged.
        cfg:
            Application config (label_colors, polygon_opacity, …).
        """
        if seg is None:
            return _ndarray_to_qpixmap(rgb)

        black = np.zeros_like(rgb, dtype=np.float32)
        composited = self._draw_polygons(black, seg, alpha_seg=1.0, cfg=cfg)
        uint8 = (composited * 255.0).clip(0, 255).astype(np.uint8)
        return _ndarray_to_qpixmap(uint8)

    def compose_depth(
        self,
        depth: np.ndarray | None,
        cfg: AppConfig,
    ) -> QPixmap:
        """Return a colorized depth QPixmap, or a null QPixmap when *depth* is None.

        Parameters
        ----------
        depth:
            Depth map as float HxW (or HxWx1).  Returns ``QPixmap()`` when None.
        cfg:
            Application config (colormap, value_range, …).
        """
        if depth is None:
            return QPixmap()

        colored = self._colorize_depth(depth, cfg)
        uint8 = (colored * 255.0).clip(0, 255).astype(np.uint8)
        return _ndarray_to_qpixmap(uint8)

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _colorize_depth(depth: np.ndarray, cfg: AppConfig) -> np.ndarray:
        """Return a float32 HxWx3 [0,1] colorized depth map."""
        # Bug #3: mpl_cm.get_cmap()는 matplotlib 3.9+에서 제거됨
        cmap = matplotlib.colormaps[cfg.depth.colormap]

        # Bug #7: HxWx1 입력을 HxW로 축약해 이후 shape 불일치 방지
        depth = depth.squeeze()

        if cfg.depth.value_range is not None:
            lo, hi = float(cfg.depth.value_range[0]), float(cfg.depth.value_range[1])
        else:
            lo, hi = float(depth.min()), float(depth.max())

        # Avoid divide-by-zero when the depth map is perfectly flat.
        span = hi - lo
        if span == 0.0:
            normalized = np.zeros_like(depth, dtype=np.float32)
        else:
            normalized = ((depth - lo) / span).astype(np.float32)
            normalized = np.clip(normalized, 0.0, 1.0)

        # matplotlib colormaps return HxWx4 (RGBA); drop alpha.
        colored: np.ndarray = cmap(normalized)[:, :, :3].astype(np.float32)
        return colored

    @staticmethod
    def _draw_polygons(
        canvas: np.ndarray,        # float32 HxWx3 [0,1]
        shapes: list[dict],
        alpha_seg: float,
        cfg: AppConfig,
    ) -> np.ndarray:
        """Rasterize polygon fills onto *canvas* and return float32 HxWx3."""
        h, w = canvas.shape[:2]

        # Work in uint8 PIL space for each polygon so we can use ImageDraw.
        # The final result is alpha-composited back onto the canvas.
        base_uint8 = (canvas * 255.0).clip(0, 255).astype(np.uint8)
        base_pil = Image.fromarray(base_uint8, mode="RGB")

        # One shared overlay image accumulates all polygon fills.
        overlay_pil = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_pil)

        # Bug #8: fill_alpha는 polygon_opacity만 사용 (alpha_seg는 블렌딩 레벨에서만 적용)
        fill_alpha = int(round(cfg.seg.polygon_opacity * 255))

        for shape in shapes:
            shape_type = shape.get("shape_type", "polygon")
            if shape_type != "polygon":
                continue

            points_raw = shape.get("points", [])
            if len(points_raw) < 3:
                continue

            label = shape.get("label", "")
            r, g, b = _get_label_color(label, cfg.seg.label_colors)

            # PIL expects a flat list of (x, y) tuples.
            # Bug #9: 길이 2 미만인 점은 IndexError 유발 — 필터링 후 처리
            try:
                poly_coords = [
                    (float(pt[0]), float(pt[1]))
                    for pt in points_raw
                    if len(pt) >= 2
                ]
            except (TypeError, IndexError):
                continue

            if len(poly_coords) < 3:
                continue

            # Fill with configured per-polygon opacity; outline at full alpha.
            draw.polygon(
                poly_coords,
                fill=(r, g, b, fill_alpha),
                outline=(r, g, b, 255),
            )

        # Composite: paste overlay onto base using overlay's own alpha channel.
        base_pil.paste(overlay_pil, mask=overlay_pil.split()[3])

        # Blend the composited PIL result against the original canvas using
        # alpha_seg so the user's global seg alpha slider is respected.
        composited = np.array(base_pil, dtype=np.float32) / 255.0
        result = canvas * (1.0 - alpha_seg) + composited * alpha_seg
        return np.clip(result, 0.0, 1.0)
