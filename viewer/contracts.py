from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DepthShapeMismatch(Exception):
    """Raised when a depth array's spatial dimensions do not match the image."""


class ConfigValidationError(Exception):
    """Raised when AppConfig fails to validate from a raw config dict/file."""


# ---------------------------------------------------------------------------
# Config models (Pydantic v2)
# ---------------------------------------------------------------------------

class DepthConfig(BaseModel):
    colormap: str = "plasma"
    value_range: list[float] | None = None  # None = auto min/max per frame
    unit: str = "meter"

    @field_validator("value_range")
    @classmethod
    def _validate_value_range(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError(
                f"value_range must be a list of exactly 2 floats [min, max], got {len(v)} element(s)."
            )
        if v[0] > v[1]:
            raise ValueError(
                f"value_range[0] (min={v[0]}) must be <= value_range[1] (max={v[1]})."
            )
        return v


class OverlayConfig(BaseModel):
    default_alpha_rgb: float = Field(default=1.0, ge=0.0, le=1.0)
    default_alpha_depth: float = Field(default=0.5, ge=0.0, le=1.0)
    default_alpha_seg: float = Field(default=0.6, ge=0.0, le=1.0)


class SegConfig(BaseModel):
    # Mutable default via default_factory to satisfy Pydantic v2 constraints.
    label_colors: dict[str, list[int]] = Field(default_factory=dict)
    polygon_opacity: float = Field(default=0.4, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    root: str
    depth_dir_name: str = "x-anylabeling-depth"
    depth_suffix: str = "_depth.npy"


class UIConfig(BaseModel):
    window_title: str = "AnyLabel Viewer"
    browser_width: int = 400
    default_sort: str = "name"  # "name" | "annotation" | "label_count"


class AppConfig(BaseModel):
    data: DataConfig
    depth: DepthConfig = Field(default_factory=DepthConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    seg: SegConfig = Field(default_factory=SegConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImageEntry:
    path: Path
    scenario: str
    camera: str
    frame_id: str
    annotation_path: Path | None   # .json — None when not annotated
    depth_path: Path | None        # .npy  — None when not available
    label_count: int               # 0 when no annotation
    labels: list[str]              # []  when no annotation


# ---------------------------------------------------------------------------
# Render parameters
# ---------------------------------------------------------------------------

@dataclass
class RenderParams:
    alpha_rgb: float
    alpha_depth: float
    alpha_seg: float
    show_rgb: bool = True
    show_depth: bool = True
    show_seg: bool = True


# ---------------------------------------------------------------------------
# Render mode
# ---------------------------------------------------------------------------

class RenderMode(Enum):
    OVERLAY    = "overlay"
    RGB_ONLY   = "rgb"
    SEG_ONLY   = "seg"
    DEPTH_ONLY = "depth"
