"""config_loader.py — Load and validate AppConfig from a YAML file.

No PyQt6 dependency.  Requires: pydantic, pyyaml.
"""

from pathlib import Path

import yaml
from pydantic import ValidationError

from contracts import AppConfig, ConfigValidationError


def load_config(path: Path) -> AppConfig:
    """Load YAML from *path*, parse into AppConfig, and return it.

    Raises
    ------
    ConfigValidationError
        If the file cannot be read, parsed as YAML, or validated by Pydantic.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigValidationError(
            f"Cannot read config file '{path}': {exc}"
        ) from exc

    try:
        raw_dict = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"YAML parse error in '{path}': {exc}"
        ) from exc

    if raw_dict is None:
        raise ConfigValidationError(
            f"Config file '{path}' is empty. "
            "Please provide a valid YAML mapping (e.g. starting with 'data:')."
        )
    if not isinstance(raw_dict, dict):
        raise ConfigValidationError(
            f"Config file '{path}' must contain a YAML mapping at the top level, "
            f"but got {type(raw_dict).__name__}."
        )

    try:
        return AppConfig.model_validate(raw_dict)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Config validation failed for '{path}':\n{exc}"
        ) from exc
