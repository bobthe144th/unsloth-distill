"""
Configuration loading for distillation mode.

Priority order (highest to lowest):
  1. Environment variable DISTILLATION
  2. .distillation_config.yaml in current directory (or specified path)
  3. Built-in defaults
"""
import os
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

VALID_MODES = {"on", "off"}


@dataclass
class DistillationConfig:
    distillation_mode: str = "off"
    drift_weight: float = 0.1
    restoration_factor: float = 0.99
    divergence_threshold: float = 0.15
    divergence_weight: float = 0.05


def _validate_mode(value) -> str:
    """Validate and normalise DISTILLATION flag value."""
    # PyYAML parses bare 'on'/'off'/'yes'/'no' as booleans
    if isinstance(value, bool):
        return "on" if value else "off"
    v = str(value).lower().strip()
    if v not in VALID_MODES:
        warnings.warn(
            f"Invalid DISTILLATION value '{value}'. "
            f"Valid values: 'on', 'off'. Defaulting to 'off'."
        )
        return "off"
    return v


def load_config(config_path: Optional[Path] = None) -> DistillationConfig:
    """
    Load distillation configuration respecting the priority order.

    Args:
        config_path: Optional explicit path to a YAML config file.
                     Defaults to .distillation_config.yaml in cwd.

    Returns:
        DistillationConfig populated with resolved values.
    """
    # Start from dataclass defaults
    defaults: dict = {f.name: f.default for f in fields(DistillationConfig)}

    # Step 1: overlay values from YAML file (if present)
    yaml_path = Path(config_path) if config_path else Path.cwd() / ".distillation_config.yaml"
    if yaml_path.exists():
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
        for key, val in data.items():
            if key in defaults:
                if key == "distillation_mode":
                    defaults[key] = _validate_mode(val)
                else:
                    defaults[key] = val

    # Step 2: environment variable overrides YAML
    env_val = os.getenv("DISTILLATION")
    if env_val is not None:
        defaults["distillation_mode"] = _validate_mode(env_val)

    return DistillationConfig(**defaults)
