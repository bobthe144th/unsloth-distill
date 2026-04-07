"""
Configuration for CKA distillation.

Flags (sourced from environment variables, .distillation_config.yaml, or direct dict):

  DISTILLATION          bool   False  Master switch. All modifications are no-ops unless True.
  PHASE_UNFREEZE        bool   False  Phase-based unfreezing schedule.
  CKA_LAMBDA            float  0.1    Weight of CKA penalty in total loss.
  PHASE_UNFREEZE_START  float  0.3    Fraction of steps after which unfreezing begins.
  PHASE_UNFREEZE_END    float  0.7    Fraction of steps at which all layers are released.
  FROZEN_LAYER_STRIDE   int    2      Freeze every Nth layer (0, N, 2N, …).

Priority (highest to lowest):
  1. Explicit dict passed to load_config(overrides=...)
  2. Environment variables (UPPER_CASE names)
  3. .distillation_config.yaml in cwd
  4. Built-in defaults
"""
import os
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    distillation: bool = False
    phase_unfreeze: bool = False
    cka_lambda: float = 0.1
    phase_unfreeze_start: float = 0.3
    phase_unfreeze_end: float = 0.7
    frozen_layer_stride: int = 2


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _parse_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    warnings.warn(
        f"Invalid boolean value {value!r} for '{name}'. Defaulting to False."
    )
    return False


def _parse_float(value, name: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid float value {value!r} for '{name}'. Defaulting to {default}."
        )
        return default


def _parse_int(value, name: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid int value {value!r} for '{name}'. Defaulting to {default}."
        )
        return default


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# Mapping: dataclass field name → env var name
_ENV_MAP: Dict[str, str] = {
    "distillation": "DISTILLATION",
    "phase_unfreeze": "PHASE_UNFREEZE",
    "cka_lambda": "CKA_LAMBDA",
    "phase_unfreeze_start": "PHASE_UNFREEZE_START",
    "phase_unfreeze_end": "PHASE_UNFREEZE_END",
    "frozen_layer_stride": "FROZEN_LAYER_STRIDE",
}


def _apply_value(cfg: Dict[str, Any], key: str, raw) -> None:
    """Parse raw string/value and update cfg dict in-place."""
    defaults = DistillationConfig()
    default = getattr(defaults, key)
    if isinstance(default, bool):
        cfg[key] = _parse_bool(raw, key)
    elif isinstance(default, float):
        cfg[key] = _parse_float(raw, key, default)
    elif isinstance(default, int):
        cfg[key] = _parse_int(raw, key, default)
    else:
        cfg[key] = raw


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DistillationConfig:
    """
    Build a DistillationConfig respecting the priority order.

    Args:
        config_path: Path to YAML config file.
                     Defaults to .distillation_config.yaml in cwd.
        overrides:   Dict of runtime overrides (highest priority).

    Returns:
        Populated DistillationConfig.
    """
    cfg: Dict[str, Any] = {f.name: f.default for f in fields(DistillationConfig)}

    # --- Layer 3: YAML file ---
    yaml_path = Path(config_path) if config_path else Path.cwd() / ".distillation_config.yaml"
    if yaml_path.exists():
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
        for key in cfg:
            # Accept both snake_case and UPPER_CASE keys in the YAML
            raw = data.get(key) or data.get(_ENV_MAP.get(key, key.upper()))
            if raw is not None:
                _apply_value(cfg, key, raw)

    # --- Layer 2: environment variables ---
    for field_name, env_name in _ENV_MAP.items():
        raw = os.getenv(env_name)
        if raw is not None:
            _apply_value(cfg, field_name, raw)

    # --- Layer 1: explicit overrides ---
    if overrides:
        for key, raw in overrides.items():
            if key in cfg:
                _apply_value(cfg, key, raw)

    return DistillationConfig(**cfg)
