"""
frozen_layer_modules — CKA distillation for Unsloth.

Always importable. Behaviour is controlled entirely by DistillationConfig;
no import-time side-effects occur when DISTILLATION=False.

Public API:
    DistillationConfig   — config dataclass
    load_config()        — config loader (env var + YAML + overrides)
    SlowDriftTrainer     — UnslothTrainer subclass with CKA distillation
    LayerFreezer         — stride-based layer freeze/unfreeze + hook registration
    linear_cka()         — CKA similarity metric
    cka_penalty()        — 1 - CKA, clamped to [0, 1]
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .config import DistillationConfig, load_config
    from .cka_loss import cka_penalty, linear_cka
    from .frozen_layer_distillation import LayerFreezer, get_transformer_layers
    from .slow_drift_frozen_layers import SlowDriftTrainer
except Exception as _exc:  # pragma: no cover
    logger.warning(
        "frozen_layer_modules: failed to import sub-modules (%s). "
        "SlowDriftTrainer will not be available.",
        _exc,
    )
    # Stubs so callers can guard with `if SlowDriftTrainer is not None`
    DistillationConfig = None   # type: ignore[assignment,misc]
    load_config = None          # type: ignore[assignment]
    cka_penalty = None          # type: ignore[assignment]
    linear_cka = None           # type: ignore[assignment]
    LayerFreezer = None         # type: ignore[assignment,misc]
    get_transformer_layers = None  # type: ignore[assignment]
    SlowDriftTrainer = None     # type: ignore[assignment,misc]

__all__ = [
    "DistillationConfig",
    "load_config",
    "SlowDriftTrainer",
    "LayerFreezer",
    "get_transformer_layers",
    "linear_cka",
    "cka_penalty",
]
