"""
frozen_layer_modules – distillation mode import guard.

When DISTILLATION="on" (env var or runtime call to set_mode()):
  - SlowDriftTrainer and AlternatingLayerFreezer are imported and exposed.

When DISTILLATION="off" (default):
  - No frozen-layer modules are loaded; both names are None.

Graceful fallback: if the sub-modules fail to import for any reason, a warning
is logged, both names remain None, and training continues in standard mode.
"""
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime state – mutated by set_mode()
# ---------------------------------------------------------------------------
DISTILLATION_MODE: bool = os.getenv("DISTILLATION", "off").lower() == "on"

SlowDriftTrainer = None
AlternatingLayerFreezer = None


def _try_import() -> None:
    """Attempt to load frozen-layer classes; fall back silently on failure."""
    global SlowDriftTrainer, AlternatingLayerFreezer
    try:
        from .slow_drift_frozen_layers import SlowDriftTrainer as _SDT
        from .frozen_layer_distillation import AlternatingLayerFreezer as _ALF
        SlowDriftTrainer = _SDT
        AlternatingLayerFreezer = _ALF
    except Exception as exc:
        logger.warning(
            "Failed to import frozen_layer_modules. "
            f"Training in standard mode. Error: {exc}"
        )
        SlowDriftTrainer = None
        AlternatingLayerFreezer = None


if DISTILLATION_MODE:
    _try_import()


def set_mode(value: str) -> bool:
    """
    Update DISTILLATION_MODE at runtime (called by Jupyter magic).

    Args:
        value: "on" or "off" (case-insensitive).

    Returns:
        True if distillation mode is now active, False otherwise.
    """
    global DISTILLATION_MODE
    from .config import _validate_mode
    validated = _validate_mode(value)
    DISTILLATION_MODE = validated == "on"
    os.environ["DISTILLATION"] = validated
    if DISTILLATION_MODE:
        try:
            _try_import()
        except Exception as exc:
            logger.warning(
                "Failed to import frozen_layer_modules. "
                f"Training in standard mode. Error: {exc}"
            )
    return DISTILLATION_MODE
