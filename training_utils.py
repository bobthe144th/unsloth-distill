"""
training_utils – trainer selection factory.

get_trainer() returns:
  - SlowDriftTrainer  when config['distillation_mode'] == 'on'
  - transformers.Trainer  otherwise (or on import failure)
"""
import logging
from typing import Any, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

# Keys consumed by distillation that are not valid TrainingArguments fields
_DISTILLATION_KEYS = frozenset(
    {"distillation_mode", "drift_weight", "restoration_factor",
     "divergence_threshold", "divergence_weight"}
)


def get_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    base_model: Optional[nn.Module] = None,
):
    """
    Select and construct an appropriate trainer.

    Args:
        model:      The model to train.
        config:     Dict that may contain distillation settings and/or
                    HuggingFace TrainingArguments kwargs.
        base_model: Required when distillation_mode='on'.

    Returns:
        SlowDriftTrainer when distillation_mode='on' (and import succeeds),
        transformers.Trainer otherwise.
    """
    mode = config.get("distillation_mode", "off")
    if isinstance(mode, str):
        mode = mode.lower().strip()

    if mode == "on":
        try:
            from frozen_layer_modules import SlowDriftTrainer
            if SlowDriftTrainer is None:
                raise ImportError("SlowDriftTrainer was not loaded (import guard returned None).")
            return SlowDriftTrainer(model, base_model, config)
        except Exception as exc:
            logger.error(
                "Failed to import frozen_layer_modules. "
                f"Training in standard mode. Error: {exc}"
            )
            # Fall through to standard trainer

    from transformers import Trainer, TrainingArguments

    hf_kwargs = {k: v for k, v in config.items() if k not in _DISTILLATION_KEYS}
    training_args = TrainingArguments(
        output_dir=hf_kwargs.pop("output_dir", "./output"),
        **hf_kwargs,
    )
    return Trainer(model=model, args=training_args)
