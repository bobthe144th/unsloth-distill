"""
training_utils – trainer selection factory.

get_trainer() returns:
  - SlowDriftTrainer (extends UnslothTrainer) when distillation_mode='on'
  - UnslothTrainer                            when distillation_mode='off'

Both trainers are HuggingFace-compatible and called via .train().
All Unsloth speedups (Triton kernels, Q-GaLore, sample packing, etc.) are
active in both paths.
"""
import logging
from typing import Any, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

# Distillation-specific keys that must not be forwarded to TrainingArguments
_DISTILLATION_KEYS = frozenset(
    {"distillation_mode", "drift_weight", "restoration_factor",
     "divergence_threshold", "divergence_weight"}
)


def get_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    base_model: Optional[nn.Module] = None,
    training_args=None,
    **trainer_kwargs,
):
    """
    Select and construct an appropriate trainer.

    Args:
        model:         The model to train (fine-tuning model).
        config:        Dict that may contain distillation settings and/or
                       HuggingFace TrainingArguments fields.
        base_model:    Required when distillation_mode='on'.
        training_args: Optional pre-built TrainingArguments / SFTConfig.
                       If None, a default TrainingArguments is constructed
                       from non-distillation keys in config.
        **trainer_kwargs: Forwarded to the trainer constructor (e.g.
                          train_dataset, eval_dataset, tokenizer).

    Returns:
        SlowDriftTrainer when distillation_mode='on' (import succeeds),
        UnslothTrainer   otherwise.
    """
    mode = config.get("distillation_mode", "off")
    if isinstance(mode, str):
        mode = mode.lower().strip()

    # Build TrainingArguments from config if not provided
    if training_args is None:
        from unsloth.trainer import UnslothTrainingArguments
        hf_kwargs = {k: v for k, v in config.items() if k not in _DISTILLATION_KEYS}
        training_args = UnslothTrainingArguments(
            output_dir=hf_kwargs.pop("output_dir", "./output"),
            **hf_kwargs,
        )

    if mode == "on":
        try:
            from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer
            distillation_config = {k: config[k] for k in _DISTILLATION_KEYS if k in config}
            return SlowDriftTrainer(
                base_model=base_model,
                distillation_config=distillation_config,
                model=model,
                args=training_args,
                **trainer_kwargs,
            )
        except Exception as exc:
            logger.error(
                "Failed to construct SlowDriftTrainer. "
                f"Training in standard mode. Error: {exc}"
            )
            # re-raise config errors so callers know they're misconfigured
            if isinstance(exc, (ValueError, TypeError)):
                raise

    from unsloth.trainer import UnslothTrainer
    return UnslothTrainer(model=model, args=training_args, **trainer_kwargs)
