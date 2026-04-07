"""
training_utils — trainer selection factory.

get_trainer() returns a SlowDriftTrainer in all cases.
When distillation_config.distillation is False (the default), SlowDriftTrainer
is a transparent no-op wrapper around UnslothTrainer — no layers are frozen,
no hooks are registered, no additional loss terms are added.

All Unsloth speedups are active in both modes.
"""
import logging
from typing import Any, Dict, Optional, Union

import torch.nn as nn

from frozen_layer_modules.config import DistillationConfig, load_config
from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer

logger = logging.getLogger(__name__)

# Keys consumed by distillation config — not valid TrainingArguments fields
_DISTILLATION_KEYS = frozenset({
    "distillation", "phase_unfreeze", "cka_lambda",
    "phase_unfreeze_start", "phase_unfreeze_end", "frozen_layer_stride",
})


def get_trainer(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    training_args=None,
    distillation_config: Optional[Union[DistillationConfig, Dict[str, Any]]] = None,
    **trainer_kwargs,
) -> SlowDriftTrainer:
    """
    Construct a SlowDriftTrainer.

    Args:
        model:               The model to train.
        config:              Optional flat dict.  Distillation keys are extracted
                             into DistillationConfig; remaining keys are used to
                             build TrainingArguments if training_args is None.
        training_args:       Pre-built TrainingArguments / SFTConfig.  Takes
                             precedence over config for HF trainer arguments.
        distillation_config: Explicit DistillationConfig or dict.  When provided,
                             takes precedence over distillation keys in config.
        **trainer_kwargs:    Forwarded to SlowDriftTrainer (e.g. train_dataset,
                             eval_dataset, tokenizer / processing_class).

    Returns:
        SlowDriftTrainer — behaves as plain UnslothTrainer when distillation=False.
    """
    config = config or {}

    # --- Resolve distillation config ---
    if distillation_config is None:
        dist_kwargs = {k: config[k] for k in _DISTILLATION_KEYS if k in config}
        distillation_config = load_config(overrides=dist_kwargs or None)
    elif isinstance(distillation_config, dict):
        distillation_config = load_config(overrides=distillation_config)

    # --- Build TrainingArguments if not provided ---
    if training_args is None:
        from unsloth.trainer import UnslothTrainingArguments
        hf_kwargs = {k: v for k, v in config.items() if k not in _DISTILLATION_KEYS}
        training_args = UnslothTrainingArguments(
            output_dir=hf_kwargs.pop("output_dir", "./output"),
            **hf_kwargs,
        )

    return SlowDriftTrainer(
        distillation_config=distillation_config,
        model=model,
        args=training_args,
        **trainer_kwargs,
    )
