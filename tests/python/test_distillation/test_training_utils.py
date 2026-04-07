"""
Unit tests for training_utils.get_trainer()

get_trainer() now returns:
  - SlowDriftTrainer (extends UnslothTrainer) when distillation_mode='on'
  - UnslothTrainer                            when distillation_mode='off'

Tests that require torch/transformers/trl are skipped when those are absent.
"""
import copy
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_HAS_TRL = importlib.util.find_spec("trl") is not None
_CAN_RUN = _HAS_TORCH and _HAS_TRANSFORMERS and _HAS_TRL

pytestmark = pytest.mark.skipif(not _CAN_RUN, reason="torch/transformers/trl not installed")

if _CAN_RUN:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from transformers import TrainingArguments
    from unsloth.trainer import UnslothTrainer
    from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer
    from training_utils import get_trainer


def _make_tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("gpt2")
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 16
    cfg.vocab_size = 128
    cfg.n_positions = 32
    return AutoModelForCausalLM.from_config(cfg)


def _make_training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        no_cuda=True,
        report_to="none",
    )


def _tiny_dataset():
    ids = torch.randint(0, 128, (4, 8))
    return TensorDataset(ids, ids)


class _DictCollator:
    def __call__(self, batch):
        ids = torch.stack([b[0] for b in batch])
        return {"input_ids": ids, "labels": ids}


class TestGetTrainer:
    def test_off_returns_unsloth_trainer(self, tmp_path):
        model = _make_tiny_model()
        args = _make_training_args(tmp_path)
        trainer = get_trainer(
            model, {"distillation_mode": "off"},
            training_args=args,
        )
        assert isinstance(trainer, UnslothTrainer)

    def test_on_returns_slow_drift_trainer(self, tmp_path):
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        trainer = get_trainer(
            model,
            {"distillation_mode": "on", "drift_weight": 0.1},
            base_model=base,
            training_args=args,
            train_dataset=ds,
            data_collator=_DictCollator(),
        )
        assert isinstance(trainer, SlowDriftTrainer)

    def test_on_without_base_model_raises(self, tmp_path):
        model = _make_tiny_model()
        args = _make_training_args(tmp_path)
        with pytest.raises(ValueError, match="base_model"):
            get_trainer(
                model,
                {"distillation_mode": "on"},
                base_model=None,
                training_args=args,
            )

    def test_mode_case_insensitive(self, tmp_path):
        model = _make_tiny_model()
        args = _make_training_args(tmp_path)
        trainer = get_trainer(
            model, {"distillation_mode": "OFF"},
            training_args=args,
        )
        assert isinstance(trainer, UnslothTrainer)

    def test_missing_mode_defaults_to_off(self, tmp_path):
        model = _make_tiny_model()
        args = _make_training_args(tmp_path)
        trainer = get_trainer(model, {}, training_args=args)
        assert isinstance(trainer, UnslothTrainer)

    def test_distillation_keys_not_forwarded_to_training_args(self, tmp_path):
        """Distillation keys must be stripped before building TrainingArguments."""
        model = _make_tiny_model()
        # Should not raise even with all distillation keys present
        trainer = get_trainer(model, {
            "distillation_mode": "off",
            "drift_weight": 0.1,
            "restoration_factor": 0.99,
            "divergence_threshold": 0.15,
            "divergence_weight": 0.05,
            "output_dir": str(tmp_path),
        })
        assert isinstance(trainer, UnslothTrainer)

    def test_fallback_on_import_error(self, tmp_path, monkeypatch):
        """When frozen_layer_modules fails to import, UnslothTrainer is returned."""
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        args = _make_training_args(tmp_path)

        with patch.dict("sys.modules", {"frozen_layer_modules.slow_drift_frozen_layers": None}):
            trainer = get_trainer(
                model,
                {"distillation_mode": "on"},
                base_model=base,
                training_args=args,
            )
        assert isinstance(trainer, UnslothTrainer)
        assert not isinstance(trainer, SlowDriftTrainer)
