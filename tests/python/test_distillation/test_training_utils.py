"""
Unit tests for training_utils.get_trainer()

get_trainer() always returns a SlowDriftTrainer.
When distillation=False it is a no-op wrapper around UnslothTrainer.
"""
import importlib
import sys
from pathlib import Path

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
    from torch.utils.data import TensorDataset
    from transformers import TrainingArguments

    from frozen_layer_modules.config import DistillationConfig
    from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer
    from unsloth.trainer import UnslothTrainer
    from training_utils import get_trainer


def _make_tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("gpt2")
    cfg.n_layer = 2; cfg.n_head = 2; cfg.n_embd = 16
    cfg.vocab_size = 64; cfg.n_positions = 16
    return AutoModelForCausalLM.from_config(cfg)


def _args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path), max_steps=1, no_cuda=True, report_to="none",
    )


def _ds():
    return TensorDataset(torch.randint(0, 64, (4, 8)), torch.randint(0, 64, (4, 8)))


class _DC:
    def __call__(self, batch):
        ids = torch.stack([b[0] for b in batch])
        return {"input_ids": ids, "labels": ids}


class TestGetTrainer:
    def test_returns_slow_drift_trainer(self, tmp_path):
        model = _make_tiny_model()
        trainer = get_trainer(model, training_args=_args(tmp_path),
                              train_dataset=_ds(), data_collator=_DC())
        assert isinstance(trainer, SlowDriftTrainer)
        assert isinstance(trainer, UnslothTrainer)

    def test_distillation_false_no_layers_frozen(self, tmp_path):
        model = _make_tiny_model()
        get_trainer(model,
                    config={"distillation": False},
                    training_args=_args(tmp_path),
                    train_dataset=_ds(), data_collator=_DC())
        # All parameters still trainable (no extra freeze applied)
        assert all(p.requires_grad for p in model.parameters())

    def test_distillation_true_freezes_layers(self, tmp_path):
        model = _make_tiny_model()
        trainer = get_trainer(
            model,
            distillation_config=DistillationConfig(distillation=True, frozen_layer_stride=2),
            training_args=_args(tmp_path),
            train_dataset=_ds(), data_collator=_DC(),
        )
        assert trainer._layer_freezer is not None
        assert len(trainer._layer_freezer.frozen_indices) > 0

    def test_distillation_keys_stripped_from_training_args(self, tmp_path):
        """Distillation keys in config dict must not reach TrainingArguments."""
        model = _make_tiny_model()
        # Should not raise even with all distillation keys present
        trainer = get_trainer(
            model,
            config={
                "distillation": False,
                "phase_unfreeze": False,
                "cka_lambda": 0.1,
                "phase_unfreeze_start": 0.3,
                "phase_unfreeze_end": 0.7,
                "frozen_layer_stride": 2,
                "output_dir": str(tmp_path),
            },
            train_dataset=_ds(), data_collator=_DC(),
        )
        assert isinstance(trainer, SlowDriftTrainer)

    def test_explicit_distillation_config_overrides_dict(self, tmp_path):
        model = _make_tiny_model()
        cfg = DistillationConfig(distillation=True, cka_lambda=0.42)
        trainer = get_trainer(
            model,
            config={"distillation": False, "cka_lambda": 0.99},
            distillation_config=cfg,
            training_args=_args(tmp_path),
            train_dataset=_ds(), data_collator=_DC(),
        )
        assert trainer._cfg.cka_lambda == pytest.approx(0.42)
