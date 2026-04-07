"""
Unit tests for SlowDriftTrainer and AlternatingLayerFreezer.

SlowDriftTrainer now extends UnslothTrainer (SFTTrainer).
Tests that require the full HF Trainer stack (model + training_args + dataset)
are skipped when torch or transformers is unavailable.

Tests that only exercise pure-Python/penalty logic run in all environments.
"""
import copy
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

# ---------------------------------------------------------------------------
# Imports (guarded so collection doesn't fail without torch)
# ---------------------------------------------------------------------------
if _CAN_RUN:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from frozen_layer_modules.frozen_layer_distillation import AlternatingLayerFreezer
    from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer


# ---------------------------------------------------------------------------
# Minimal transformer model (CPU, no GPU needed)
# ---------------------------------------------------------------------------

def _make_tiny_model():
    """Return a tiny GPT-2 config causal LM suitable for CPU tests."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("gpt2")
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 16
    cfg.vocab_size = 128
    cfg.n_positions = 32
    return AutoModelForCausalLM.from_config(cfg)


def _default_distillation_config():
    return {
        "drift_weight": 0.1,
        "restoration_factor": 0.99,
        "divergence_threshold": 0.15,
        "divergence_weight": 0.05,
    }


def _make_training_args(tmp_path):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=2,         # keep tests fast
        logging_steps=1,
        no_cuda=True,        # CPU-only
        report_to="none",
    )


def _tiny_dataset(vocab=128, seq=8, n=4):
    ids = torch.randint(0, vocab, (n, seq))
    return TensorDataset(ids, ids)


class _DictCollator:
    def __call__(self, batch):
        input_ids = torch.stack([b[0] for b in batch])
        return {"input_ids": input_ids, "labels": input_ids}


# ---------------------------------------------------------------------------
# AlternatingLayerFreezer tests (unchanged API)
# ---------------------------------------------------------------------------

class TestAlternatingLayerFreezer:
    def test_non_transformer_raises(self):
        with pytest.raises(ValueError, match="Transformer"):
            AlternatingLayerFreezer(nn.Linear(8, 8))

    def test_freeze_even_layers(self):
        model = _make_tiny_model()
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_even_layers()
        assert 0 in freezer.get_frozen_indices()
        for p in freezer.layers[0].parameters():
            assert not p.requires_grad

    def test_freeze_odd_layers(self):
        model = _make_tiny_model()
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_odd_layers()
        assert 1 in freezer.get_frozen_indices()

    def test_unfreeze_all(self):
        model = _make_tiny_model()
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_even_layers()
        freezer.unfreeze_all()
        assert freezer.get_frozen_indices() == set()
        for layer in freezer.layers:
            for p in layer.parameters():
                assert p.requires_grad

    def test_alternating_step(self):
        model = _make_tiny_model()
        freezer = AlternatingLayerFreezer(model)
        # step() alternates: after first step, even layers frozen
        freezer.step()
        frozen_after_1 = set(freezer.get_frozen_indices())
        freezer.step()
        frozen_after_2 = set(freezer.get_frozen_indices())
        assert frozen_after_1 != frozen_after_2

    def test_compute_constraint_loss(self):
        model = _make_tiny_model()
        freezer = AlternatingLayerFreezer(model)
        t1 = torch.randn(2, 8)
        t2 = torch.randn(2, 8)
        loss = freezer.compute_constraint_loss(t1, t2, weight=1.0)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# SlowDriftTrainer tests (new HF Trainer-based API)
# ---------------------------------------------------------------------------

class TestSlowDriftTrainer:
    def test_requires_base_model(self, tmp_path):
        model = _make_tiny_model()
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        with pytest.raises(ValueError, match="base_model"):
            SlowDriftTrainer(
                base_model=None,
                distillation_config=_default_distillation_config(),
                model=model,
                args=args,
                train_dataset=ds,
                data_collator=_DictCollator(),
            )

    def test_non_transformer_raises(self, tmp_path):
        bad_model = nn.Sequential(nn.Linear(8, 8))
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        with pytest.raises(ValueError, match="Transformer"):
            SlowDriftTrainer(
                base_model=bad_model,
                distillation_config=_default_distillation_config(),
                model=bad_model,
                args=args,
                train_dataset=ds,
                data_collator=_DictCollator(),
            )

    def test_base_model_frozen_after_init(self, tmp_path):
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        SlowDriftTrainer(
            base_model=base,
            distillation_config=_default_distillation_config(),
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=_DictCollator(),
        )
        for p in base.parameters():
            assert not p.requires_grad

    def test_drift_penalty_near_zero_at_init(self, tmp_path):
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        trainer = SlowDriftTrainer(
            base_model=base,
            distillation_config=_default_distillation_config(),
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=_DictCollator(),
        )
        penalty = trainer._compute_drift_penalty(model)
        # At init, snapshot == current weights → drift ≈ 0
        assert penalty.item() == pytest.approx(0.0, abs=1e-5)

    def test_drift_penalty_nonzero_after_modification(self, tmp_path):
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        trainer = SlowDriftTrainer(
            base_model=base,
            distillation_config=_default_distillation_config(),
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=_DictCollator(),
        )
        # Freeze even layers so drift_penalty targets them
        trainer._layer_freezer.freeze_even_layers()
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    p.add_(torch.ones_like(p))
        penalty = trainer._compute_drift_penalty(model)
        assert penalty.item() > 0

    def test_restore_moves_params_toward_snapshot(self, tmp_path):
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        cfg = _default_distillation_config()
        cfg["restoration_factor"] = 0.5  # strong restoration for clear signal
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        trainer = SlowDriftTrainer(
            base_model=base,
            distillation_config=cfg,
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=_DictCollator(),
        )
        # Drastically move frozen layer params
        trainer._layer_freezer.freeze_even_layers()
        with torch.no_grad():
            for i in trainer._layer_freezer.get_frozen_indices():
                for p in trainer._layer_freezer.layers[i].parameters():
                    p.fill_(100.0)

        trainer._restore_frozen_layers(model)
        for i in trainer._layer_freezer.get_frozen_indices():
            for p in trainer._layer_freezer.layers[i].parameters():
                # Should have moved well below 100
                assert p.abs().max().item() < 100.0

    def test_is_unsloth_trainer_subclass(self, tmp_path):
        """Verify the inheritance chain is correct."""
        from unsloth.trainer import UnslothTrainer
        model = _make_tiny_model()
        base = copy.deepcopy(model)
        args = _make_training_args(tmp_path)
        ds = _tiny_dataset()
        trainer = SlowDriftTrainer(
            base_model=base,
            distillation_config=_default_distillation_config(),
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=_DictCollator(),
        )
        assert isinstance(trainer, UnslothTrainer)
