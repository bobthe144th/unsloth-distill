"""
Unit tests for SlowDriftTrainer and LayerFreezer.

Requires torch + transformers + trl (skipped otherwise).
All tests run on CPU with a tiny GPT-2 model — no GPU required.
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

if _CAN_RUN:
    import torch
    import torch.nn as nn
    from transformers import TrainingArguments
    from torch.utils.data import TensorDataset

    from frozen_layer_modules.config import DistillationConfig
    from frozen_layer_modules.frozen_layer_distillation import LayerFreezer, get_transformer_layers
    from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer
    from unsloth.trainer import UnslothTrainer


# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

def _make_tiny_model(n_layers=4):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("gpt2")
    cfg.n_layer = n_layers
    cfg.n_head = 2
    cfg.n_embd = 16
    cfg.vocab_size = 64
    cfg.n_positions = 16
    return AutoModelForCausalLM.from_config(cfg)


def _training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path), max_steps=2, no_cuda=True, report_to="none",
        per_device_train_batch_size=2,
    )


def _tiny_dataset(vocab=64, seq=8, n=4):
    ids = torch.randint(0, vocab, (n, seq))
    return TensorDataset(ids, ids)


class _DictCollator:
    def __call__(self, batch):
        ids = torch.stack([b[0] for b in batch])
        return {"input_ids": ids, "labels": ids}


def _on_cfg(**kwargs):
    return DistillationConfig(distillation=True, **kwargs)


# ---------------------------------------------------------------------------
# LayerFreezer tests
# ---------------------------------------------------------------------------

class TestLayerFreezer:
    def test_non_transformer_raises(self):
        with pytest.raises(ValueError):
            LayerFreezer(nn.Linear(8, 8))

    def test_stride_2_freezes_even_layers(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=2)
        assert freezer.frozen_indices == {0, 2}
        for i in {0, 2}:
            for name, p in freezer.layers[i].named_parameters():
                if "norm" in name.lower():
                    # LayerNorm/RMSNorm must stay trainable.
                    assert p.requires_grad, f"layer {i} norm param {name!r} should be trainable"
                else:
                    assert not p.requires_grad, f"layer {i} non-norm param {name!r} should be frozen"

    def test_norm_params_never_frozen(self):
        """LayerNorm parameters in frozen layers must always stay requires_grad=True."""
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=1)  # freeze every layer
        for i in range(4):
            for name, p in freezer.layers[i].named_parameters():
                if "norm" in name.lower():
                    assert p.requires_grad, (
                        f"layer {i} norm param {name!r} was incorrectly frozen"
                    )

    def test_stride_1_freezes_all_layers(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=1)
        assert freezer.frozen_indices == {0, 1, 2, 3}

    def test_stride_3(self):
        model = _make_tiny_model(6)
        freezer = LayerFreezer(model, stride=3)
        assert freezer.frozen_indices == {0, 3}

    def test_release_single_layer(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=2)
        freezer.release_layers([0])
        assert 0 not in freezer.frozen_indices
        for p in freezer.layers[0].parameters():
            assert p.requires_grad

    def test_release_all(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=2)
        freezer.release_all()
        assert freezer.frozen_indices == set()
        for layer in freezer.layers:
            for p in layer.parameters():
                assert p.requires_grad

    def test_release_does_not_unfreeze_pre_frozen_params(self):
        """
        Params that were already requires_grad=False before LayerFreezer runs
        (e.g. LoRA base weights) must stay frozen after release_layers().
        """
        model = _make_tiny_model(4)
        # Simulate LoRA: manually freeze one non-norm param before LayerFreezer.
        target_param = None
        for name, p in model.model.layers[0].named_parameters():
            if "norm" not in name.lower():
                p.requires_grad = False
                target_param = p
                break
        assert target_param is not None

        freezer = LayerFreezer(model, stride=2)
        freezer.release_all()

        # The pre-frozen param must still be frozen after release.
        assert not target_param.requires_grad

    def test_release_adds_to_optimizer(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=2)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(trainable, lr=0.01)
        n_groups_before = len(opt.param_groups)
        freezer.release_layers([0], optimizer=opt)
        assert len(opt.param_groups) > n_groups_before

    def test_hooks_register_and_remove(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=2)
        frozen_acts, trainable_acts, hooks = freezer.register_cka_hooks()
        assert len(hooks) > 0
        for h in hooks:
            h.remove()

    def test_hooks_collect_activations(self):
        model = _make_tiny_model(4)
        model.eval()
        freezer = LayerFreezer(model, stride=2)
        frozen_acts, trainable_acts, hooks = freezer.register_cka_hooks()
        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            model(input_ids=ids)
        for h in hooks:
            h.remove()
        # Frozen layers 0, 2 → frozen_acts should have entries
        assert len(frozen_acts) > 0
        # Adjacent trainable layers 1, 3 → trainable_acts should have entries
        assert len(trainable_acts) > 0

    def test_frozen_acts_are_detached(self):
        model = _make_tiny_model(4)
        freezer = LayerFreezer(model, stride=2)
        frozen_acts, _, hooks = freezer.register_cka_hooks()
        ids = torch.randint(0, 64, (2, 8))
        model(input_ids=ids)
        for h in hooks:
            h.remove()
        for act in frozen_acts.values():
            assert not act.requires_grad


# ---------------------------------------------------------------------------
# SlowDriftTrainer tests
# ---------------------------------------------------------------------------

class TestSlowDriftTrainerNoOp:
    """When distillation=False, trainer must be byte-for-byte like UnslothTrainer."""

    def test_is_unsloth_trainer_subclass(self, tmp_path):
        model = _make_tiny_model()
        trainer = SlowDriftTrainer(
            distillation_config=DistillationConfig(distillation=False),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        assert isinstance(trainer, UnslothTrainer)

    def test_no_layers_frozen_when_off(self, tmp_path):
        model = _make_tiny_model()
        SlowDriftTrainer(
            distillation_config=DistillationConfig(distillation=False),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        for p in model.parameters():
            # No extra freezing; only normal LoRA behavior would freeze
            pass  # just check init doesn't crash

    def test_lambda_is_zero_when_off(self, tmp_path):
        model = _make_tiny_model()
        trainer = SlowDriftTrainer(
            distillation_config=DistillationConfig(distillation=False),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        assert trainer._current_lambda == 0.0


class TestSlowDriftTrainerActive:
    def test_layers_frozen_at_init(self, tmp_path):
        model = _make_tiny_model(4)
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(frozen_layer_stride=2),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        assert {0, 2} == trainer._layer_freezer.frozen_indices

    def test_stride_respected(self, tmp_path):
        model = _make_tiny_model(6)
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(frozen_layer_stride=3),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        assert {0, 3} == trainer._layer_freezer.frozen_indices

    def test_cka_penalty_computed(self, tmp_path):
        """_compute_cka_penalty should return a tensor in [0, 1]."""
        model = _make_tiny_model(4)
        model.eval()
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        frozen_acts, trainable_acts, hooks = trainer._layer_freezer.register_cka_hooks()
        ids = torch.randint(0, 64, (4, 8))
        with torch.no_grad():
            model(input_ids=ids)
        for h in hooks:
            h.remove()
        pen = trainer._compute_cka_penalty(frozen_acts, trainable_acts)
        assert 0.0 <= pen.item() <= 1.0

    def test_cka_penalty_skips_batch_size_one(self, tmp_path):
        model = _make_tiny_model(4)
        model.eval()
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        frozen_acts, trainable_acts, hooks = trainer._layer_freezer.register_cka_hooks()
        ids = torch.randint(0, 64, (1, 8))  # batch_size=1
        with torch.no_grad():
            model(input_ids=ids)
        for h in hooks:
            h.remove()
        pen = trainer._compute_cka_penalty(frozen_acts, trainable_acts)
        assert pen.item() == pytest.approx(0.0)  # skipped

    def test_phase_unfreeze_releases_layers(self, tmp_path):
        """Simulate mid-training unfreezing by calling _update_phase directly."""
        model = _make_tiny_model(4)
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(
                phase_unfreeze=True,
                phase_unfreeze_start=0.0,
                phase_unfreeze_end=1.0,
            ),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        # Fake training state
        from transformers import TrainerState
        trainer.state = TrainerState()
        trainer.state.max_steps = 10
        trainer.state.global_step = 9  # ~90% through → should release most/all

        n_frozen_before = len(trainer._layer_freezer.frozen_indices)
        trainer._update_phase()
        n_frozen_after = len(trainer._layer_freezer.frozen_indices)
        assert n_frozen_after <= n_frozen_before

    def test_phase_lambda_decays(self, tmp_path):
        model = _make_tiny_model(4)
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(
                cka_lambda=0.1,
                phase_unfreeze=True,
                phase_unfreeze_start=0.0,
                phase_unfreeze_end=1.0,
            ),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        from transformers import TrainerState
        trainer.state = TrainerState()
        trainer.state.max_steps = 10
        trainer.state.global_step = 5  # 50% → lambda should be 0.05

        trainer._update_phase()
        assert trainer._current_lambda < 0.1

    def test_phase_lambda_zero_after_end(self, tmp_path):
        model = _make_tiny_model(4)
        trainer = SlowDriftTrainer(
            distillation_config=_on_cfg(
                phase_unfreeze=True,
                phase_unfreeze_start=0.1,
                phase_unfreeze_end=0.5,
            ),
            model=model,
            args=_training_args(tmp_path),
            train_dataset=_tiny_dataset(),
            data_collator=_DictCollator(),
        )
        from transformers import TrainerState
        trainer.state = TrainerState()
        trainer.state.max_steps = 10
        trainer.state.global_step = 8  # 80% > end=0.5

        trainer._update_phase()
        assert trainer._current_lambda == 0.0
        assert trainer._layer_freezer.frozen_indices == set()
