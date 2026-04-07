"""
Unit tests for SlowDriftTrainer and AlternatingLayerFreezer.

All tests run on CPU with tiny models – no GPU required.
"""
import copy
import sys
from pathlib import Path
from typing import Iterator

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frozen_layer_modules.frozen_layer_distillation import AlternatingLayerFreezer
from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer


# ---------------------------------------------------------------------------
# Tiny transformer-like model for testing (no GPU needed)
# ---------------------------------------------------------------------------

class _TinyLayer(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        return self.linear(x)


class _TinyTransformer(nn.Module):
    """Mimics the .model.layers structure of HuggingFace causal LMs."""
    def __init__(self, vocab: int = 32, d: int = 8, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([_TinyLayer(d) for _ in range(n_layers)])
        })
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.vocab = vocab
        self.d = d

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model["layers"]:
            x = layer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab), labels.view(-1)
            )

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.loss = loss
        return out


def _make_model(n_layers: int = 4) -> _TinyTransformer:
    return _TinyTransformer(n_layers=n_layers)


def _default_config():
    return {
        "drift_weight": 0.1,
        "restoration_factor": 0.99,
        "divergence_threshold": 0.15,
        "divergence_weight": 0.05,
        "distillation_mode": "on",
    }


def _tiny_dataloader(model: _TinyTransformer, batches: int = 2, batch_size: int = 2):
    """Yield simple random batches matching the tiny model vocabulary."""
    for _ in range(batches):
        ids = torch.randint(0, model.vocab, (batch_size, 4))
        yield {"input_ids": ids, "labels": ids}


# ---------------------------------------------------------------------------
# AlternatingLayerFreezer tests
# ---------------------------------------------------------------------------

class TestAlternatingLayerFreezer:
    def test_freeze_even_layers(self):
        model = _make_model(4)
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_even_layers()
        frozen = freezer.get_frozen_indices()
        assert frozen == {0, 2}
        for i in frozen:
            for p in freezer.layers[i].parameters():
                assert not p.requires_grad, f"Layer {i} should be frozen"

    def test_freeze_odd_layers(self):
        model = _make_model(4)
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_odd_layers()
        frozen = freezer.get_frozen_indices()
        assert frozen == {1, 3}

    def test_unfreeze_all(self):
        model = _make_model(4)
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_even_layers()
        freezer.unfreeze_all()
        assert freezer.get_frozen_indices() == set()
        for layer in freezer.layers:
            for p in layer.parameters():
                assert p.requires_grad

    def test_get_frozen_indices_is_copy(self):
        model = _make_model(4)
        freezer = AlternatingLayerFreezer(model)
        freezer.freeze_even_layers()
        indices = freezer.get_frozen_indices()
        indices.add(99)  # mutate the copy
        assert 99 not in freezer.get_frozen_indices()

    def test_non_transformer_raises(self):
        bad_model = nn.Linear(8, 8)
        with pytest.raises(ValueError, match="Transformer"):
            AlternatingLayerFreezer(bad_model)

    def test_compute_constraint_loss(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 4)
        model = _make_model(2)
        freezer = AlternatingLayerFreezer(model)
        loss = freezer.compute_constraint_loss(t1, t2, weight=1.0)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# SlowDriftTrainer tests
# ---------------------------------------------------------------------------

class TestSlowDriftTrainer:
    def test_init_requires_base_model(self):
        model = _make_model()
        with pytest.raises(ValueError, match="base_model"):
            SlowDriftTrainer(model, None, _default_config())

    def test_init_non_transformer_raises(self):
        bad = nn.Linear(8, 8)
        with pytest.raises(ValueError, match="Transformer"):
            SlowDriftTrainer(bad, bad, _default_config())

    def test_base_model_frozen(self):
        model = _make_model()
        base = copy.deepcopy(model)
        SlowDriftTrainer(model, base, _default_config())
        for p in base.parameters():
            assert not p.requires_grad

    def test_drift_penalty_near_zero_at_init(self):
        model = _make_model()
        base = copy.deepcopy(model)
        trainer = SlowDriftTrainer(model, base, _default_config())
        penalty = trainer.compute_drift_penalty()
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_drift_penalty_nonzero_after_modification(self):
        model = _make_model()
        base = copy.deepcopy(model)
        trainer = SlowDriftTrainer(model, base, _default_config())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))
        penalty = trainer.compute_drift_penalty()
        assert penalty.item() > 0

    def test_divergence_penalty_zero_for_identical_logits(self):
        model = _make_model()
        base = copy.deepcopy(model)
        trainer = SlowDriftTrainer(model, base, _default_config())
        logits = torch.randn(2, 4, model.vocab)
        penalty = trainer.compute_divergence_penalty(logits, logits.clone())
        # KL(p||p) = 0 which is below threshold → penalty = 0
        assert penalty.item() == pytest.approx(0.0, abs=1e-5)

    def test_divergence_penalty_positive_for_divergent_logits(self):
        model = _make_model()
        base = copy.deepcopy(model)
        cfg = _default_config()
        cfg["divergence_threshold"] = 0.0  # any KL > 0 triggers penalty
        trainer = SlowDriftTrainer(model, base, cfg)
        logits = torch.randn(2, 4, model.vocab)
        base_logits = torch.randn(2, 4, model.vocab) * 5  # very different
        penalty = trainer.compute_divergence_penalty(logits, base_logits)
        assert penalty.item() > 0

    def test_restore_frozen_layers_moves_toward_snapshot(self):
        model = _make_model(2)
        base = copy.deepcopy(model)
        cfg = _default_config()
        cfg["restoration_factor"] = 0.5  # large restoration for clear signal
        trainer = SlowDriftTrainer(model, base, cfg)

        # Modify frozen layer 0
        with torch.no_grad():
            for p in trainer.layer_freezer.layers[0].parameters():
                p.fill_(10.0)

        snapshot_val = list(trainer._base_snapshot.values())[0]
        trainer.restore_frozen_layers()

        for p in trainer.layer_freezer.layers[0].parameters():
            # Should have moved closer to 0.0 (original) from 10.0
            assert p.abs().max().item() < 10.0

    def test_train_epoch_returns_metrics(self):
        model = _make_model()
        base = copy.deepcopy(model)
        trainer = SlowDriftTrainer(model, base, _default_config())
        dl = list(_tiny_dataloader(model, batches=2))
        metrics = trainer.train_epoch(dl)
        assert "loss" in metrics
        assert "drift_penalty" in metrics
        assert "divergence_penalty" in metrics
        assert metrics["steps"] == 2

    def test_train_epoch_loss_is_finite(self):
        model = _make_model()
        base = copy.deepcopy(model)
        trainer = SlowDriftTrainer(model, base, _default_config())
        dl = list(_tiny_dataloader(model, batches=3))
        metrics = trainer.train_epoch(dl)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
