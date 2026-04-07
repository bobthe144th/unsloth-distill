"""
Unit tests for training_utils.get_trainer()
"""
import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training_utils import get_trainer


# ---------------------------------------------------------------------------
# Minimal transformer model (same as in test_slow_drift_trainer)
# ---------------------------------------------------------------------------

class _TinyLayer(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        return self.linear(x)


class _TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([_TinyLayer(d) for _ in range(n_layers)])
        })
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model["layers"]:
            x = layer(x)
        logits = self.lm_head(x)
        out = MagicMock()
        out.logits = logits
        out.loss = torch.tensor(0.5)
        return out


def _make_model():
    return _TinyTransformer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetTrainer:
    def test_off_returns_hf_trainer(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "off")
        from transformers import Trainer
        model = _make_model()
        trainer = get_trainer(model, {"distillation_mode": "off"})
        assert isinstance(trainer, Trainer)

    def test_on_returns_slow_drift_trainer(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "on")
        from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer
        model = _make_model()
        base = copy.deepcopy(model)
        trainer = get_trainer(model, {"distillation_mode": "on"}, base_model=base)
        assert isinstance(trainer, SlowDriftTrainer)

    def test_mode_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "off")
        from transformers import Trainer
        model = _make_model()
        trainer = get_trainer(model, {"distillation_mode": "OFF"})
        assert isinstance(trainer, Trainer)

    def test_missing_mode_defaults_to_standard(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "off")
        from transformers import Trainer
        model = _make_model()
        trainer = get_trainer(model, {})
        assert isinstance(trainer, Trainer)

    def test_fallback_on_import_error(self, monkeypatch):
        """When frozen_layer_modules fails to import, standard Trainer is returned."""
        monkeypatch.setenv("DISTILLATION", "on")
        from transformers import Trainer

        # Patch the import inside get_trainer to simulate failure
        import training_utils as tu
        original = __builtins__

        with patch.dict(
            "sys.modules",
            {"frozen_layer_modules": None},
        ):
            model = _make_model()
            # This will hit the ImportError branch because the module is None
            trainer = get_trainer(model, {"distillation_mode": "on"})
        assert isinstance(trainer, Trainer)

    def test_distillation_keys_stripped_from_hf_args(self, monkeypatch):
        """Distillation-specific keys must not be forwarded to TrainingArguments."""
        monkeypatch.setenv("DISTILLATION", "off")
        model = _make_model()
        # Should not raise even though drift_weight etc. are present
        trainer = get_trainer(model, {
            "distillation_mode": "off",
            "drift_weight": 0.1,
            "restoration_factor": 0.99,
            "divergence_threshold": 0.15,
            "divergence_weight": 0.05,
        })
        from transformers import Trainer
        assert isinstance(trainer, Trainer)
