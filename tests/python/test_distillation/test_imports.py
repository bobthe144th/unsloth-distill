"""
Unit tests for frozen_layer_modules import behaviour.

The package is always importable.  When torch is absent, sub-module imports
fail gracefully and stubs are set to None.
"""
import importlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestPackageImport:
    def test_package_importable(self):
        import frozen_layer_modules  # noqa: F401

    def test_config_always_importable(self):
        from frozen_layer_modules.config import DistillationConfig, load_config
        assert DistillationConfig is not None
        assert load_config is not None

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_cka_loss_importable(self):
        from frozen_layer_modules.cka_loss import cka_penalty, linear_cka
        assert cka_penalty is not None
        assert linear_cka is not None

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_layer_freezer_importable(self):
        from frozen_layer_modules.frozen_layer_distillation import LayerFreezer
        assert LayerFreezer is not None

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_slow_drift_trainer_importable(self):
        from frozen_layer_modules.slow_drift_frozen_layers import SlowDriftTrainer
        assert SlowDriftTrainer is not None

    def test_distillation_config_defaults(self):
        from frozen_layer_modules.config import DistillationConfig
        cfg = DistillationConfig()
        assert cfg.distillation is False
        assert cfg.phase_unfreeze is False
        assert cfg.cka_lambda == pytest.approx(0.1)
        assert cfg.frozen_layer_stride == 2
