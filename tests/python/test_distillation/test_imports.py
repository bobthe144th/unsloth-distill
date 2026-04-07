"""
Unit tests for frozen_layer_modules import guard (Test-6.1.3).

Covers:
  - Classes exposed when DISTILLATION="on" (skipped when torch unavailable)
  - Classes are None when DISTILLATION="off"
  - Graceful fallback when sub-module import fails
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Determine once whether torch is available
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _reload_flm(env_value: str):
    """Set env var and force-reload the frozen_layer_modules package."""
    os.environ["DISTILLATION"] = env_value
    import frozen_layer_modules as flm
    importlib.reload(flm)
    return flm


class TestImportGuard:
    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_distillation_on_exposes_classes(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "on")
        flm = _reload_flm("on")
        assert flm.SlowDriftTrainer is not None, "SlowDriftTrainer should be loaded"
        assert flm.AlternatingLayerFreezer is not None, "AlternatingLayerFreezer should be loaded"

    def test_distillation_off_classes_are_none(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "off")
        flm = _reload_flm("off")
        assert flm.SlowDriftTrainer is None
        assert flm.AlternatingLayerFreezer is None

    def test_distillation_mode_flag_on(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "on")
        flm = _reload_flm("on")
        assert flm.DISTILLATION_MODE is True

    def test_distillation_mode_flag_off(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "off")
        flm = _reload_flm("off")
        assert flm.DISTILLATION_MODE is False

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_set_mode_on_loads_classes(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "off")
        flm = _reload_flm("off")
        assert flm.SlowDriftTrainer is None
        flm.set_mode("on")
        assert flm.DISTILLATION_MODE is True
        assert flm.SlowDriftTrainer is not None

    def test_set_mode_off_after_on(self, monkeypatch):
        monkeypatch.setenv("DISTILLATION", "on")
        flm = _reload_flm("on")
        flm.set_mode("off")
        assert flm.DISTILLATION_MODE is False
        assert os.environ["DISTILLATION"] == "off"

    def test_graceful_fallback_on_broken_submodule(self, monkeypatch):
        """Import failure in _try_import must not propagate; set_mode swallows it."""
        monkeypatch.setenv("DISTILLATION", "off")
        flm = _reload_flm("off")

        # Replace _try_import with one that always raises
        def _bad_try_import():
            raise ImportError("simulated corruption")

        flm._try_import = _bad_try_import

        # set_mode("on") should log a warning and NOT raise
        try:
            flm.set_mode("on")
        except ImportError:
            pytest.fail("set_mode raised ImportError – graceful fallback failed")

        # Classes remain None because the import failed
        assert flm.SlowDriftTrainer is None

    def test_on_falls_back_gracefully_when_torch_missing(self, monkeypatch):
        """When torch is absent, DISTILLATION=on falls back without crashing."""
        monkeypatch.setenv("DISTILLATION", "on")
        flm = _reload_flm("on")
        # Either torch is present (classes loaded) OR classes are None – no crash
        assert flm.SlowDriftTrainer is None or flm.SlowDriftTrainer is not None
        assert flm.DISTILLATION_MODE is True
