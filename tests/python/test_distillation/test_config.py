"""
Unit tests for frozen_layer_modules.config

Covers:
  - Test-6.1.1: Flag parsing (on/off/uppercase/invalid)
  - Test-6.1.2: Config file loading and env-var priority
"""
import os
import sys
import warnings
from pathlib import Path

import pytest

# Ensure the package root is on sys.path so frozen_layer_modules is importable
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(monkeypatch, env_value=None, config_path=None):
    """Reload load_config with a specific env var value."""
    if env_value is None:
        monkeypatch.delenv("DISTILLATION", raising=False)
    else:
        monkeypatch.setenv("DISTILLATION", env_value)
    from frozen_layer_modules.config import load_config
    return load_config(config_path=config_path)


# ---------------------------------------------------------------------------
# Flag parsing (Test-6.1.1)
# ---------------------------------------------------------------------------

class TestFlagParsing:
    def test_on_lowercase(self, monkeypatch):
        cfg = _load(monkeypatch, "on")
        assert cfg.distillation_mode == "on"

    def test_off_lowercase(self, monkeypatch):
        cfg = _load(monkeypatch, "off")
        assert cfg.distillation_mode == "off"

    def test_on_uppercase(self, monkeypatch):
        cfg = _load(monkeypatch, "ON")
        assert cfg.distillation_mode == "on"

    def test_off_uppercase(self, monkeypatch):
        cfg = _load(monkeypatch, "OFF")
        assert cfg.distillation_mode == "off"

    def test_on_mixed_case(self, monkeypatch):
        cfg = _load(monkeypatch, "On")
        assert cfg.distillation_mode == "on"

    def test_invalid_warns_and_defaults_off(self, monkeypatch):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _load(monkeypatch, "maybe")
        assert cfg.distillation_mode == "off"
        messages = [str(w.message) for w in caught]
        assert any("maybe" in m for m in messages), f"Expected warning mentioning 'maybe', got: {messages}"

    def test_missing_env_defaults_off(self, monkeypatch):
        cfg = _load(monkeypatch, env_value=None)
        assert cfg.distillation_mode == "off"


# ---------------------------------------------------------------------------
# Config file loading (Test-6.1.2)
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_yaml_loaded_when_present(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISTILLATION", raising=False)
        cfg_file = tmp_path / ".distillation_config.yaml"
        cfg_file.write_text("distillation_mode: on\ndrift_weight: 0.2\n")
        cfg = _load(monkeypatch, config_path=cfg_file)
        assert cfg.distillation_mode == "on"
        assert cfg.drift_weight == pytest.approx(0.2)

    def test_yaml_missing_uses_defaults(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISTILLATION", raising=False)
        cfg = _load(monkeypatch, config_path=tmp_path / "nonexistent.yaml")
        assert cfg.distillation_mode == "off"
        assert cfg.drift_weight == pytest.approx(0.1)

    def test_env_overrides_yaml(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / ".distillation_config.yaml"
        cfg_file.write_text("distillation_mode: off\n")
        monkeypatch.setenv("DISTILLATION", "on")
        cfg = _load(monkeypatch, env_value="on", config_path=cfg_file)
        assert cfg.distillation_mode == "on"

    def test_non_mode_keys_from_yaml_survive_env_override(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / ".distillation_config.yaml"
        cfg_file.write_text("drift_weight: 0.42\nrestoration_factor: 0.95\n")
        cfg = _load(monkeypatch, env_value="off", config_path=cfg_file)
        assert cfg.drift_weight == pytest.approx(0.42)
        assert cfg.restoration_factor == pytest.approx(0.95)

    def test_defaults_when_no_yaml_no_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISTILLATION", raising=False)
        cfg = _load(monkeypatch, config_path=tmp_path / "missing.yaml")
        assert cfg.distillation_mode == "off"
        assert cfg.drift_weight == pytest.approx(0.1)
        assert cfg.restoration_factor == pytest.approx(0.99)
        assert cfg.divergence_threshold == pytest.approx(0.15)
        assert cfg.divergence_weight == pytest.approx(0.05)
