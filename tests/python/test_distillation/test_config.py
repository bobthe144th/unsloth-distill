"""
Unit tests for frozen_layer_modules.config

Covers all six flags, env-var parsing, YAML loading, and priority order.
"""
import os
import sys
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(monkeypatch, env=None, yaml_text=None, overrides=None, tmp_path=None):
    """Helper: set env vars, optionally write a YAML file, then load config."""
    # Clear all distillation env vars
    for var in ("DISTILLATION", "PHASE_UNFREEZE", "CKA_LAMBDA",
                "PHASE_UNFREEZE_START", "PHASE_UNFREEZE_END", "FROZEN_LAYER_STRIDE"):
        monkeypatch.delenv(var, raising=False)

    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))

    config_path = None
    if yaml_text and tmp_path:
        p = tmp_path / ".distillation_config.yaml"
        p.write_text(yaml_text)
        config_path = p

    from frozen_layer_modules.config import load_config
    return load_config(config_path=config_path, overrides=overrides)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_all_defaults(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, tmp_path=tmp_path)
        assert cfg.distillation is False
        assert cfg.phase_unfreeze is False
        assert cfg.cka_lambda == pytest.approx(0.1)
        assert cfg.phase_unfreeze_start == pytest.approx(0.3)
        assert cfg.phase_unfreeze_end == pytest.approx(0.7)
        assert cfg.frozen_layer_stride == 2


# ---------------------------------------------------------------------------
# Boolean flag parsing (DISTILLATION, PHASE_UNFREEZE)
# ---------------------------------------------------------------------------

class TestBoolParsing:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_truthy_values(self, monkeypatch, tmp_path, val):
        cfg = _load(monkeypatch, env={"DISTILLATION": val}, tmp_path=tmp_path)
        assert cfg.distillation is True

    @pytest.mark.parametrize("val", ["0", "false", "False", "FALSE", "no", "off"])
    def test_falsy_values(self, monkeypatch, tmp_path, val):
        cfg = _load(monkeypatch, env={"DISTILLATION": val}, tmp_path=tmp_path)
        assert cfg.distillation is False

    def test_invalid_bool_warns_and_defaults_false(self, monkeypatch, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _load(monkeypatch, env={"DISTILLATION": "maybe"}, tmp_path=tmp_path)
        assert cfg.distillation is False
        assert any("maybe" in str(w.message) for w in caught)

    def test_phase_unfreeze_parsed(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, env={"PHASE_UNFREEZE": "true"}, tmp_path=tmp_path)
        assert cfg.phase_unfreeze is True


# ---------------------------------------------------------------------------
# Float / int flag parsing
# ---------------------------------------------------------------------------

class TestNumericParsing:
    def test_cka_lambda_from_env(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, env={"CKA_LAMBDA": "0.5"}, tmp_path=tmp_path)
        assert cfg.cka_lambda == pytest.approx(0.5)

    def test_phase_unfreeze_start_from_env(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, env={"PHASE_UNFREEZE_START": "0.2"}, tmp_path=tmp_path)
        assert cfg.phase_unfreeze_start == pytest.approx(0.2)

    def test_phase_unfreeze_end_from_env(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, env={"PHASE_UNFREEZE_END": "0.8"}, tmp_path=tmp_path)
        assert cfg.phase_unfreeze_end == pytest.approx(0.8)

    def test_frozen_layer_stride_from_env(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, env={"FROZEN_LAYER_STRIDE": "3"}, tmp_path=tmp_path)
        assert cfg.frozen_layer_stride == 3

    def test_invalid_float_warns_and_uses_default(self, monkeypatch, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _load(monkeypatch, env={"CKA_LAMBDA": "abc"}, tmp_path=tmp_path)
        assert cfg.cka_lambda == pytest.approx(0.1)
        assert caught


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_yaml_loaded(self, monkeypatch, tmp_path):
        yaml = "distillation: true\ncka_lambda: 0.25\nfrozen_layer_stride: 4\n"
        cfg = _load(monkeypatch, yaml_text=yaml, tmp_path=tmp_path)
        assert cfg.distillation is True
        assert cfg.cka_lambda == pytest.approx(0.25)
        assert cfg.frozen_layer_stride == 4

    def test_yaml_missing_uses_defaults(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, tmp_path=tmp_path)
        assert cfg.distillation is False

    def test_yaml_bool_on_off(self, monkeypatch, tmp_path):
        # PyYAML parses bare 'on'/'off' as Python booleans
        yaml = "distillation: on\nphase_unfreeze: off\n"
        cfg = _load(monkeypatch, yaml_text=yaml, tmp_path=tmp_path)
        assert cfg.distillation is True
        assert cfg.phase_unfreeze is False


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------

class TestPriority:
    def test_env_overrides_yaml(self, monkeypatch, tmp_path):
        yaml = "cka_lambda: 0.9\n"
        cfg = _load(monkeypatch, env={"CKA_LAMBDA": "0.05"}, yaml_text=yaml, tmp_path=tmp_path)
        assert cfg.cka_lambda == pytest.approx(0.05)  # env wins

    def test_overrides_beat_env(self, monkeypatch, tmp_path):
        cfg = _load(monkeypatch, env={"CKA_LAMBDA": "0.5"},
                    overrides={"cka_lambda": 0.01}, tmp_path=tmp_path)
        assert cfg.cka_lambda == pytest.approx(0.01)  # override wins

    def test_yaml_beats_defaults(self, monkeypatch, tmp_path):
        yaml = "frozen_layer_stride: 3\n"
        cfg = _load(monkeypatch, yaml_text=yaml, tmp_path=tmp_path)
        assert cfg.frozen_layer_stride == 3
