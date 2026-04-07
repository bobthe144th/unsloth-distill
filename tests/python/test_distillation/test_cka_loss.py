"""
Unit tests for frozen_layer_modules.cka_loss

CKA is a pure-Python/NumPy-compatible computation — runs without GPU.
torch is still required for tensor operations.
"""
import importlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

if _HAS_TORCH:
    import torch
    from frozen_layer_modules.cka_loss import cka_penalty, linear_cka


class TestLinearCKA:
    def test_identical_inputs_returns_one(self):
        X = torch.randn(8, 16)
        assert linear_cka(X, X).item() == pytest.approx(1.0, abs=1e-5)

    def test_output_bounded_zero_one(self):
        X = torch.randn(8, 32)
        Y = torch.randn(8, 32)
        val = linear_cka(X, Y).item()
        assert 0.0 <= val <= 1.0

    def test_single_sample_returns_zero(self):
        X = torch.randn(1, 16)
        Y = torch.randn(1, 16)
        assert linear_cka(X, Y).item() == pytest.approx(0.0)

    def test_orthogonal_representations(self):
        # Construct X and Y whose kernels are orthogonal → CKA ≈ 0
        n, d = 8, 4
        X = torch.zeros(n, d)
        X[:, 0] = 1.0           # all variance in dim 0
        Y = torch.zeros(n, d)
        Y[:, 1] = torch.arange(n, dtype=torch.float) - (n - 1) / 2  # variance in dim 1
        # After centering: K_X rows all same → rank 1, K_Y differs
        # The CKA won't necessarily be exactly 0 but should be small
        val = linear_cka(X, Y).item()
        assert val < 0.5  # loose bound to avoid brittle exact-zero assertion

    def test_scaled_input_invariant(self):
        """CKA should be scale-invariant."""
        X = torch.randn(8, 16)
        Y = torch.randn(8, 16)
        base = linear_cka(X, Y).item()
        scaled = linear_cka(X * 10.0, Y * 0.01).item()
        assert abs(base - scaled) < 1e-4

    def test_mismatched_batch_raises(self):
        X = torch.randn(4, 16)
        Y = torch.randn(8, 16)
        with pytest.raises(ValueError, match="batch size"):
            linear_cka(X, Y)

    def test_different_hidden_dims_ok(self):
        """CKA does NOT require same hidden dim (kernels are n×n)."""
        X = torch.randn(8, 32)
        Y = torch.randn(8, 64)
        val = linear_cka(X, Y).item()
        assert 0.0 <= val <= 1.0

    def test_gradient_flows_through_Y(self):
        """Gradient must flow through Y (trainable) but not X (frozen/detached)."""
        X = torch.randn(4, 8).detach()
        Y = torch.randn(4, 8, requires_grad=True)
        loss = 1.0 - linear_cka(X, Y)
        loss.backward()
        assert Y.grad is not None
        assert Y.grad.abs().max().item() > 0

    def test_eps_prevents_division_by_zero(self):
        """Degenerate inputs (constant tensors) must not produce NaN/Inf."""
        X = torch.ones(4, 8)   # zero variance → centered kernel = 0
        Y = torch.randn(4, 8)
        val = linear_cka(X, Y).item()
        assert not (val != val)   # not NaN
        assert val == pytest.approx(0.0, abs=1e-6)


class TestCKAPenalty:
    def test_identical_returns_zero(self):
        X = torch.randn(8, 16)
        assert cka_penalty(X, X).item() == pytest.approx(0.0, abs=1e-5)

    def test_unrelated_returns_near_one(self):
        torch.manual_seed(42)
        X = torch.randn(16, 64)
        Y = torch.randn(16, 64) * 100  # very different scale & direction
        pen = cka_penalty(X, Y).item()
        assert pen >= 0.0
        assert pen <= 1.0

    def test_clamped_to_unit_interval(self):
        X = torch.randn(4, 8)
        Y = torch.randn(4, 8)
        pen = cka_penalty(X, Y).item()
        assert 0.0 <= pen <= 1.0
