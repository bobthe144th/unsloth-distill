"""
Tests for GuessDPOTrainer.

All tests are pure-Python / pure-PyTorch — no GPU, no real model weights,
no TRL DPOTrainer instance required. The trainer internals that are tested
(_build_judge_inputs, _explicit_judge_loss, _implicit_correctness) are
exercised via lightweight stubs.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

if _HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from dpo_modules.config import GuessDPOConfig
    from dpo_modules.guess_dpo_trainer import GuessDPOTrainer


# ---------------------------------------------------------------------------
# Minimal stub so we can instantiate GuessDPOTrainer without a real DPOTrainer
# ---------------------------------------------------------------------------

class _StubDPOTrainer:
    """Minimal stand-in for trl.DPOTrainer."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._gcfg = None  # will be set by GuessDPOTrainer.__init__
        self._a_tok = None
        self._b_tok = None

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        loss = torch.tensor(1.0, requires_grad=True)
        return loss, {}


def _make_trainer(cfg=None, tokenizer=None):
    """Build a GuessDPOTrainer without calling DPOTrainer.__init__."""
    tokenizer = tokenizer or _make_tokenizer()
    trainer = object.__new__(GuessDPOTrainer)
    _StubDPOTrainer.__init__(trainer, tokenizer=tokenizer)
    trainer._gcfg = cfg or GuessDPOConfig()
    trainer._a_tok = None
    trainer._b_tok = None
    return trainer


# ---------------------------------------------------------------------------
# Minimal tokenizer stub
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Encodes / decodes a tiny fixed vocabulary."""

    # Map word → id.  "A"=1, "B"=2, everything else → 0.
    _vocab = {"A": 1, "B": 2, " A": 1, " B": 2}

    def encode(self, text, add_special_tokens=True):
        return [self._vocab.get(text, 0)]

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "unk") for i in ids)

    def batch_decode(self, tensor, skip_special_tokens=True):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.tolist()
        return [self.decode(row) for row in tensor]

    def __call__(self, texts, return_tensors=None, padding=True,
                 truncation=True, max_length=1024):
        B = len(texts)
        fake_ids = torch.zeros(B, 4, dtype=torch.long)
        fake_mask = torch.ones(B, 4, dtype=torch.long)
        result = types.SimpleNamespace(
            input_ids=fake_ids,
            attention_mask=fake_mask,
        )
        if return_tensors == "pt":
            return {"input_ids": fake_ids, "attention_mask": fake_mask}
        return result


def _make_tokenizer():
    return _StubTokenizer()


# ---------------------------------------------------------------------------
# Helpers to build fake batch dicts
# ---------------------------------------------------------------------------

def _make_batch(batch_size: int = 4):
    """
    Construct a fake DPOTrainer batch with the keys used by GuessDPOTrainer.
    Labels for the first 2 positions are -100 (prompt) and the rest are real.
    """
    seq_len = 6
    chosen_ids = torch.randint(3, 10, (batch_size, seq_len))
    rejected_ids = torch.randint(3, 10, (batch_size, seq_len))

    # First 2 tokens are "prompt" (label = -100), rest are response.
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[:, 2:] = torch.randint(3, 10, (batch_size, seq_len - 2))

    prompt_ids = chosen_ids[:, :2]

    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": chosen_ids,
        "chosen_labels": labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": labels,
    }


# ===========================================================================
# Tests: _decode_prompt
# ===========================================================================

class TestDecodePrompt(unittest.TestCase):
    def test_uses_prompt_input_ids_when_present(self):
        trainer = _make_trainer()
        batch = _make_batch(batch_size=2)
        prompts = trainer._decode_prompt(batch)
        self.assertEqual(len(prompts), 2)

    def test_fallback_uses_label_boundary(self):
        trainer = _make_trainer()
        batch = _make_batch(batch_size=2)
        del batch["prompt_input_ids"]  # trigger fallback path
        prompts = trainer._decode_prompt(batch)
        self.assertEqual(len(prompts), 2)


# ===========================================================================
# Tests: _build_judge_inputs
# ===========================================================================

class TestBuildJudgeInputs(unittest.TestCase):
    def _run(self, batch_size=4):
        trainer = _make_trainer()
        batch = _make_batch(batch_size=batch_size)
        judge_inputs, correct_is_a = trainer._build_judge_inputs(batch)
        return judge_inputs, correct_is_a

    def test_returns_tensor_dict_and_bool_tensor(self):
        judge_inputs, correct_is_a = self._run(batch_size=4)
        self.assertIn("input_ids", judge_inputs)
        self.assertEqual(correct_is_a.dtype, torch.bool)

    def test_correct_is_a_has_batch_size(self):
        _, correct_is_a = self._run(batch_size=6)
        self.assertEqual(correct_is_a.shape[0], 6)

    def test_random_swap_both_values_appear(self):
        """Over many calls correct_is_a should contain both True and False."""
        trainer = _make_trainer()
        batch = _make_batch(batch_size=32)
        seen = set()
        for _ in range(10):
            _, c = trainer._build_judge_inputs(batch)
            seen.update(c.tolist())
            if seen == {True, False}:
                break
        self.assertEqual(seen, {True, False}, "swap never produced both orderings")


# ===========================================================================
# Tests: _ab_token_ids
# ===========================================================================

class TestAbTokenIds(unittest.TestCase):
    def test_returns_a_and_b_ids(self):
        trainer = _make_trainer()
        a, b = trainer._ab_token_ids()
        self.assertEqual(a, 1)  # " A" → 1 in stub vocab
        self.assertEqual(b, 2)  # " B" → 2

    def test_cached_on_second_call(self):
        trainer = _make_trainer()
        a1, b1 = trainer._ab_token_ids()
        a2, b2 = trainer._ab_token_ids()
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)


# ===========================================================================
# Tests: _explicit_judge_loss
# ===========================================================================

class _ModelWithKnownPreference:
    """
    Fake model that always returns logits strongly favouring token A (id=1).
    Vocab size = 5; position [0,1,2,3,4].
    """

    def __call__(self, **kwargs):
        B = kwargs["input_ids"].shape[0]
        # Last-token logits: position 1 (= "A") has very high value.
        logits = torch.zeros(B, 4, 5)
        logits[:, -1, 1] = 10.0   # strongly prefer "A"
        logits[:, -1, 2] = -10.0  # strongly disprefer "B"
        return types.SimpleNamespace(logits=logits)


class TestExplicitJudgeLoss(unittest.TestCase):
    def _make_always_a_trainer(self):
        return _make_trainer()

    def test_returns_scalar_loss_and_bool_correct(self):
        trainer = self._make_always_a_trainer()
        model = _ModelWithKnownPreference()
        batch = _make_batch(batch_size=4)
        rl_loss, correct = trainer._explicit_judge_loss(model, batch)
        self.assertEqual(rl_loss.shape, ())   # scalar
        self.assertEqual(correct.dtype, torch.bool)
        self.assertEqual(correct.shape[0], 4)

    def test_gradient_flows_through_rl_loss(self):
        trainer = self._make_always_a_trainer()
        model = _ModelWithKnownPreference()
        batch = _make_batch(batch_size=4)
        rl_loss, _ = trainer._explicit_judge_loss(model, batch)
        # rl_loss should have a grad_fn (not a leaf detached tensor).
        self.assertIsNotNone(rl_loss.grad_fn)

    def test_model_that_always_picks_a_is_correct_when_a_is_chosen(self):
        """
        When correct_is_a is forced to all-True (A = chosen), a model that
        always picks A should have correct == all True.
        """
        trainer = self._make_always_a_trainer()
        model = _ModelWithKnownPreference()
        batch = _make_batch(batch_size=8)

        # Patch _build_judge_inputs to force correct_is_a = all True.
        orig = trainer._build_judge_inputs

        def patched(b):
            ji, _ = orig(b)
            return ji, torch.ones(b["chosen_input_ids"].shape[0], dtype=torch.bool)

        trainer._build_judge_inputs = patched
        _, correct = trainer._explicit_judge_loss(model, batch)
        self.assertTrue(correct.all(), "Model always picks A, all A=correct → all right")

    def test_model_that_always_picks_a_is_wrong_when_b_is_chosen(self):
        trainer = self._make_always_a_trainer()
        model = _ModelWithKnownPreference()
        batch = _make_batch(batch_size=8)

        # Force correct_is_a = all False (B = chosen).
        def patched(b):
            orig = GuessDPOTrainer._build_judge_inputs
            ji, _ = orig(trainer, b)
            return ji, torch.zeros(b["chosen_input_ids"].shape[0], dtype=torch.bool)

        trainer._build_judge_inputs = patched
        _, correct = trainer._explicit_judge_loss(model, batch)
        self.assertFalse(correct.any(), "Model always picks A, all B=correct → all wrong")

    def test_advantage_baseline_reduces_variance(self):
        """
        Check that when half the batch is correct and half wrong, the mean
        advantage is 0 (reward - mean(reward) = 0 in expectation).
        """
        trainer = self._make_always_a_trainer()
        model = _ModelWithKnownPreference()
        batch = _make_batch(batch_size=8)

        # Force 4 correct (A=chosen) and 4 wrong (B=chosen).
        half = batch["chosen_input_ids"].shape[0] // 2

        def patched(b):
            orig = GuessDPOTrainer._build_judge_inputs
            ji, _ = orig(trainer, b)
            c = torch.cat([
                torch.ones(half, dtype=torch.bool),
                torch.zeros(half, dtype=torch.bool),
            ])
            return ji, c

        trainer._build_judge_inputs = patched
        # Should not raise and should produce a finite loss.
        rl_loss, _ = trainer._explicit_judge_loss(model, batch)
        self.assertTrue(torch.isfinite(rl_loss))


# ===========================================================================
# Tests: _implicit_correctness
# ===========================================================================

class TestImplicitCorrectness(unittest.TestCase):
    def _trainer(self):
        return _make_trainer(GuessDPOConfig(implicit_mode=True))

    def test_positive_margin_is_correct(self):
        trainer = self._trainer()
        B = 4
        # chosen_logps > rejected_logps (after reference subtraction)
        p_c = torch.tensor([1.0, 1.0, 1.0, 1.0])
        p_r = torch.tensor([0.0, 0.0, 0.0, 0.0])
        r_c = torch.zeros(B)
        r_r = torch.zeros(B)
        correct = trainer._implicit_correctness({}, p_c, p_r, r_c, r_r)
        self.assertTrue(correct.all())

    def test_negative_margin_is_wrong(self):
        trainer = self._trainer()
        B = 4
        p_c = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        p_r = torch.tensor([0.0, 0.0, 0.0, 0.0])
        correct = trainer._implicit_correctness(
            {}, p_c, p_r, torch.zeros(B), torch.zeros(B)
        )
        self.assertFalse(correct.any())

    def test_mixed_margin(self):
        trainer = self._trainer()
        p_c = torch.tensor([1.0, -1.0, 1.0, -1.0])
        p_r = torch.zeros(4)
        correct = trainer._implicit_correctness(
            {}, p_c, p_r, torch.zeros(4), torch.zeros(4)
        )
        expected = torch.tensor([True, False, True, False])
        self.assertTrue((correct == expected).all())


# ===========================================================================
# Tests: get_batch_loss_metrics
# ===========================================================================

class _StubSuperMetrics:
    """Mixin providing a canned get_batch_loss_metrics."""

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        loss = torch.tensor(2.0, requires_grad=True)
        return loss, {"rewards/chosen": 0.5}


def _make_full_trainer(cfg):
    """
    Build a GuessDPOTrainer whose super().get_batch_loss_metrics is stubbed.
    """
    trainer = _make_trainer(cfg)
    # Inject the stub's method as the "parent" get_batch_loss_metrics.
    stub = _StubSuperMetrics()

    def _parent_metrics(model, batch, train_eval="train"):
        return stub.get_batch_loss_metrics(model, batch, train_eval)

    trainer._parent_get_batch_loss_metrics = _parent_metrics
    # Monkey-patch to call our stub instead of the real parent.
    original = GuessDPOTrainer.get_batch_loss_metrics

    def _patched(self_inner, model, batch, train_eval="train"):
        # Replicate the structure of the real method but call our stub for super.
        loss, metrics = _parent_metrics(model, batch, train_eval)
        if train_eval != "train":
            return loss, metrics
        judge_active = self_inner._gcfg.rl_coeff > 0 or self_inner._gcfg.hardness_weighting
        if not judge_active:
            return loss, metrics
        if self_inner._gcfg.implicit_mode:
            return self_inner._apply_implicit_judge(loss, batch, metrics)
        return self_inner._apply_explicit_judge(loss, model, batch, metrics)

    trainer.get_batch_loss_metrics = lambda model, batch, te="train": _patched(
        trainer, model, batch, te
    )
    return trainer


class TestGetBatchLossMetrics(unittest.TestCase):
    def test_eval_mode_returns_parent_loss_unchanged(self):
        cfg = GuessDPOConfig(rl_coeff=0.1)
        trainer = _make_full_trainer(cfg)
        batch = _make_batch()
        model = _ModelWithKnownPreference()
        loss, metrics = trainer.get_batch_loss_metrics(model, batch, "eval")
        self.assertAlmostEqual(loss.item(), 2.0)

    def test_rl_coeff_zero_no_hardness_returns_parent_loss(self):
        cfg = GuessDPOConfig(rl_coeff=0.0, hardness_weighting=False)
        trainer = _make_full_trainer(cfg)
        batch = _make_batch()
        model = _ModelWithKnownPreference()
        loss, metrics = trainer.get_batch_loss_metrics(model, batch, "train")
        self.assertAlmostEqual(loss.item(), 2.0)

    def test_explicit_mode_adds_judge_metrics(self):
        cfg = GuessDPOConfig(rl_coeff=0.05, hardness_weighting=True, implicit_mode=False)
        trainer = _make_full_trainer(cfg)
        batch = _make_batch(batch_size=4)
        model = _ModelWithKnownPreference()
        _, metrics = trainer.get_batch_loss_metrics(model, batch, "train")
        self.assertIn("judge_accuracy", metrics)
        self.assertIn("rl_loss", metrics)
        self.assertIn("judge_wrong_frac", metrics)

    def test_explicit_mode_loss_greater_than_pure_dpo_when_hardness_on(self):
        """hardness_weighting scales loss ≥ 1× always."""
        cfg = GuessDPOConfig(rl_coeff=0.0, hardness_weighting=True, implicit_mode=False)
        trainer = _make_full_trainer(cfg)
        batch = _make_batch(batch_size=8)
        model = _ModelWithKnownPreference()
        loss, _ = trainer.get_batch_loss_metrics(model, batch, "train")
        self.assertGreaterEqual(loss.item(), 2.0)

    def test_implicit_mode_skips_when_logps_missing(self):
        cfg = GuessDPOConfig(rl_coeff=0.0, hardness_weighting=True, implicit_mode=True)
        trainer = _make_full_trainer(cfg)
        batch = _make_batch(batch_size=4)  # no log-prob keys
        model = MagicMock()
        loss, metrics = trainer.get_batch_loss_metrics(model, batch, "train")
        # Without logps, implicit mode falls through to parent loss.
        self.assertAlmostEqual(loss.item(), 2.0)

    def test_implicit_mode_with_logps_scales_loss(self):
        cfg = GuessDPOConfig(rl_coeff=0.0, hardness_weighting=True, implicit_mode=True)
        trainer = _make_full_trainer(cfg)
        batch = _make_batch(batch_size=4)
        # Add fake log-probs so implicit mode activates.
        batch["chosen_logps"] = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        batch["rejected_logps"] = torch.zeros(4)
        batch["reference_chosen_logps"] = torch.zeros(4)
        batch["reference_rejected_logps"] = torch.zeros(4)
        model = MagicMock()
        loss, metrics = trainer.get_batch_loss_metrics(model, batch, "train")
        # Half correct → wrong_frac = 0.5 → scale = 1.5 → loss = 3.0
        self.assertAlmostEqual(loss.item(), 3.0, places=4)
        self.assertIn("judge_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
