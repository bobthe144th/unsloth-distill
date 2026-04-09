"""
GuessDPOTrainer: DPO with online judge self-play.

Training loop per batch
-----------------------
  1. Standard DPO forward pass → L_dpo (scalar or per-example depending on mode).
  2. Judge step (explicit mode):
       a. Decode prompt + chosen + rejected texts from the tokenised batch.
       b. Randomly assign chosen/rejected to A/B slots (prevents position bias).
       c. Tokenise a judge prompt and run a single forward pass.
       d. Extract log π(A) and log π(B) at the last token position.
       e. Greedy guess = argmax(log π(A), log π(B)).
       f. Reward  r = +1 if correct, −1 if wrong.
       g. Advantage = r − mean(r)  (baseline reduces variance, no bias).
       h. REINFORCE  L_rl = −mean(advantage · log π(guessed_token)).
  2'. Judge step (implicit mode, no extra forward pass):
       Compute reward_margin = log π(chosen) − log π(rejected).
       Correct if margin > 0.  Upweight only (no separate RL loss).
  3. Total loss:
       explicit: L = w · L_dpo + rl_coeff · L_rl
       implicit: L = w · L_dpo
       where w = 2 for wrong-guess pairs, 1 for correct (if hardness_weighting).

The REINFORCE signal directly trains the model's judgment capability.
The hardness weighting automatically focuses DPO gradient on hard examples.
Both effects compound: better responses → better judgments → cleaner signal.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from trl import DPOTrainer as _DPOTrainerBase
except ImportError:  # pragma: no cover
    _DPOTrainerBase = object  # type: ignore[assignment,misc]

from .config import GuessDPOConfig

logger = logging.getLogger(__name__)


class GuessDPOTrainer(_DPOTrainerBase):  # type: ignore[misc]
    """
    DPO trainer with an explicit online judge step.

    Args:
        guess_config: GuessDPOConfig controlling the judge step.
        *args / **kwargs: Forwarded verbatim to DPOTrainer.

    When guess_config.rl_coeff == 0 and guess_config.hardness_weighting is
    False the trainer is identical to DPOTrainer.
    """

    def __init__(
        self,
        guess_config: Optional[GuessDPOConfig] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._gcfg = guess_config or GuessDPOConfig()

        # Cached token ids for "A" and "B".
        self._a_tok: Optional[int] = None
        self._b_tok: Optional[int] = None

        logger.info(
            "GuessDPOTrainer: beta=%.3f | rl_coeff=%.3f | hardness_weighting=%s | "
            "implicit_mode=%s | judge_max_length=%d",
            self._gcfg.beta,
            self._gcfg.rl_coeff,
            self._gcfg.hardness_weighting,
            self._gcfg.implicit_mode,
            self._gcfg.judge_max_length,
        )

    # ------------------------------------------------------------------
    # Token-id resolution
    # ------------------------------------------------------------------

    def _ab_token_ids(self) -> Tuple[int, int]:
        """
        Return (a_id, b_id) — the last token of " A" and " B".

        Using a leading space handles tokenisers that represent capital letters
        differently in mid-sentence vs start-of-sentence positions.
        """
        if self._a_tok is None:
            self._a_tok = self.tokenizer.encode(" A", add_special_tokens=False)[-1]
            self._b_tok = self.tokenizer.encode(" B", add_special_tokens=False)[-1]
        return self._a_tok, self._b_tok  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Judge input construction
    # ------------------------------------------------------------------

    def _decode_prompt(self, inputs: Dict[str, torch.Tensor]) -> list[str]:
        """
        Extract prompt text from the tokenised batch.

        Supports two DPOTrainer variants:
          - TRL >= 0.9 with explicit 'prompt_input_ids'
          - Older TRL where the prompt is the prefix of chosen_input_ids with
            chosen_labels == -100.
        """
        if "prompt_input_ids" in inputs:
            return self.tokenizer.batch_decode(
                inputs["prompt_input_ids"], skip_special_tokens=True
            )

        # Fallback: walk chosen_input_ids up to the first labelled token.
        prompts = []
        for ids, labels in zip(inputs["chosen_input_ids"], inputs["chosen_labels"]):
            non_ignore = (labels != -100).nonzero(as_tuple=True)[0]
            end = int(non_ignore[0]) if len(non_ignore) else len(ids)
            prompts.append(
                self.tokenizer.decode(ids[:end], skip_special_tokens=True)
            )
        return prompts

    def _decode_response(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[str]:
        """Decode only the response portion (where labels != -100)."""
        texts = []
        for ids, lbl in zip(input_ids, labels):
            mask = lbl != -100
            texts.append(
                self.tokenizer.decode(ids[mask], skip_special_tokens=True)
            )
        return texts

    def _build_judge_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Construct judge prompts for the batch.

        Returns
        -------
        judge_inputs : tokenised dict ready for model(**judge_inputs)
        correct_is_a : BoolTensor[B] — True when A slot holds the chosen response.
        """
        device = inputs["chosen_input_ids"].device
        prompts = self._decode_prompt(inputs)
        chosen_texts = self._decode_response(
            inputs["chosen_input_ids"], inputs["chosen_labels"]
        )
        rejected_texts = self._decode_response(
            inputs["rejected_input_ids"], inputs["rejected_labels"]
        )

        batch_size = len(prompts)
        # Randomly swap A/B to prevent the model from learning a position bias.
        swap = torch.rand(batch_size) > 0.5  # swap[i]=True → A=rejected, B=chosen

        judge_texts = []
        for i in range(batch_size):
            a = rejected_texts[i] if swap[i] else chosen_texts[i]
            b = chosen_texts[i] if swap[i] else rejected_texts[i]
            judge_texts.append(
                self._gcfg.judge_template.format(
                    prompt=prompts[i], a=a, b=b
                )
            )

        judge_inputs = self.tokenizer(
            judge_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._gcfg.judge_max_length,
        ).to(device)

        # correct_is_a[i] = True when A holds the chosen (preferred) response.
        # Not swapped → A=chosen → correct is A.
        correct_is_a = ~swap
        return judge_inputs, correct_is_a.to(device)

    # ------------------------------------------------------------------
    # Explicit judge RL loss
    # ------------------------------------------------------------------

    def _explicit_judge_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the judge forward pass and compute a REINFORCE loss.

        Returns
        -------
        rl_loss  : scalar Tensor (gradient flows through log π(action))
        correct  : BoolTensor[B] — True for pairs the model guessed right
        """
        judge_inputs, correct_is_a = self._build_judge_inputs(inputs)
        a_id, b_id = self._ab_token_ids()
        device = correct_is_a.device

        # Forward pass — gradients needed for REINFORCE.
        logits = model(**judge_inputs).logits[:, -1, :]  # [B, vocab]

        # Normalise between A and B only.
        ab_logits = logits[:, [a_id, b_id]]  # [B, 2]
        ab_log_probs = F.log_softmax(ab_logits, dim=-1)  # [B, 2]

        # Greedy guess (argmax is non-differentiable; gradient via log_p_action).
        # Index 0 → guessed A, index 1 → guessed B.
        with torch.no_grad():
            action = ab_logits.argmax(dim=-1)  # [B]
            # correct_idx: 0 if A is correct, 1 if B is correct.
            correct_idx = (~correct_is_a).long()
            reward = (action == correct_idx).float() * 2.0 - 1.0  # {+1, −1}
            baseline = reward.mean()
            advantage = reward - baseline

        # log π(guessed_token) — this is where gradients flow.
        log_p_action = ab_log_probs[
            torch.arange(len(action), device=device), action
        ]
        rl_loss = -(advantage * log_p_action).mean()

        correct = action == correct_idx
        return rl_loss, correct

    # ------------------------------------------------------------------
    # Implicit judge (no extra forward pass)
    # ------------------------------------------------------------------

    def _implicit_correctness(
        self,
        inputs: Dict[str, torch.Tensor],
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implicit "guess": the model prefers chosen iff its reward margin > 0.

        reward_margin = (log π(chosen) − log π_ref(chosen))
                      − (log π(rejected) − log π_ref(rejected))

        Returns BoolTensor[B] — True when the model "guesses" correctly.
        """
        margin = (
            (policy_chosen_logps - ref_chosen_logps)
            - (policy_rejected_logps - ref_rejected_logps)
        )
        return margin > 0

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        train_eval: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Extend the parent's DPO metrics with the judge RL signal.

        Explicit mode adds:
            metrics["judge_accuracy"]  — fraction of pairs guessed correctly
            metrics["rl_loss"]         — REINFORCE term magnitude

        Both modes add:
            metrics["judge_wrong_frac"] — fraction of pairs receiving 2× weight
        """
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        # Only apply judge logic during training.
        if train_eval != "train":
            return loss, metrics

        judge_active = self._gcfg.rl_coeff > 0 or self._gcfg.hardness_weighting
        if not judge_active:
            return loss, metrics

        if self._gcfg.implicit_mode:
            loss, metrics = self._apply_implicit_judge(loss, batch, metrics)
        else:
            loss, metrics = self._apply_explicit_judge(loss, model, batch, metrics)

        return loss, metrics

    def _apply_explicit_judge(
        self,
        dpo_loss: torch.Tensor,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        rl_loss, correct = self._explicit_judge_loss(model, batch)

        accuracy = correct.float().mean().item()
        wrong_frac = 1.0 - accuracy
        metrics["judge_accuracy"] = accuracy
        metrics["judge_wrong_frac"] = wrong_frac
        metrics["rl_loss"] = rl_loss.item()

        total = dpo_loss
        if self._gcfg.hardness_weighting:
            # Scale the whole-batch DPO loss up by the fraction of hard pairs.
            # Equivalent to averaging 2×loss for wrong and 1×loss for correct.
            hardness_scale = 1.0 + wrong_frac
            total = dpo_loss * hardness_scale
        if self._gcfg.rl_coeff > 0:
            total = total + self._gcfg.rl_coeff * rl_loss

        logger.debug(
            "GuessDPO | dpo=%.4f rl=%.4f accuracy=%.3f",
            dpo_loss.item(), rl_loss.item(), accuracy,
        )
        return total, metrics

    def _apply_implicit_judge(
        self,
        dpo_loss: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Implicit mode: no extra forward pass.

        The model's log-prob ratio (relative to the reference) determines
        whether it "guessed" correctly.  We can only apply hardness weighting
        here — there is no separate RL loss to add.
        """
        # We need per-example log-probs.  These live in the batch only if the
        # parent pre-computed them (TRL >= 0.9 with precompute_ref_log_probs).
        # If not available, skip silently.
        needed = {
            "chosen_logps", "rejected_logps",
            "reference_chosen_logps", "reference_rejected_logps",
        }
        if not needed.issubset(batch.keys()):
            logger.debug(
                "GuessDPO implicit mode: per-example log-probs not in batch, "
                "skipping hardness weighting."
            )
            return dpo_loss, metrics

        correct = self._implicit_correctness(
            batch,
            batch["chosen_logps"],
            batch["rejected_logps"],
            batch["reference_chosen_logps"],
            batch["reference_rejected_logps"],
        )
        wrong_frac = (~correct).float().mean().item()
        metrics["judge_accuracy"] = 1.0 - wrong_frac
        metrics["judge_wrong_frac"] = wrong_frac

        if self._gcfg.hardness_weighting:
            hardness_scale = 1.0 + wrong_frac
            dpo_loss = dpo_loss * hardness_scale

        return dpo_loss, metrics
