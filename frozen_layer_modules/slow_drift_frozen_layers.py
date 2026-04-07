"""
SlowDriftTrainer: extends UnslothTrainer with CKA representation-anchored distillation.

All behaviour is gated behind the DISTILLATION flag in DistillationConfig.
When DISTILLATION=False the trainer is byte-for-byte identical to UnslothTrainer.

Three standard HuggingFace hooks carry the distillation logic:

  compute_loss()    — adds CKA penalty (λ · mean(1 - CKA(frozen, adjacent)))
  training_step()   — updates phase-based unfreezing schedule each step
  (no epoch callback needed — no weight snapshots required)

No base model is required: CKA is computed between adjacent frozen and trainable
layers of the *same* model, anchoring the trainable layers' representations to
those of the frozen layers.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from unsloth.trainer import UnslothTrainer

from .cka_loss import cka_penalty
from .config import DistillationConfig
from .frozen_layer_distillation import LayerFreezer

logger = logging.getLogger(__name__)


class SlowDriftTrainer(UnslothTrainer):
    """
    CKA-distillation trainer built on UnslothTrainer.

    All Unsloth speedups (Triton kernels, Q-GaLore, embedding LR, sample
    packing, padding-free) are fully inherited.

    Args:
        distillation_config: DistillationConfig (or plain dict) controlling
                             all distillation behaviour.  When
                             distillation_config.distillation is False
                             (the default), this class is a no-op wrapper
                             around UnslothTrainer.
        *args / **kwargs:    Forwarded verbatim to UnslothTrainer / SFTTrainer.

    Raises:
        ValueError: if distillation=True and the model has no accessible
                    transformer layer list.
    """

    def __init__(
        self,
        distillation_config: Optional[Union[DistillationConfig, Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Normalise config
        if distillation_config is None:
            distillation_config = DistillationConfig()
        elif isinstance(distillation_config, dict):
            from .config import load_config
            distillation_config = load_config(overrides=distillation_config)

        self._cfg: DistillationConfig = distillation_config

        if not self._cfg.distillation:
            # No-op: set attributes to safe defaults and return
            self._layer_freezer: Optional[LayerFreezer] = None
            self._current_lambda: float = 0.0
            logger.debug("SlowDriftTrainer: distillation disabled — no-op mode.")
            return

        # --- Distillation active ---
        self._layer_freezer = LayerFreezer(self.model, self._cfg.frozen_layer_stride)
        self._current_lambda = self._cfg.cka_lambda

        logger.info(
            "SlowDriftTrainer: CKA distillation active | "
            "lambda=%.4f | phase_unfreeze=%s | stride=%d | "
            "frozen_layers=%s",
            self._cfg.cka_lambda,
            self._cfg.phase_unfreeze,
            self._cfg.frozen_layer_stride,
            sorted(self._layer_freezer.frozen_indices),
        )

    # ------------------------------------------------------------------
    # HuggingFace hook: compute_loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        CE loss + λ · CKA_penalty when distillation is active.
        Pure pass-through to UnslothTrainer otherwise.
        """
        if not self._cfg.distillation or self._current_lambda == 0.0:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        # Register hooks to capture activations this forward pass
        frozen_acts, trainable_acts, hooks = self._layer_freezer.register_cka_hooks()
        try:
            outputs = model(**inputs)
            ce_loss = outputs.loss
        finally:
            for h in hooks:
                h.remove()

        cka_loss = self._compute_cka_penalty(frozen_acts, trainable_acts)
        total_loss = ce_loss + self._current_lambda * cka_loss

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "compute_loss | ce=%.4f cka=%.4f lambda=%.4f total=%.4f",
                ce_loss.item(), cka_loss.item(),
                self._current_lambda, total_loss.item(),
            )

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # HuggingFace hook: training_step
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Update phase-based unfreezing schedule before each step.
        No-op when distillation or phase_unfreeze is disabled.
        """
        if self._cfg.distillation and self._cfg.phase_unfreeze:
            self._update_phase()
        return super().training_step(model, inputs, *args, **kwargs)

    # ------------------------------------------------------------------
    # CKA penalty aggregation
    # ------------------------------------------------------------------

    def _compute_cka_penalty(
        self,
        frozen_acts: Dict[int, torch.Tensor],
        trainable_acts: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Mean CKA penalty across all adjacent frozen/trainable pairs.

        Each activation is mean-pooled over the sequence dimension:
            [B, T, H] → [B, H]
        before computing CKA, keeping the kernel matrices B×B.

        Returns a scalar clamped to [0, 1].
        """
        device = next(self.model.parameters()).device
        penalties: List[torch.Tensor] = []

        for frozen_idx, frozen_act in frozen_acts.items():
            trainable_idx = frozen_idx + 1
            if trainable_idx not in trainable_acts:
                continue

            trainable_act = trainable_acts[trainable_idx]
            batch_size = frozen_act.shape[0]

            if batch_size <= 1:
                # HSIC undefined for n=1 — skip this pair
                continue

            # Mean-pool over sequence dimension → [B, H]
            X = frozen_act.mean(dim=1) if frozen_act.dim() == 3 else frozen_act
            Y = trainable_act.mean(dim=1) if trainable_act.dim() == 3 else trainable_act

            # Ensure same device
            X = X.to(device)
            Y = Y.to(device)

            penalties.append(cka_penalty(X, Y))

        if not penalties:
            return torch.tensor(0.0, device=device)

        return torch.stack(penalties).mean().clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Phase-based unfreezing
    # ------------------------------------------------------------------

    def _update_phase(self) -> None:
        """
        Called once per training step.  Computes current progress fraction p,
        releases layers (deepest first) and decays λ according to the schedule.

        Schedule:
          p < start              → all frozen, λ = CKA_LAMBDA
          start ≤ p ≤ end        → linear release + linear λ decay
          p > end                → all layers trainable, λ = 0
        """
        state = self.state
        if state.max_steps <= 0:
            return

        p = state.global_step / state.max_steps
        start = self._cfg.phase_unfreeze_start
        end = self._cfg.phase_unfreeze_end

        if p < start:
            self._current_lambda = self._cfg.cka_lambda
            return

        if p > end:
            if self._layer_freezer.frozen_indices:
                self._layer_freezer.release_all(self.optimizer)
                logger.info("SlowDriftTrainer: all layers released (p=%.3f).", p)
            self._current_lambda = 0.0
            return

        # Within the unfreezing window
        progress = (p - start) / (end - start)  # [0, 1]
        self._current_lambda = self._cfg.cka_lambda * (1.0 - progress)

        # Determine how many layers to release (deepest first)
        all_frozen_sorted = sorted(self._layer_freezer.frozen_indices, reverse=True)
        if not all_frozen_sorted:
            return

        # Total number of layers that *started* frozen — use the initial count
        # (frozen_indices shrinks as we release, so track using stride)
        n_initially_frozen = len(
            [i for i in range(self._layer_freezer.n_layers)
             if i % self._cfg.frozen_layer_stride == 0]
        )
        num_to_release = int(progress * n_initially_frozen)
        n_already_released = n_initially_frozen - len(all_frozen_sorted)
        n_new = max(0, num_to_release - n_already_released)

        if n_new > 0:
            to_release = all_frozen_sorted[:n_new]
            self._layer_freezer.release_layers(to_release, self.optimizer)
            logger.info(
                "SlowDriftTrainer: released layers %s (p=%.3f, lambda=%.4f).",
                sorted(to_release), p, self._current_lambda,
            )
