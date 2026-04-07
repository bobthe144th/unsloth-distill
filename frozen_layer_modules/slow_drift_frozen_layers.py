"""
SlowDriftTrainer: extends UnslothTrainer with frozen-layer distillation.

Inheriting from UnslothTrainer (→ SFTTrainer) preserves all Unsloth speedups:
  - Custom Triton kernels (already patched into the model at load time)
  - Q-GaLore / embedding-LR optimizer via UnslothTrainingArguments
  - Sample packing and padding-free batching
  - Flash Attention 2

Distillation additions (injected via HuggingFace's standard hooks):
  - compute_loss()    : adds drift_penalty + divergence_penalty to CE loss
  - training_step()   : calls AlternatingLayerFreezer.step() before each step
  - DistillationCallback : snapshots params at epoch start, restores at epoch end
"""
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import TrainingArguments as HFTrainingArguments

from unsloth.trainer import UnslothTrainer
from .frozen_layer_distillation import AlternatingLayerFreezer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Epoch-level callback: snapshot → restore
# ---------------------------------------------------------------------------

class _DistillationCallback(TrainerCallback):
    """Snapshots model params at epoch start; restores frozen layers at epoch end."""

    def __init__(self, distillation_trainer: "SlowDriftTrainer") -> None:
        self._dt = distillation_trainer

    def on_epoch_begin(
        self,
        args: HFTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        model = kwargs.get("model", self._dt.model)
        self._dt._epoch_snapshot = {
            name: param.detach().clone().cpu()
            for name, param in model.named_parameters()
        }
        logger.debug("DistillationCallback: epoch snapshot captured.")
        return control

    def on_epoch_end(
        self,
        args: HFTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        model = kwargs.get("model", self._dt.model)
        self._dt._restore_frozen_layers(model)
        self._dt._log_layer_drift_norms(model)
        return control


# ---------------------------------------------------------------------------
# SlowDriftTrainer
# ---------------------------------------------------------------------------

class SlowDriftTrainer(UnslothTrainer):
    """
    Distillation-aware trainer built on top of UnslothTrainer.

    All Unsloth speed-ups (Q-GaLore, embedding LR, sample packing, padding-free,
    Triton kernels) are inherited automatically.  Three distillation-specific
    behaviours are added via standard HuggingFace hooks:

      1. compute_loss()   — appends drift_penalty + divergence_penalty to CE loss.
      2. training_step()  — calls AlternatingLayerFreezer.step() before each step.
      3. callback         — snapshots params at epoch start; restores at epoch end.

    Args:
        base_model: Frozen reference model (kept in eval mode, no gradients).
                    Required; raises ValueError when None.
        distillation_config: Dict with optional distillation hyperparameters:
            drift_weight         (default 0.1)
            restoration_factor   (default 0.99)
            divergence_threshold (default 0.15)
            divergence_weight    (default 0.05)
        *args / **kwargs: Forwarded verbatim to UnslothTrainer / SFTTrainer.

    Raises:
        ValueError: if base_model is None.
        ValueError: if model has no accessible transformer layer list
                    (detected by AlternatingLayerFreezer).
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        distillation_config: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        if base_model is None:
            raise ValueError(
                "Distillation mode requires base_model parameter. "
                "Provide it to SlowDriftTrainer()."
            )

        cfg = distillation_config or {}
        self._base_model = base_model
        self._drift_weight = float(cfg.get("drift_weight", 0.1))
        self._restoration_factor = float(cfg.get("restoration_factor", 0.99))
        self._divergence_threshold = float(cfg.get("divergence_threshold", 0.15))
        self._divergence_weight = float(cfg.get("divergence_weight", 0.05))

        # UnslothTrainer.__init__ sets self.model
        super().__init__(*args, **kwargs)

        # Freeze base model entirely
        for p in self._base_model.parameters():
            p.requires_grad = False
        self._base_model.eval()

        # Architecture validation + alternating layer freezer
        # (raises ValueError for non-transformer models)
        self._layer_freezer = AlternatingLayerFreezer(self.model)

        # Initial snapshot (updated at each epoch start by the callback)
        self._epoch_snapshot: Dict[str, torch.Tensor] = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
        }

        # Register the distillation callback
        self.add_callback(_DistillationCallback(self))

        logger.info(
            "SlowDriftTrainer ready | "
            f"drift_weight={self._drift_weight} | "
            f"restoration_factor={self._restoration_factor} | "
            f"divergence_threshold={self._divergence_threshold} | "
            f"divergence_weight={self._divergence_weight} | "
            f"frozen_layers={sorted(self._layer_freezer.get_frozen_indices())}"
        )

    # ------------------------------------------------------------------
    # HuggingFace hook: called once per batch
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Standard CE loss + drift_penalty + divergence_penalty.

        Plugs into HF Trainer's gradient accumulation, mixed-precision,
        and gradient checkpointing as-is.
        """
        outputs = model(**inputs)
        ce_loss = outputs.loss

        drift = self._compute_drift_penalty(model)
        div = self._compute_divergence_penalty(outputs.logits, inputs)

        total_loss = ce_loss + drift + div

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "compute_loss | ce=%.4f drift=%.4f div=%.4f total=%.4f",
                ce_loss.item(), drift.item(), div.item(), total_loss.item(),
            )

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # HuggingFace hook: called once per optimizer step
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Advance the AlternatingLayerFreezer before each step."""
        self._layer_freezer.step()
        return super().training_step(model, inputs, *args, **kwargs)

    # ------------------------------------------------------------------
    # Penalty computations
    # ------------------------------------------------------------------

    def _compute_drift_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Mean L2 distance of currently-frozen params from their epoch-start snapshot,
        weighted by drift_weight.
        """
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad and name in self._epoch_snapshot:
                snap = self._epoch_snapshot[name].to(device)
                total = total + (param - snap).pow(2).mean()
                count += 1
        return self._drift_weight * (total / max(count, 1))

    def _compute_divergence_penalty(
        self,
        logits: torch.Tensor,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        KL-divergence penalty between fine-tune and base model output distributions.
        Only applied at positions where the base model is confident
        (max softmax probability > divergence_threshold).
        """
        device = logits.device
        fwd_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
        base_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
            if k in fwd_keys
        }
        base_inputs.setdefault("use_cache", False)

        self._base_model.to(device)
        with torch.no_grad():
            base_out = self._base_model(**base_inputs)
        base_logits = base_out.logits

        # Align sequence dimension (shift for next-token prediction)
        seq = min(logits.shape[1], base_logits.shape[1])
        ft = logits[:, :seq, :]
        base = base_logits[:, :seq, :]

        p_base = F.softmax(base, dim=-1)
        log_p_ft = F.log_softmax(ft, dim=-1)

        # Confidence mask
        mask = p_base.max(dim=-1).values > self._divergence_threshold  # [B, T]
        if not mask.any():
            return torch.tensor(0.0, device=device)

        kl_per_pos = F.kl_div(log_p_ft, p_base, reduction="none").sum(dim=-1)
        return self._divergence_weight * kl_per_pos[mask].mean()

    # ------------------------------------------------------------------
    # Post-epoch restoration (called by _DistillationCallback)
    # ------------------------------------------------------------------

    def _restore_frozen_layers(self, model: nn.Module) -> None:
        """
        Interpolate frozen layer params toward their epoch-start snapshot:
            p ← restoration_factor * p0 + (1 - restoration_factor) * p
        """
        device = next(model.parameters()).device
        rf = self._restoration_factor
        frozen_indices = self._layer_freezer.get_frozen_indices()
        layers = self._layer_freezer.layers

        with torch.no_grad():
            for i in frozen_indices:
                for name, param in layers[i].named_parameters():
                    for snap_key, snap_val in self._epoch_snapshot.items():
                        if snap_key.endswith(name):
                            p0 = snap_val.to(device)
                            param.data.mul_(1.0 - rf).add_(p0 * rf)
                            break

    # ------------------------------------------------------------------
    # Diagnostics (called by _DistillationCallback)
    # ------------------------------------------------------------------

    def _log_layer_drift_norms(self, model: nn.Module) -> None:
        """Log per-frozen-layer drift norms for monitoring."""
        device = next(model.parameters()).device
        for i in sorted(self._layer_freezer.get_frozen_indices()):
            layer = self._layer_freezer.layers[i]
            norms = []
            for name, param in layer.named_parameters():
                for snap_key, snap_val in self._epoch_snapshot.items():
                    if snap_key.endswith(name):
                        norms.append((param - snap_val.to(device)).norm().item())
                        break
            if norms:
                logger.info(
                    "  Layer %d drift norms (post-restore): %s",
                    i, [f"{v:.4f}" for v in norms],
                )
