"""
SlowDriftTrainer: distillation-aware training loop that combines:
  - drift_penalty   : L2 distance of fine-tune parameters from their initial values
  - divergence_penalty: KL divergence between fine-tune and base model logits
Post-epoch: partially restores frozen layer parameters toward their initial values
            using the configured restoration_factor.
"""
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_layer_distillation import AlternatingLayerFreezer

logger = logging.getLogger(__name__)


class SlowDriftTrainer:
    """
    Distillation trainer with frozen-layer drift regularisation.

    Args:
        model:      Fine-tuning model (gradients enabled on non-frozen layers).
        base_model: Reference model (kept frozen in inference mode).
        config:     Dict with optional keys:
                      drift_weight         (default 0.1)
                      restoration_factor   (default 0.99)
                      divergence_threshold (default 0.15)
                      divergence_weight    (default 0.05)

    Raises:
        ValueError: if base_model is None.
        ValueError: if model is not a recognisable transformer architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        base_model: Optional[nn.Module],
        config: Dict[str, Any],
    ):
        if base_model is None:
            raise ValueError(
                "Distillation mode requires base_model parameter. "
                "Provide it to SlowDriftTrainer()."
            )

        # Architecture check – raises ValueError for non-transformers
        self.layer_freezer = AlternatingLayerFreezer(model)

        self.model = model
        self.base_model = base_model
        self.config = config

        self.drift_weight = float(config.get("drift_weight", 0.1))
        self.restoration_factor = float(config.get("restoration_factor", 0.99))
        self.divergence_threshold = float(config.get("divergence_threshold", 0.15))
        self.divergence_weight = float(config.get("divergence_weight", 0.05))

        # Freeze base model entirely
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        # Snapshot of fine-tune model parameters at initialisation time
        # (used for drift computation and post-epoch restoration)
        self._base_snapshot: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        # Start with even layers frozen
        self.layer_freezer.freeze_even_layers()

        logger.info(
            "SlowDriftTrainer initialised | "
            f"drift_weight={self.drift_weight} | "
            f"restoration_factor={self.restoration_factor} | "
            f"divergence_threshold={self.divergence_threshold} | "
            f"divergence_weight={self.divergence_weight} | "
            f"frozen_layers={sorted(self.layer_freezer.get_frozen_indices())}"
        )

    # ------------------------------------------------------------------
    # Penalty computations
    # ------------------------------------------------------------------

    def compute_drift_penalty(self) -> torch.Tensor:
        """
        Mean L2 distance of current model parameters from the initial snapshot,
        weighted by drift_weight.
        """
        device = next(self.model.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0
        for name, param in self.model.named_parameters():
            if name in self._base_snapshot:
                base = self._base_snapshot[name].to(device)
                total = total + (param - base).pow(2).mean()
                count += 1
        return self.drift_weight * (total / max(count, 1))

    def compute_divergence_penalty(
        self,
        logits: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL-divergence penalty applied only when the divergence exceeds
        divergence_threshold, weighted by divergence_weight.

        Args:
            logits:      Fine-tune model logits  (batch, seq, vocab).
            base_logits: Base model logits        (batch, seq, vocab).

        Returns:
            Scalar penalty tensor (≥ 0).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        base_probs = F.softmax(base_logits.detach(), dim=-1)
        kl = F.kl_div(log_probs, base_probs, reduction="batchmean")
        penalty = F.relu(kl - self.divergence_threshold)
        return self.divergence_weight * penalty

    # ------------------------------------------------------------------
    # Post-epoch restoration
    # ------------------------------------------------------------------

    def restore_frozen_layers(self) -> None:
        """
        Linearly interpolate frozen layer parameters back toward their initial
        snapshot values:  p ← p * restoration_factor + p0 * (1 - restoration_factor)
        """
        device = next(self.model.parameters()).device
        frozen = self.layer_freezer.get_frozen_indices()
        layers = self.layer_freezer.layers

        for i in frozen:
            for name, param in layers[i].named_parameters():
                # Match by suffix against the full parameter name in the snapshot
                for snap_key, snap_val in self._base_snapshot.items():
                    if snap_key.endswith(name):
                        target = snap_val.to(device)
                        with torch.no_grad():
                            param.data.mul_(self.restoration_factor).add_(
                                target * (1.0 - self.restoration_factor)
                            )
                        break

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """
        Run one full training epoch over the provided dataloader.

        Args:
            dataloader: Iterable yielding dicts of tensors (model inputs).
            optimizer:  Optional pre-built optimizer.  If None, a default
                        AdamW is constructed over trainable parameters.

        Returns:
            Dict with averaged metrics:
              loss, drift_penalty, divergence_penalty, steps
        """
        if optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=2e-5)

        self.model.train()
        total_loss = total_drift = total_div = 0.0
        steps = 0

        for batch in dataloader:
            optimizer.zero_grad()

            device = next(self.model.parameters()).device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = self.model(**inputs)
            with torch.no_grad():
                base_outputs = self.base_model(**inputs)

            standard_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            drift = self.compute_drift_penalty()
            div = self.compute_divergence_penalty(outputs.logits, base_outputs.logits)

            loss = standard_loss + drift + div
            loss.backward()
            optimizer.step()

            total_loss += standard_loss.item()
            total_drift += drift.item()
            total_div += div.item()
            steps += 1

        # Post-epoch restoration of frozen layer weights
        self.restore_frozen_layers()
        self._log_epoch_metrics(steps, total_loss, total_drift, total_div)

        n = max(steps, 1)
        return {
            "loss": total_loss / n,
            "drift_penalty": total_drift / n,
            "divergence_penalty": total_div / n,
            "steps": steps,
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_epoch_metrics(
        self,
        steps: int,
        total_loss: float,
        total_drift: float,
        total_div: float,
    ) -> None:
        n = max(steps, 1)
        logger.info(
            f"Epoch complete | steps={steps} | "
            f"loss={total_loss/n:.4f} | "
            f"drift={total_drift/n:.4f} | "
            f"divergence={total_div/n:.4f}"
        )
        self._log_layer_drift_norms()

    def _log_layer_drift_norms(self) -> None:
        """Log per-frozen-layer parameter drift norms for diagnostics."""
        device = next(self.model.parameters()).device
        for i in sorted(self.layer_freezer.get_frozen_indices()):
            layer = self.layer_freezer.layers[i]
            norms = []
            for name, param in layer.named_parameters():
                for snap_key, snap_val in self._base_snapshot.items():
                    if snap_key.endswith(name):
                        norms.append((param - snap_val.to(device)).norm().item())
                        break
            if norms:
                logger.info(
                    f"  Layer {i} drift norms: "
                    f"{[f'{v:.4f}' for v in norms]}"
                )
