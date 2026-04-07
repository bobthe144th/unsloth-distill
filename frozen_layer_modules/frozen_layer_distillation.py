"""
AlternatingLayerFreezer: freezes transformer layers alternately and provides
constraint loss computation between a fine-tuning model and a reference model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AlternatingLayerFreezer:
    """
    Manages freezing / unfreezing of transformer layers in an alternating pattern.

    Supports:
      - freeze_even_layers() / freeze_odd_layers() for alternating patterns
      - freeze_layers(indices) for arbitrary subsets
      - unfreeze_all() to restore gradient flow
      - compute_constraint_loss() for L2 constraint between two model outputs
    """

    def __init__(self, model: nn.Module):
        layers = self._get_transformer_layers(model)
        self.model = model
        self.layers = layers
        self.n_layers = len(layers)
        self._frozen_indices: set = set()

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _get_transformer_layers(model: nn.Module):
        """
        Walk common attribute names to find the transformer layer list.
        Raises ValueError when no recognisable structure is found.
        """
        container_attrs = ("model", "transformer", "encoder", "decoder")
        layer_attrs = ("layers", "h", "blocks")

        for c_attr in container_attrs:
            sub = getattr(model, c_attr, None)
            if sub is None:
                continue
            for l_attr in layer_attrs:
                candidate = getattr(sub, l_attr, None)
                if candidate is not None and len(candidate) > 0:
                    return candidate

        raise ValueError(
            f"Model must be a Transformer with an accessible layer list "
            f"(checked attrs: {container_attrs} × {layer_attrs}). "
            f"Got: {type(model).__name__}. Distillation mode disabled."
        )

    # ------------------------------------------------------------------
    # Freeze / unfreeze API
    # ------------------------------------------------------------------

    def freeze_layers(self, indices: List[int]) -> None:
        """Freeze the given layer indices; all others remain trainable."""
        self._frozen_indices = set(indices)
        for i, layer in enumerate(self.layers):
            trainable = i not in self._frozen_indices
            for p in layer.parameters():
                p.requires_grad = trainable

    def freeze_even_layers(self) -> None:
        """Freeze layers 0, 2, 4, …"""
        self.freeze_layers([i for i in range(self.n_layers) if i % 2 == 0])

    def freeze_odd_layers(self) -> None:
        """Freeze layers 1, 3, 5, …"""
        self.freeze_layers([i for i in range(self.n_layers) if i % 2 != 0])

    def unfreeze_all(self) -> None:
        """Restore gradient flow to every layer."""
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad = True
        self._frozen_indices = set()

    def get_frozen_indices(self) -> set:
        """Return a copy of the currently frozen layer indices."""
        return set(self._frozen_indices)

    # ------------------------------------------------------------------
    # Loss helper
    # ------------------------------------------------------------------

    def compute_constraint_loss(
        self,
        model_outputs: torch.Tensor,
        base_outputs: torch.Tensor,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Weighted MSE loss between fine-tune model outputs and frozen base outputs.

        Args:
            model_outputs: Tensor from the fine-tuning model.
            base_outputs:  Tensor from the reference (base) model (detached).
            weight:        Scalar multiplier for the constraint.

        Returns:
            Scalar constraint loss tensor.
        """
        return weight * F.mse_loss(model_outputs, base_outputs.detach())
