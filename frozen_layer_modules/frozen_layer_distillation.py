"""
LayerFreezer: stride-based transformer layer freezing with forward-hook activation
collection for CKA distillation.

Frozen layers:   layer_idx % stride == 0
Trainable adjacent (CKA target):  layer_idx % stride == 1 (i.e. frozen_idx + 1)

Embeddings and the LM head are excluded — only the main transformer block stack
is subject to alternating freeze logic.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_norm_param(name: str) -> bool:
    """Return True for LayerNorm / RMSNorm parameters (should never be frozen)."""
    return "norm" in name.lower()


# ---------------------------------------------------------------------------
# Layer discovery (identical attribute walk to before)
# ---------------------------------------------------------------------------

def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """
    Walk common HuggingFace CausalLM attribute paths to find the transformer
    decoder layer list.  Raises ValueError for unrecognised architectures.
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

    arch = getattr(getattr(model, "config", None), "architectures", ["Unknown"])
    raise ValueError(
        f"Cannot locate transformer layer list "
        f"(checked {container_attrs} × {layer_attrs}). "
        f"Architecture: {arch}."
    )


# ---------------------------------------------------------------------------
# LayerFreezer
# ---------------------------------------------------------------------------

class LayerFreezer:
    """
    Manages stride-based layer freezing and forward-hook activation collection.

    Args:
        model:  The model being fine-tuned.
        stride: Freeze every ``stride``-th layer (default 2 → even indices).
    """

    def __init__(self, model: nn.Module, stride: int = 2) -> None:
        self.model = model
        self.stride = stride
        self.layers: nn.ModuleList = get_transformer_layers(model)
        self.n_layers: int = len(self.layers)
        self.frozen_indices: Set[int] = set()
        # Tracks exactly which parameters we froze per layer so that release
        # never accidentally touches base-model weights (already frozen by LoRA)
        # or LayerNorm parameters (kept trainable throughout).
        self._our_frozen: Dict[int, List[nn.Parameter]] = {}
        self._apply_initial_freeze()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _apply_initial_freeze(self) -> None:
        """Freeze every stride-th layer at initialisation."""
        for i in range(self.n_layers):
            if i % self.stride == 0:
                self._freeze_layer(i)
        logger.info(
            "LayerFreezer: %d/%d transformer layers frozen (stride=%d, indices=%s).",
            len(self.frozen_indices), self.n_layers, self.stride,
            sorted(self.frozen_indices),
        )

    def _freeze_layer(self, idx: int) -> None:
        """
        Freeze non-norm, currently-trainable parameters in layer ``idx``.

        LayerNorm / RMSNorm parameters are intentionally skipped: freezing them
        while adjacent layers train causes normalisation mismatch that undoes
        their unfreezing benefit later.

        Only parameters that are *currently* trainable are recorded — this
        avoids touching base-model weights that LoRA has already frozen.
        """
        frozen = []
        for name, p in self.layers[idx].named_parameters():
            if _is_norm_param(name):
                continue
            if p.requires_grad:
                p.requires_grad = False
                frozen.append(p)
        self._our_frozen[idx] = frozen
        self.frozen_indices.add(idx)

    def release_layers(
        self,
        indices,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """
        Unfreeze the given layer indices and, if an optimizer is supplied,
        add the newly-trainable parameters as a new param group so they
        receive gradient updates immediately.

        Args:
            indices:   Iterable of layer indices to release.
            optimizer: Live optimizer instance.  If provided, newly released
                       parameters are appended as an extra param group with
                       the same LR as param_groups[0].
        """
        newly_trainable: List[nn.Parameter] = []
        for idx in list(indices):
            if idx not in self.frozen_indices:
                continue
            for p in self._our_frozen.pop(idx, []):
                p.requires_grad = True
                newly_trainable.append(p)
            self.frozen_indices.discard(idx)
            logger.debug("LayerFreezer: layer %d released.", idx)

        if newly_trainable and optimizer is not None:
            base_lr = optimizer.param_groups[0]["lr"]
            optimizer.add_param_group({"params": newly_trainable, "lr": base_lr})
            logger.info(
                "LayerFreezer: %d params added to optimizer (lr=%.2e).",
                len(newly_trainable), base_lr,
            )

    def release_all(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Release all remaining frozen layers."""
        self.release_layers(list(self.frozen_indices), optimizer)

    # ------------------------------------------------------------------
    # Forward-hook activation collection
    # ------------------------------------------------------------------

    def register_cka_hooks(
        self,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List]:
        """
        Register forward hooks to collect activations for CKA computation.

        Frozen layers: activations are detached (no gradient).
        Trainable adjacent layers (frozen_idx + 1): activations keep grad.

        Returns:
            frozen_acts:    Dict mapping frozen layer index → activation tensor.
            trainable_acts: Dict mapping trainable layer index → activation tensor.
            hooks:          List of hook handles — call handle.remove() when done.

        Usage::

            frozen_acts, trainable_acts, hooks = freezer.register_cka_hooks()
            try:
                outputs = model(**inputs)
            finally:
                for h in hooks:
                    h.remove()
        """
        frozen_acts: Dict[int, torch.Tensor] = {}
        trainable_acts: Dict[int, torch.Tensor] = {}
        hooks: List = []

        for i in range(self.n_layers):
            if i in self.frozen_indices:
                # Frozen layer — detach output
                def _frozen_hook(m, inp, out, idx=i):
                    act = out[0] if isinstance(out, tuple) else out
                    frozen_acts[idx] = act.detach()

                hooks.append(self.layers[i].register_forward_hook(_frozen_hook))

            elif (i - 1) in self.frozen_indices:
                # Immediately adjacent trainable layer — keep gradient
                def _trainable_hook(m, inp, out, idx=i):
                    act = out[0] if isinstance(out, tuple) else out
                    trainable_acts[idx] = act

                hooks.append(self.layers[i].register_forward_hook(_trainable_hook))

        return frozen_acts, trainable_acts, hooks
