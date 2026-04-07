"""
Linear Centered Kernel Alignment (CKA) loss for representation-anchored distillation.

Formula (linear kernel):
    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    HSIC(X, Y) = (1 / (n-1)^2) * tr(K_Xc K_Yc)
                 where K_Xc = HX(HX)^T  (H = centering matrix)

Efficient identity for symmetric kernel matrices:
    tr(A B) = <A, B>_F  (Frobenius inner product, when both symmetric)

So:
    CKA(X, Y) = <K_Xc, K_Yc>_F / (||K_Xc||_F * ||K_Yc||_F + eps)

This is the cosine similarity of the centred kernel matrices, bounded in [0, 1].
"""
import torch
import torch.nn.functional as F


def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute linear CKA between two activation matrices.

    Args:
        X: Tensor of shape [n, d_x] — activations from the frozen layer (detached).
        Y: Tensor of shape [n, d_y] — activations from the adjacent trainable layer.
        eps: Small constant added to denominator for numerical stability.

    Returns:
        Scalar tensor in [0, 1].  Returns 0.0 if n <= 1 (HSIC undefined).

    Raises:
        ValueError: if X and Y have different batch sizes.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same batch size; got {X.shape[0]} and {Y.shape[0]}."
        )

    n = X.shape[0]
    if n <= 1:
        # HSIC requires at least 2 samples
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    # --- Center each representation (subtract column means) ---
    # H X  = X - mean(X, dim=0)  →  this is equivalent to H K H
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # --- Centred linear kernels [n, n] ---
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # --- Frobenius inner products (use element-wise product + sum for symmetric matrices) ---
    # tr(K_X K_Y) = <K_X, K_Y>_F  when both are symmetric
    inner_xy = (K_X * K_Y).sum()
    norm_x = (K_X * K_X).sum().sqrt()
    norm_y = (K_Y * K_Y).sum().sqrt()

    cka = inner_xy / (norm_x * norm_y + eps)
    return cka.clamp(0.0, 1.0)


def cka_penalty(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalty = 1 - CKA(X, Y), clamped to [0, 1].

    A penalty of 0 means the representations are perfectly aligned;
    a penalty of 1 means they are completely unrelated.
    """
    return (1.0 - linear_cka(X, Y, eps=eps)).clamp(0.0, 1.0)
