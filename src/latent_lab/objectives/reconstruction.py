from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor


def reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    kind: str,
    reduction: str = "mean",
) -> Tensor:
    if kind == "mse":
        elementwise = functional.mse_loss(reconstruction, target, reduction="none")
    elif kind == "bce":
        elementwise = functional.binary_cross_entropy(
            reconstruction, target, reduction="none"
        )
    else:
        raise ValueError(f"Unsupported reconstruction loss: {kind}")

    if reduction == "mean":
        return elementwise.mean()
    if reduction == "sum_per_sample":
        return elementwise.flatten(start_dim=1).sum(dim=1).mean()
    raise ValueError(f"Unsupported reconstruction reduction: {reduction}")
