from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(loss_type: str, recon: torch.Tensor, recon_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if loss_type == "bernoulli":
        loss = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(recon, target, reduction="none")
    elif loss_type == "mse":
        loss = F.mse_loss(recon, target, reduction="none")
    elif loss_type == "mixed":
        loss = 0.5 * F.l1_loss(recon, target, reduction="none") + 0.5 * F.mse_loss(recon, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss.flatten(start_dim=1).mean(dim=1)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def apply_free_bits(kl_dims: torch.Tensor, free_bits: float) -> torch.Tensor:
    if free_bits <= 0.0:
        return kl_dims.sum(dim=1)
    return torch.clamp(kl_dims, min=free_bits).sum(dim=1)
