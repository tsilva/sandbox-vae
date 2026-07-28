from __future__ import annotations

import torch
from torch import Tensor

from latent_lab.models.outputs import LatentModelOutput
from latent_lab.objectives.reconstruction import reconstruction_loss


def kl_divergence_per_dimension(mu: Tensor, logvar: Tensor) -> Tensor:
    per_element = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    return per_element.mean(dim=0)


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return kl_divergence_per_dimension(mu, logvar).sum()


def variational_objective(
    output: LatentModelOutput,
    target: Tensor,
    reconstruction_kind: str,
    beta: float,
    reconstruction_reduction: str = "sum_per_sample",
    free_bits: float = 0.0,
) -> dict[str, Tensor]:
    reconstruction = reconstruction_loss(
        output.reconstruction,
        target,
        reconstruction_kind,
        reconstruction_reduction,
    )
    per_dimension_kl = kl_divergence_per_dimension(
        output.extras["mu"], output.extras["logvar"]
    )
    kl = per_dimension_kl.sum()
    effective_kl = per_dimension_kl.clamp_min(free_bits).sum()
    beta_tensor = torch.as_tensor(beta, device=kl.device)
    weighted_kl = beta_tensor * effective_kl
    return {
        "loss": reconstruction + weighted_kl,
        "reconstruction_loss": reconstruction,
        "kl_loss": kl,
        "effective_kl_loss": effective_kl,
        "weighted_kl_loss": weighted_kl,
        "beta": beta_tensor,
    }
