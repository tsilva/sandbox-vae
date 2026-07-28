from __future__ import annotations

import torch
from torch import Tensor

from latent_lab.models.outputs import LatentModelOutput
from latent_lab.objectives.reconstruction import reconstruction_loss


def autoencoder_objective(
    output: LatentModelOutput,
    target: Tensor,
    reconstruction_kind: str,
    reconstruction_reduction: str = "mean",
    latent_l1_weight: float = 0.0,
) -> dict[str, Tensor]:
    reconstruction = reconstruction_loss(
        output.reconstruction,
        target,
        reconstruction_kind,
        reconstruction_reduction,
    )
    latent_l1 = output.latent.abs().mean()
    weighted_latent_l1 = torch.as_tensor(
        latent_l1_weight, device=latent_l1.device
    ) * latent_l1
    return {
        "loss": reconstruction + weighted_latent_l1,
        "reconstruction_loss": reconstruction,
        "latent_l1": latent_l1,
        "weighted_latent_l1": weighted_latent_l1,
    }
