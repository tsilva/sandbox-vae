from __future__ import annotations

from typing import Any, Callable

from torch import Tensor

from latent_lab.models.outputs import LatentModelOutput
from latent_lab.objectives.ae import autoencoder_objective
from latent_lab.objectives.vae import variational_objective
from latent_lab.objectives.vqvae import vector_quantized_objective

Objective = Callable[[LatentModelOutput, Tensor], dict[str, Tensor]]


def build_objective(
    config: dict[str, Any],
    *,
    beta_override: float | None = None,
) -> Objective:
    kind = str(config["kind"]).lower()
    reconstruction = str(config.get("reconstruction", "mse"))
    reduction = str(
        config.get(
            "reconstruction_reduction",
            "sum_per_sample" if kind == "vae" else "mean",
        )
    )
    if kind == "ae":
        latent_l1_weight = float(config.get("latent_l1_weight", 0.0))
        return lambda output, target: autoencoder_objective(
            output,
            target,
            reconstruction,
            reduction,
            latent_l1_weight,
        )
    if kind == "vae":
        beta = (
            float(beta_override)
            if beta_override is not None
            else float(config.get("beta", 1.0))
        )
        free_bits = float(config.get("free_bits", 0.0))
        return lambda output, target: variational_objective(
            output,
            target,
            reconstruction,
            beta,
            reduction,
            free_bits,
        )
    if kind == "vqvae":
        return lambda output, target: vector_quantized_objective(
            output, target, reconstruction, reduction
        )
    raise ValueError(f"Unsupported objective kind: {kind}")


__all__ = ["Objective", "build_objective"]
