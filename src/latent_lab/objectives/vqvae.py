from __future__ import annotations

from torch import Tensor

from latent_lab.models.outputs import LatentModelOutput
from latent_lab.objectives.reconstruction import reconstruction_loss


def vector_quantized_objective(
    output: LatentModelOutput,
    target: Tensor,
    reconstruction_kind: str,
    reconstruction_reduction: str = "mean",
) -> dict[str, Tensor]:
    reconstruction = reconstruction_loss(
        output.reconstruction,
        target,
        reconstruction_kind,
        reconstruction_reduction,
    )
    codebook = output.extras["codebook_loss"]
    commitment = output.extras["weighted_commitment_loss"]
    return {
        "loss": reconstruction + codebook + commitment,
        "reconstruction_loss": reconstruction,
        "codebook_loss": codebook,
        "commitment_loss": output.extras["commitment_loss"],
        "weighted_commitment_loss": commitment,
        "codebook_perplexity": output.extras["codebook_perplexity"],
        "codes_used": output.extras["codes_used"],
    }
