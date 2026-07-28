from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(self, encoder_latent: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if encoder_latent.ndim != 4:
            raise ValueError("VectorQuantizer expects BCHW encoder latents")

        flat = encoder_latent.permute(0, 2, 3, 1).contiguous()
        flat = flat.view(-1, self.embedding_dim)
        weights = self.codebook.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + weights.square().sum(dim=1)
            - 2 * flat @ weights.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(
            encoder_latent.shape[0],
            encoder_latent.shape[2],
            encoder_latent.shape[3],
            self.embedding_dim,
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        codebook_loss = functional.mse_loss(quantized, encoder_latent.detach())
        commitment_loss = functional.mse_loss(encoder_latent, quantized.detach())
        straight_through = encoder_latent + (quantized - encoder_latent).detach()

        counts = torch.bincount(
            indices, minlength=self.codebook.num_embeddings
        ).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(
            -(probabilities * torch.log(probabilities + 1e-10)).sum()
        )

        return straight_through, {
            "indices": indices.view(
                encoder_latent.shape[0],
                encoder_latent.shape[2],
                encoder_latent.shape[3],
            ),
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "weighted_commitment_loss": self.commitment_weight * commitment_loss,
            "codebook_perplexity": perplexity,
            "codes_used": (counts > 0).sum(),
        }

