from __future__ import annotations

from torch import Tensor, nn

from latent_lab.models.outputs import LatentModelOutput
from latent_lab.models.quantizer import VectorQuantizer


class VectorQuantizedAutoencoder(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        embedding_dim: int,
        codebook_size: int,
        commitment_weight: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 3, padding=1),
        )
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_weight=commitment_weight,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels, channels, 4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> LatentModelOutput:
        encoder_latent = self.encoder(inputs)
        quantized, extras = self.quantizer(encoder_latent)
        extras["encoder_latent"] = encoder_latent
        extras["quantized_latent"] = quantized
        return LatentModelOutput(
            reconstruction=self.decoder(quantized),
            latent=quantized,
            extras=extras,
        )

