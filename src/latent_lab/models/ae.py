from __future__ import annotations

import torch
from torch import Tensor, nn

from latent_lab.models.components import flattened_size, mlp, output_activation
from latent_lab.models.outputs import LatentModelOutput


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        output_activation_name: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        input_dim = flattened_size(input_shape)
        self.encoder = mlp(
            [input_dim, *hidden_dims, latent_dim],
            activation,
        )
        self.decoder = mlp(
            [latent_dim, *reversed(hidden_dims), input_dim],
            activation,
            final_activation=output_activation(output_activation_name),
        )

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(torch.flatten(inputs, start_dim=1))

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).reshape(latent.shape[0], *self.input_shape)

    def forward(self, inputs: Tensor) -> LatentModelOutput:
        latent = self.encode(inputs)
        return LatentModelOutput(
            reconstruction=self.decode(latent),
            latent=latent,
        )
