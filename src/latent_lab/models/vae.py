from __future__ import annotations

import torch
from torch import Tensor, nn

from latent_lab.models.components import flattened_size, mlp, output_activation
from latent_lab.models.outputs import LatentModelOutput


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    standard_deviation = torch.exp(0.5 * logvar)
    noise = torch.randn_like(standard_deviation)
    return mu + standard_deviation * noise


class VariationalAutoencoder(nn.Module):
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
        final_hidden = hidden_dims[-1] if hidden_dims else input_dim
        self.encoder = mlp([input_dim, *hidden_dims], activation)
        self.mu = nn.Linear(final_hidden, latent_dim)
        self.logvar = nn.Linear(final_hidden, latent_dim)
        self.decoder = mlp(
            [latent_dim, *reversed(hidden_dims), input_dim],
            activation,
            final_activation=output_activation(output_activation_name),
        )

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(torch.flatten(inputs, start_dim=1))
        return self.mu(features), self.logvar(features)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).reshape(latent.shape[0], *self.input_shape)

    def forward(self, inputs: Tensor) -> LatentModelOutput:
        mu, logvar = self.encode(inputs)
        latent = reparameterize(mu, logvar) if self.training else mu
        return LatentModelOutput(
            reconstruction=self.decode(latent),
            latent=latent,
            extras={"mu": mu, "logvar": logvar},
        )
