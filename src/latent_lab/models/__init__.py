from __future__ import annotations

from typing import Any

from torch import nn

from latent_lab.data.datasets import DatasetSpec
from latent_lab.models.ae import Autoencoder
from latent_lab.models.vae import VariationalAutoencoder
from latent_lab.models.vqvae import VectorQuantizedAutoencoder


def build_model(config: dict[str, Any], dataset: DatasetSpec) -> nn.Module:
    kind = str(config["kind"]).lower()
    if kind == "ae":
        return Autoencoder(
            input_shape=dataset.input_shape,
            latent_dim=int(config["latent_dim"]),
            hidden_dims=[int(value) for value in config.get("hidden_dims", [])],
            activation=str(config.get("activation", "relu")),
            output_activation_name=str(
                config.get("output_activation", "sigmoid")
            ),
        )
    if kind == "vae":
        return VariationalAutoencoder(
            input_shape=dataset.input_shape,
            latent_dim=int(config["latent_dim"]),
            hidden_dims=[int(value) for value in config.get("hidden_dims", [])],
            activation=str(config.get("activation", "relu")),
            output_activation_name=str(
                config.get("output_activation", "sigmoid")
            ),
        )
    if kind == "vqvae":
        return VectorQuantizedAutoencoder(
            channels=dataset.channels,
            hidden_channels=int(config.get("hidden_channels", 64)),
            embedding_dim=int(config["embedding_dim"]),
            codebook_size=int(config["codebook_size"]),
            commitment_weight=float(config.get("commitment_weight", 0.25)),
        )
    raise ValueError(f"Unsupported model kind: {kind}")


__all__ = [
    "Autoencoder",
    "VariationalAutoencoder",
    "VectorQuantizedAutoencoder",
    "build_model",
]
