from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _channels_for_variant(variant: str) -> int:
    return 16 if variant == "weak" else 32


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, image_size: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        steps = int(math.log2(image_size)) - 2
        channels = base_channels
        layers = []
        current = in_channels
        for _ in range(steps):
            layers.append(nn.Conv2d(current, channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current = channels
            channels = min(channels * 2, 256)
        self.net = nn.Sequential(*layers)
        self.out_channels = current
        self.fc = nn.Linear(current * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.fc(features.flatten(start_dim=1))


class VariationalHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(x), self.logvar(x)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int, image_size: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        steps = int(math.log2(image_size)) - 2
        channel_stack = []
        current = base_channels
        for _ in range(steps - 1):
            channel_stack.append(current)
            current = min(current * 2, 256)
        bottleneck_channels = current
        self.fc = nn.Linear(latent_dim, bottleneck_channels * 4 * 4)
        layers = []
        current = bottleneck_channels
        for out_ch in reversed(channel_stack):
            layers.append(nn.ConvTranspose2d(current, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current = out_ch
        layers.append(nn.ConvTranspose2d(current, out_channels, kernel_size=4, stride=2, padding=1))
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], -1, 4, 4)
        return self.net(x)


@dataclass
class ModelOutput:
    recon: torch.Tensor
    recon_logits: torch.Tensor
    latent: torch.Tensor
    mu: torch.Tensor | None
    logvar: torch.Tensor | None


class Autoencoder(nn.Module):
    def __init__(self, in_channels: int, image_size: int, latent_dim: int, decoder_variant: str = "standard"):
        super().__init__()
        width = _channels_for_variant(decoder_variant)
        self.encoder = ConvEncoder(in_channels, image_size, latent_dim, base_channels=width)
        self.decoder = ConvDecoder(in_channels, image_size, latent_dim, base_channels=width)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z))

    def forward(self, x: torch.Tensor) -> ModelOutput:
        latent = self.encoder(x)
        logits = self.decoder(latent)
        recon = torch.sigmoid(logits)
        return ModelOutput(recon=recon, recon_logits=logits, latent=latent, mu=None, logvar=None)


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels: int, image_size: int, latent_dim: int, decoder_variant: str = "standard"):
        super().__init__()
        width = _channels_for_variant(decoder_variant)
        self.encoder = ConvEncoder(in_channels, image_size, latent_dim, base_channels=width)
        self.head = VariationalHead(latent_dim, latent_dim)
        self.decoder = ConvDecoder(in_channels, image_size, latent_dim, base_channels=width)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(x)
        return mu

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z))

    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(count, self.head.mu.out_features, device=device)
        return self.decode_latent(z)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        features = self.encoder(x)
        mu, logvar = self.head(features)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        recon = torch.sigmoid(logits)
        return ModelOutput(recon=recon, recon_logits=logits, latent=z, mu=mu, logvar=logvar)
