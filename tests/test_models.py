import torch

from latent_lab.models.ae import Autoencoder
from latent_lab.models.vae import VariationalAutoencoder
from latent_lab.models.vqvae import VectorQuantizedAutoencoder


def test_ae_preserves_image_shape() -> None:
    model = Autoencoder((1, 28, 28), latent_dim=8, hidden_dims=[32])
    output = model(torch.rand(4, 1, 28, 28))
    assert output.reconstruction.shape == (4, 1, 28, 28)
    assert output.latent.shape == (4, 8)


def test_vae_preserves_image_shape_and_exposes_distribution() -> None:
    model = VariationalAutoencoder((1, 28, 28), 8, [32])
    output = model(torch.rand(4, 1, 28, 28))
    assert output.reconstruction.shape == (4, 1, 28, 28)
    assert output.extras["mu"].shape == (4, 8)
    assert output.extras["logvar"].shape == (4, 8)


def test_vqvae_preserves_image_shape() -> None:
    model = VectorQuantizedAutoencoder(
        channels=1,
        hidden_channels=16,
        embedding_dim=8,
        codebook_size=16,
        commitment_weight=0.25,
    )
    output = model(torch.rand(4, 1, 28, 28))
    assert output.reconstruction.shape == (4, 1, 28, 28)
    assert output.extras["indices"].shape == (4, 7, 7)

