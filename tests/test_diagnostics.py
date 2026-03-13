import torch

from sandbox_autoencoders.training.diagnostics import summarize_latents


def test_latent_diagnostics_handle_near_zero_kl():
    mu = torch.zeros(8, 4)
    logvar = torch.zeros(8, 4)
    latent = torch.zeros(8, 4)
    stats = summarize_latents(mu, logvar, latent)
    assert stats["active_latents"] == 0
    assert all(abs(value) < 1e-8 for value in stats["kl_per_dim"])


def test_latent_diagnostics_single_active_dimension():
    mu = torch.zeros(16, 4)
    mu[:, 0] = torch.linspace(-2, 2, 16)
    logvar = torch.zeros(16, 4)
    latent = mu.clone()
    stats = summarize_latents(mu, logvar, latent)
    assert stats["active_latents"] == 1
    assert stats["kl_per_dim"][0] > 0.0
