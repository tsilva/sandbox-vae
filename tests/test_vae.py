import torch

from latent_lab.models.vae import reparameterize
from latent_lab.objectives.vae import kl_divergence


def test_standard_normal_has_zero_kl() -> None:
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    assert torch.equal(kl_divergence(mu, logvar), torch.tensor(0.0))


def test_reparameterization_passes_gradients() -> None:
    torch.manual_seed(0)
    mu = torch.zeros(4, 8, requires_grad=True)
    logvar = torch.zeros(4, 8, requires_grad=True)
    reparameterize(mu, logvar).sum().backward()
    assert mu.grad is not None
    assert logvar.grad is not None
    assert torch.count_nonzero(logvar.grad) > 0

