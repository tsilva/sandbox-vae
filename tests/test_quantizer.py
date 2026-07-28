import torch

from latent_lab.models.quantizer import VectorQuantizer


def test_quantizer_selects_nearest_code() -> None:
    quantizer = VectorQuantizer(2, 2, commitment_weight=0.25)
    with torch.no_grad():
        quantizer.codebook.weight.copy_(
            torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        )
    latent = torch.tensor([[[[0.9]], [[0.8]]]])
    quantized, extras = quantizer(latent)
    assert extras["indices"].item() == 1
    assert torch.allclose(quantized, torch.ones_like(quantized))


def test_straight_through_and_codebook_gradients() -> None:
    quantizer = VectorQuantizer(4, 2, commitment_weight=0.25)
    latent = torch.randn(2, 2, 3, 3, requires_grad=True)
    quantized, extras = quantizer(latent)
    loss = (
        quantized.mean()
        + extras["codebook_loss"]
        + extras["weighted_commitment_loss"]
    )
    loss.backward()
    assert latent.grad is not None
    assert torch.count_nonzero(latent.grad) > 0
    assert quantizer.codebook.weight.grad is not None
    assert torch.count_nonzero(quantizer.codebook.weight.grad) > 0

