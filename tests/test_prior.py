import torch
import torch.nn.functional as functional

from latent_lab.models.prior import AutoregressiveCodePrior


def test_code_prior_logits_and_gradients() -> None:
    prior = AutoregressiveCodePrior(
        codebook_size=16,
        embedding_dim=8,
        hidden_dim=12,
    )
    tokens = torch.randint(0, 16, (4, 9))
    logits = prior(tokens)
    assert logits.shape == (4, 9, 16)
    loss = functional.cross_entropy(
        logits.reshape(-1, 16), tokens.reshape(-1)
    )
    loss.backward()
    assert prior.embedding.weight.grad is not None
    assert torch.count_nonzero(prior.embedding.weight.grad) > 0


def test_code_prior_sampling_range() -> None:
    prior = AutoregressiveCodePrior(
        codebook_size=8,
        embedding_dim=4,
        hidden_dim=6,
    )
    samples = prior.sample(3, 7, temperature=0.8)
    assert samples.shape == (3, 7)
    assert samples.min() >= 0
    assert samples.max() < 8
