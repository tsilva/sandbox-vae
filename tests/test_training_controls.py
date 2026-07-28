import torch

from latent_lab.training.trainer import _effective_beta, corrupt_inputs


def test_gaussian_corruption_preserves_shape_and_range() -> None:
    torch.manual_seed(0)
    inputs = torch.full((2, 1, 8, 8), 0.5)
    corrupted = corrupt_inputs(
        inputs,
        {
            "kind": "gaussian",
            "standard_deviation": 0.3,
            "clamp": True,
        },
    )
    assert corrupted.shape == inputs.shape
    assert not torch.equal(corrupted, inputs)
    assert corrupted.min() >= 0
    assert corrupted.max() <= 1


def test_kl_warmup_reaches_target_beta() -> None:
    config = {"kind": "vae", "beta": 2.0, "kl_warmup_epochs": 4}
    assert _effective_beta(config, 1) == 0.5
    assert _effective_beta(config, 2) == 1.0
    assert _effective_beta(config, 4) == 2.0
    assert _effective_beta(config, 10) == 2.0
