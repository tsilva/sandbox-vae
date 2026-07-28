from latent_lab.config import with_overrides


def test_dotted_overrides_do_not_mutate_baseline() -> None:
    baseline = {"model": {"latent_dim": 8}, "training": {"seed": 0}}
    changed = with_overrides(
        baseline, {"model.latent_dim": 32, "training.seed": 2}
    )
    assert baseline["model"]["latent_dim"] == 8
    assert changed["model"]["latent_dim"] == 32
    assert changed["training"]["seed"] == 2

