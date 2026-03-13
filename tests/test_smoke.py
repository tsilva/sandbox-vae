from pathlib import Path

import pytest

from sandbox_autoencoders.experiments import (
    exp_01_autoencoder_baseline,
    exp_02_vanilla_vae_baseline,
    exp_03_observation_model_ablation,
    exp_04_beta_and_warmup_sweep,
    exp_05_latent_capacity_sweep,
    exp_06_kl_stability_and_collapse,
    exp_07_disentanglement_and_traversals,
    exp_08_real_image_vae,
    exp_09_sampling_and_geometry,
    exp_10_temporal_vae_extension,
)


EXPERIMENTS = [
    exp_01_autoencoder_baseline.main,
    exp_02_vanilla_vae_baseline.main,
    exp_03_observation_model_ablation.main,
    exp_04_beta_and_warmup_sweep.main,
    exp_05_latent_capacity_sweep.main,
    exp_06_kl_stability_and_collapse.main,
    exp_07_disentanglement_and_traversals.main,
    exp_08_real_image_vae.main,
    exp_09_sampling_and_geometry.main,
    exp_10_temporal_vae_extension.main,
]


@pytest.mark.parametrize("entrypoint", EXPERIMENTS)
def test_every_experiment_smoke(entrypoint, tmp_path: Path):
    output_root = tmp_path / entrypoint.__module__.split(".")[-1]
    entrypoint(
        [
            "--output-dir",
            str(output_root),
            "--data-root",
            str(tmp_path / "data"),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--wandb-mode",
            "disabled",
        ]
    )
    experiment_dir = next(output_root.iterdir())
    for name in ["history.json", "summary.json", "notes.md", "recon_grid.png", "latent_stats.json"]:
        assert (experiment_dir / name).exists()


def test_wandb_offline_and_disabled_modes_complete(tmp_path: Path):
    disabled_root = tmp_path / "disabled"
    offline_root = tmp_path / "offline"
    exp_01_autoencoder_baseline.main(
        [
            "--output-dir",
            str(disabled_root),
            "--data-root",
            str(tmp_path / "data"),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--wandb-mode",
            "disabled",
        ]
    )
    exp_02_vanilla_vae_baseline.main(
        [
            "--output-dir",
            str(offline_root),
            "--data-root",
            str(tmp_path / "data"),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--wandb-mode",
            "offline",
        ]
    )
    for root in [disabled_root, offline_root]:
        experiment_dir = next(root.iterdir())
        for name in ["history.json", "summary.json", "notes.md", "recon_grid.png", "latent_stats.json"]:
            assert (experiment_dir / name).exists()
