from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sandbox_autoencoders.training.runner import VariantConfig, execute_experiment
from sandbox_autoencoders.utils.cli import build_parser
from sandbox_autoencoders.utils.io import mini_sweep_mode


@dataclass
class ExperimentSpec:
    experiment_id: str
    description: str
    variants_factory: Callable[[int | None], list[VariantConfig]]
    default_dataset: str
    default_epochs: int


def _pick(variants: list[VariantConfig]) -> list[VariantConfig]:
    return variants[:2] if mini_sweep_mode() and len(variants) > 2 else variants


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    specs: dict[str, ExperimentSpec] = {
        "exp_01_autoencoder_baseline": ExperimentSpec(
            experiment_id="exp_01_autoencoder_baseline",
            description="Deterministic autoencoder baseline on MNIST.",
            default_dataset="mnist",
            default_epochs=3,
            variants_factory=lambda epochs: [
                VariantConfig(
                    name="baseline",
                    dataset_id="mnist",
                    model_type="ae",
                    latent_dim=16,
                    loss_type="mse",
                    epochs=epochs or 3,
                    enable_interpolation=True,
                    notes={
                        "healthy": "Reconstructions sharpen quickly and the deterministic bottleneck remains active.",
                        "failure": "Reconstructions stay blurry or collapse to the dataset mean, which usually means the bottleneck is too small or optimization is unstable.",
                        "compare": "This is the baseline for later KL tradeoffs, so compare sharpness and latent activity to the first VAE run.",
                    },
                    tags=["baseline"],
                )
            ],
        ),
        "exp_02_vanilla_vae_baseline": ExperimentSpec(
            experiment_id="exp_02_vanilla_vae_baseline",
            description="Vanilla VAE baseline on MNIST.",
            default_dataset="mnist",
            default_epochs=4,
            variants_factory=lambda epochs: [
                VariantConfig(
                    name="vanilla_vae",
                    dataset_id="mnist",
                    model_type="vae",
                    latent_dim=16,
                    loss_type="bernoulli",
                    beta=1e-3,
                    epochs=epochs or 4,
                    enable_prior_samples=True,
                    enable_interpolation=True,
                    tags=["baseline"],
                )
            ],
        ),
        "exp_03_observation_model_ablation": ExperimentSpec(
            experiment_id="exp_03_observation_model_ablation",
            description="Observation model comparison on MNIST.",
            default_dataset="mnist",
            default_epochs=3,
            variants_factory=lambda epochs: _pick(
                [
                    VariantConfig(name="bernoulli", dataset_id="mnist", model_type="vae", latent_dim=16, loss_type="bernoulli", beta=1e-3, epochs=epochs or 3, enable_prior_samples=True, tags=["ablation"]),
                    VariantConfig(name="mse", dataset_id="mnist", model_type="vae", latent_dim=16, loss_type="mse", beta=1e-3, epochs=epochs or 3, enable_prior_samples=True, tags=["ablation"]),
                    VariantConfig(name="mixed", dataset_id="mnist", model_type="vae", latent_dim=16, loss_type="mixed", beta=1e-3, epochs=epochs or 3, enable_prior_samples=True, tags=["ablation"]),
                ]
            ),
        ),
        "exp_04_beta_and_warmup_sweep": ExperimentSpec(
            experiment_id="exp_04_beta_and_warmup_sweep",
            description="Beta and warmup sweep on MNIST.",
            default_dataset="mnist",
            default_epochs=4,
            variants_factory=lambda epochs: _pick(
                [
                    VariantConfig(name=f"beta_{beta}_warmup_{warmup}", dataset_id="mnist", model_type="vae", latent_dim=16, loss_type="bernoulli", beta=beta, beta_schedule="warmup" if warmup else "constant", warmup_epochs=warmup, epochs=epochs or 4, enable_prior_samples=True, tags=["sweep"])
                    for beta in [1e-4, 3e-4, 1e-3, 3e-3]
                    for warmup in [0, 10]
                ]
            ),
        ),
        "exp_05_latent_capacity_sweep": ExperimentSpec(
            experiment_id="exp_05_latent_capacity_sweep",
            description="Latent capacity sweep on dSprites.",
            default_dataset="dsprites",
            default_epochs=4,
            variants_factory=lambda epochs: _pick(
                [
                    VariantConfig(
                        name=f"latent_{latent_dim}",
                        dataset_id="dsprites",
                        model_type="vae",
                        latent_dim=latent_dim,
                        loss_type="bernoulli",
                        beta=1e-3,
                        epochs=epochs or 4,
                        enable_interpolation=True,
                        enable_traversals=True,
                        tags=["capacity"],
                    )
                    for latent_dim in [8, 16, 32, 64, 128, 256]
                ]
            ),
        ),
        "exp_06_kl_stability_and_collapse": ExperimentSpec(
            experiment_id="exp_06_kl_stability_and_collapse",
            description="KL stability strategies on dSprites.",
            default_dataset="dsprites",
            default_epochs=5,
            variants_factory=lambda epochs: _pick(
                [
                    VariantConfig(name="warmup", dataset_id="dsprites", model_type="vae", latent_dim=32, loss_type="bernoulli", beta=1e-3, beta_schedule="warmup", warmup_epochs=5, epochs=epochs or 5, enable_traversals=True, tags=["collapse"]),
                    VariantConfig(name="free_bits", dataset_id="dsprites", model_type="vae", latent_dim=32, loss_type="bernoulli", beta=1e-3, free_bits=0.03, epochs=epochs or 5, enable_traversals=True, tags=["collapse"]),
                    VariantConfig(name="cyclical", dataset_id="dsprites", model_type="vae", latent_dim=32, loss_type="bernoulli", beta=1e-3, beta_schedule="cyclical", epochs=epochs or 5, enable_traversals=True, tags=["collapse"]),
                ]
            ),
        ),
        "exp_07_disentanglement_and_traversals": ExperimentSpec(
            experiment_id="exp_07_disentanglement_and_traversals",
            description="Disentanglement proxies and latent traversals on dSprites.",
            default_dataset="dsprites",
            default_epochs=5,
            variants_factory=lambda epochs: [
                VariantConfig(
                    name="traversal_probe",
                    dataset_id="dsprites",
                    model_type="vae",
                    latent_dim=32,
                    loss_type="bernoulli",
                    beta=2e-3,
                    epochs=epochs or 5,
                    enable_interpolation=True,
                    enable_traversals=True,
                    tags=["disentanglement"],
                )
            ],
        ),
        "exp_08_real_image_vae": ExperimentSpec(
            experiment_id="exp_08_real_image_vae",
            description="Real-image VAE comparison on CelebA.",
            default_dataset="celeba",
            default_epochs=3,
            variants_factory=lambda epochs: _pick(
                [
                    VariantConfig(name=f"{decoder}_{loss}", dataset_id="celeba", model_type="vae", latent_dim=64, loss_type=loss, decoder_variant=decoder, beta=1e-3, epochs=epochs or 3, enable_prior_samples=True, enable_interpolation=True, tags=["real-image"])
                    for decoder in ["standard", "weak"]
                    for loss in ["l1", "mse", "mixed"]
                ]
            ),
        ),
        "exp_09_sampling_and_geometry": ExperimentSpec(
            experiment_id="exp_09_sampling_and_geometry",
            description="Sampling and geometry diagnostics on CelebA.",
            default_dataset="celeba",
            default_epochs=4,
            variants_factory=lambda epochs: [
                VariantConfig(
                    name="geometry_probe",
                    dataset_id="celeba",
                    model_type="vae",
                    latent_dim=64,
                    loss_type="mixed",
                    beta=1e-3,
                    epochs=epochs or 4,
                    enable_prior_samples=True,
                    enable_interpolation=True,
                    enable_sampling_diagnostics=True,
                    tags=["sampling", "geometry"],
                )
            ],
        ),
        "exp_10_temporal_vae_extension": ExperimentSpec(
            experiment_id="exp_10_temporal_vae_extension",
            description="Optional temporal VAE extension on MovingMNIST.",
            default_dataset="moving_mnist",
            default_epochs=4,
            variants_factory=lambda epochs: _pick(
                [
                    VariantConfig(name="framewise", dataset_id="moving_mnist", model_type="vae", latent_dim=16, loss_type="bernoulli", beta=1e-3, temporal=True, smoothness_penalty=0.0, epochs=epochs or 4, enable_prior_samples=True, tags=["temporal"]),
                    VariantConfig(name="sequence", dataset_id="moving_mnist", model_type="vae", latent_dim=16, loss_type="bernoulli", beta=1e-3, temporal=True, smoothness_penalty=0.01, epochs=epochs or 4, enable_prior_samples=True, tags=["temporal"]),
                    VariantConfig(name="smooth_sequence", dataset_id="moving_mnist", model_type="vae", latent_dim=16, loss_type="bernoulli", beta=1e-3, temporal=True, smoothness_penalty=0.05, epochs=epochs or 4, enable_prior_samples=True, tags=["temporal"]),
                ]
            ),
        ),
    }
    return specs[experiment_id]


def run_experiment_main(experiment_id: str, argv: list[str] | None = None) -> dict:
    spec = get_experiment_spec(experiment_id)
    parser = build_parser(spec.description)
    args = parser.parse_args(argv)
    epochs = args.epochs or spec.default_epochs
    variants = spec.variants_factory(epochs)
    if args.dataset:
        for variant in variants:
            variant.dataset_id = args.dataset
    return execute_experiment(
        experiment_id=spec.experiment_id,
        variants=variants,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        num_workers=args.num_workers,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
    )
