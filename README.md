<div align="center">
  <img src="logo.png" alt="sandbox-vae" width="512"/>

  # Zero-to-Hero VAE Curriculum
</div>

This repository is a runnable curriculum for learning variational autoencoders from first principles through diagnostics, ablations, and an optional temporal extension. The core track stays on static image VAEs first, then adds a sequence-focused experiment after the fundamentals are already in place.

The implementation lives under `src/sandbox_autoencoders`. Each experiment is a thin module under `src/sandbox_autoencoders/experiments` that delegates to shared dataset, model, training, artifact, and W&B infrastructure.

## What You Get

- A dataset registry with `mnist`, `dsprites`, `celeba`, and `moving_mnist`
- Shared AE/VAE training code with standard local outputs and W&B logging
- Ten experiment entrypoints, including sweep and ablation experiments
- Test-mode synthetic adapters so smoke tests run quickly without large downloads

## Installation

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The package expects Python 3.11+ and uses `torch`, `torchvision`, `numpy`, `Pillow`, and `wandb`.

## Datasets

The dataset registry supports these ids:

- `mnist`: downloaded automatically via `torchvision`
- `dsprites`: downloaded automatically to `data/dsprites/dsprites.npz`
- `celeba`: requested via `torchvision.datasets.CelebA`
- `moving_mnist`: generated from MNIST digits at runtime

During tests, setting `SANDBOX_AUTOENCODERS_TEST_MODE=1` swaps in tiny synthetic datasets so the experiments run in seconds.

## Shared CLI

Every experiment accepts the same core flags:

```bash
--dataset
--data-root
--output-dir
--epochs
--batch-size
--seed
--device
--wandb-mode            # online | offline | disabled
--wandb-project
--wandb-entity
--wandb-run-name
--wandb-group
--wandb-tags
```

Typical local run:

```bash
python -m sandbox_autoencoders.experiments.exp_02_vanilla_vae_baseline \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode disabled
```

Typical W&B run:

```bash
python -m sandbox_autoencoders.experiments.exp_05_latent_capacity_sweep \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode online \
  --wandb-project vae-curriculum \
  --wandb-tags capacity dsprites
```

## Standard Outputs

Each experiment writes into `OUTPUT_DIR/<experiment_id>/`.

Required outputs:

- `history.json`
- `summary.json`
- `notes.md`
- `recon_grid.png`
- `latent_stats.json`

Optional outputs, depending on the experiment:

- `interp_sheet.png`
- `prior_samples.png`
- `traversal_sheet.png`

Sweep and ablation experiments also create one subdirectory per variant inside the experiment directory.

## How To Read The Runs

Healthy patterns to look for:

- `val_recon_loss` falling while `active_latents` stays above zero
- `train_kl_loss` and `val_kl_loss` remaining nonzero for VAEs
- Better prior samples as the latent space becomes more usable
- Traversal sheets where one factor changes at a time instead of many factors moving together

Warning patterns to look for:

- `val_kl_loss` going to nearly zero while reconstructions keep improving: posterior collapse
- `active_latents` staying low as `latent_dim` increases: unused capacity
- Prior samples looking much worse than reconstructions: prior mismatch
- Temporal smoothness improving while consecutive frames become too similar: over-smoothing

## Experiment Guide

### 1. `exp_01_autoencoder_baseline`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_01_autoencoder_baseline
```

Default dataset: `mnist`

Objective:

- Build intuition for encoder-decoder training without KL regularization
- Establish the reconstruction baseline that later VAE experiments are judged against

What this experiment does:

- Trains a deterministic convolutional autoencoder
- Saves reconstructions and an interpolation sheet

What to look for:

- Reconstructions should sharpen quickly
- The bottleneck should remain active instead of collapsing to nearly constant codes
- Interpolations can look smooth even though the model has no learned generative prior

Watch out for:

- Blurry or average-looking reconstructions after several epochs
- Latent activity near zero, which means the bottleneck is not being used meaningfully

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_01_autoencoder_baseline \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode disabled
```

### 2. `exp_02_vanilla_vae_baseline`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_02_vanilla_vae_baseline
```

Default dataset: `mnist`

Objective:

- Introduce the ELBO tradeoff between reconstruction quality and KL regularization
- Learn how posterior collapse and prior sampling show up in practice

What this experiment does:

- Trains a vanilla VAE with Bernoulli-style reconstruction loss
- Logs reconstruction grids, latent statistics, prior samples, and interpolations

What to look for:

- Reconstructions may be slightly worse than the autoencoder, which is expected
- `val_kl_loss` should stay measurably above zero
- Prior samples should begin to resemble digits instead of noise

Watch out for:

- `val_kl_loss` collapsing to near zero while reconstructions still improve
- Prior samples remaining incoherent even after reconstructions look decent

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_02_vanilla_vae_baseline \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 3. `exp_03_observation_model_ablation`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_03_observation_model_ablation
```

Default dataset: `mnist`

Objective:

- Learn that the reconstruction term is a likelihood choice, not just a metric choice
- Compare how Bernoulli-style, MSE, and mixed losses change the training signal

What this experiment does:

- Runs one VAE variant per reconstruction objective
- Stores each variant in its own subdirectory

What to look for:

- Different loss families can produce different tradeoffs in sharpness, stability, and KL usage
- Bernoulli often behaves differently from continuous losses on MNIST-like data

Watch out for:

- Treating lower `val_recon_loss` across different objectives as directly comparable without context
- Forgetting to compare sample quality and latent usage, not just the scalar loss

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_03_observation_model_ablation \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 4. `exp_04_beta_and_warmup_sweep`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_04_beta_and_warmup_sweep
```

Default dataset: `mnist`

Objective:

- Learn how the KL weight changes the recon/KL tradeoff
- See how warmup affects early optimization and latent engagement

What this experiment does:

- Sweeps beta across `1e-4`, `3e-4`, `1e-3`, and `3e-3`
- Compares constant-beta and warmup variants

What to look for:

- Lower beta usually improves reconstructions but can weaken the prior
- Higher beta usually raises pressure on the latent and can blur outputs
- Warmup can smooth the early training curve and reduce instant collapse

Watch out for:

- Choosing the best run from `val_loss` alone without checking samples
- Assuming high KL is always good; it can also reflect unstable or over-regularized training

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_04_beta_and_warmup_sweep \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 5. `exp_05_latent_capacity_sweep`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_05_latent_capacity_sweep
```

Default dataset: `dsprites`

Objective:

- Learn how latent size affects bottleneck strength and actual latent usage
- Move from “more dimensions” to “more active dimensions”

What this experiment does:

- Sweeps `latent_dim` across `8`, `16`, `32`, `64`, `128`, and `256`
- Saves reconstructions, interpolations, and traversal sheets for each variant

What to look for:

- Small latents should visibly bottleneck reconstructions
- Increasing `latent_dim` should stop helping once active dimensions plateau
- Traversals should become easier to interpret on a factorized dataset like dSprites

Watch out for:

- Assuming that a larger latent space implies more information use
- Ignoring `active_latents` when comparing model capacity

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_05_latent_capacity_sweep \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 6. `exp_06_kl_stability_and_collapse`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_06_kl_stability_and_collapse
```

Default dataset: `dsprites`

Objective:

- Compare practical anti-collapse strategies
- Understand how warmup, free bits, and cyclical annealing behave differently

What this experiment does:

- Trains one run each for warmup, free bits, and cyclical beta schedules
- Logs traversal sheets so latent re-engagement is visible in images, not just scalars

What to look for:

- Free bits should keep some KL alive instead of letting everything shrink away
- Cyclical schedules should show repeated KL engagement over time
- Warmup should make early training smoother than a constant high beta

Watch out for:

- Good reconstructions hiding a collapsed latent
- Large KL without better samples or more interpretable traversals

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_06_kl_stability_and_collapse \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 7. `exp_07_disentanglement_and_traversals`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_07_disentanglement_and_traversals
```

Default dataset: `dsprites`

Objective:

- Connect latent traversals to interpretable factors
- Use a simple factor-predictability proxy instead of treating “disentanglement” as purely visual

What this experiment does:

- Trains a single dSprites VAE tuned for traversal quality
- Writes traversal sheets and factor-predictability metrics into the summary artifacts

What to look for:

- Single latent dimensions should correspond more closely to one visible factor at a time
- The MIG-style proxy should improve relative to weaker settings

Watch out for:

- Traversals that change multiple attributes at once
- Over-reading a proxy metric without checking the actual images

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_07_disentanglement_and_traversals \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 8. `exp_08_real_image_vae`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_08_real_image_vae
```

Default dataset: `celeba`

Objective:

- Move from toy data to a realistic image manifold
- Compare decoder strength and reconstruction objectives on a harder dataset

What this experiment does:

- Runs six variants: `standard` and `weak` decoders crossed with `l1`, `mse`, and `mixed` losses
- Saves reconstructions, prior samples, and interpolations for each variant

What to look for:

- Decoder strength can improve reconstructions while reducing reliance on the latent
- `l1` often preserves edges better than `mse`
- Prior samples often reveal issues that reconstructions hide

Watch out for:

- Declaring a decoder better solely because its reconstructions are prettier
- Ignoring whether the latent still contributes to generation

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_08_real_image_vae \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 9. `exp_09_sampling_and_geometry`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_09_sampling_and_geometry
```

Default dataset: `celeba`

Objective:

- Study prior mismatch and local geometry instead of just minimizing reconstruction loss
- Compare generated samples against nearby real examples

What this experiment does:

- Trains a CelebA VAE with sampling diagnostics enabled
- Writes posterior summary statistics and nearest-neighbor retrieval indices into `summary.json`

What to look for:

- Prior samples should move closer to posterior-style quality as the aggregate posterior aligns with the prior
- Interpolations should move smoothly between identities or attributes instead of cutting through unrealistic faces

Watch out for:

- Posterior samples looking acceptable while prior samples remain poor
- Interpolations that immediately wash out important attributes

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_09_sampling_and_geometry \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

### 10. `exp_10_temporal_vae_extension`

Module:

```bash
python -m sandbox_autoencoders.experiments.exp_10_temporal_vae_extension
```

Default dataset: `moving_mnist`

Objective:

- Extend the static-image VAE stack into a simple temporal setting
- Study latent smoothness penalties and frame-to-frame consistency

What this experiment does:

- Compares `framewise`, `sequence`, and `smooth_sequence` variants
- Uses MovingMNIST sequences and applies optional latent smoothness penalties

What to look for:

- Moderate smoothness should reduce frame-to-frame jitter in the latent
- Reconstruction quality should remain recognizable while neighboring frames become more coherent

Watch out for:

- Smoothness penalties making all frames look too similar
- Treating temporal consistency as a free improvement if reconstruction diversity collapses

Recommended command:

```bash
python -m sandbox_autoencoders.experiments.exp_10_temporal_vae_extension \
  --output-dir outputs \
  --data-root data \
  --device cpu \
  --wandb-mode offline
```

## Development and Tests

Run the full test suite:

```bash
pytest -q
```

The tests cover:

- Every experiment entrypoint in smoke-test mode
- Dataset adapter contracts and sample shapes
- Latent diagnostics edge cases
- Fixed-seed reproducibility for the vanilla VAE baseline
- W&B `offline` and `disabled` execution

## Practical Notes

- Most real datasets will download on first use and may take time.
- CelebA handling depends on `torchvision` dataset availability in your environment.
- Sweep experiments create multiple variant subdirectories by design.
- The `notes.md` written by each run is a compact per-experiment reading guide; the `README` is the top-level curriculum guide.
