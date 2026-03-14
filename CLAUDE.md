# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A zero-to-hero VAE curriculum: 10 progressive experiments teaching variational autoencoders from first principles. Package name is `sandbox-autoencoders`, installed from `src/sandbox_autoencoders/`.

## Commands

```bash
# Install (editable + dev deps)
pip install -e .[dev]

# Run all tests (smoke tests use synthetic data, no downloads needed)
pytest -q

# Run a single test
pytest tests/test_smoke.py::test_exp_02 -q

# Run a specific experiment
python -m sandbox_autoencoders.experiments.exp_02_vanilla_vae_baseline \
  --output-dir outputs --data-root data --device cpu --wandb-mode disabled
```

## Architecture

**Experiment pattern**: Each `exp_XX_*.py` is a thin CLI module calling `run_experiment_main(experiment_id)`. All experiment specs live in `experiments/shared.py` as `ExperimentSpec` objects containing a `variants_factory` that produces `VariantConfig` lists. The training loop in `training/runner.py::execute_experiment` handles everything: model creation, training, artifact generation, and W&B logging.

**Multi-variant experiments** (sweeps/ablations) run multiple `VariantConfig`s, each in its own subdirectory. The best variant's artifacts are copied to the root experiment directory.

**Model hierarchy**: `ConvEncoder` + `ConvDecoder` compose into `Autoencoder` (deterministic) or `VariationalAutoencoder` (adds `VariationalHead` for mu/logvar). Both return `ModelOutput` dataclass. Decoder has `standard` (32 base channels) and `weak` (16) variants.

**Loss system**: `training/losses.py` provides reconstruction losses (bernoulli, l1, mse, mixed) and KL computation with optional free_bits. Beta scheduling (constant, warmup, cyclical) is in `training/schedules.py`.

**Dataset adapters**: Registry in `datasets/registry.py` maps IDs (mnist, dsprites, celeba, moving_mnist) to `DatasetAdapter` subclasses. Setting `SANDBOX_AUTOENCODERS_TEST_MODE=1` swaps in tiny synthetic datasets.

**Test mode**: Tests set `SANDBOX_AUTOENCODERS_TEST_MODE=1` via `conftest.py`. Sweep experiments auto-reduce to 2 variants via `_pick()` in `shared.py`. The `mini_sweep_mode()` check in `utils/io.py` controls this.

## Adding a New Experiment

1. Add an `ExperimentSpec` entry in `experiments/shared.py::get_experiment_spec`
2. Create `experiments/exp_XX_name.py` with `if __name__ == "__main__": run_experiment_main("exp_XX_name")`
3. Add a smoke test in `tests/test_smoke.py`
4. Use `_pick()` wrapper in `variants_factory` if >2 variants (for test mode)

## Key Conventions

- All experiments share the same CLI interface via `utils/cli.py::build_parser`
- Deterministic seeding via `seed_everything()` with `torch.use_deterministic_algorithms`
- Outputs per variant: `history.json`, `summary.json`, `notes.md`, `recon_grid.png`, `latent_stats.json`
- Optional outputs: `interp_sheet.png`, `prior_samples.png`, `traversal_sheet.png`

## Important

- README.md must be kept up to date with any significant project changes
