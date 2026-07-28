# Lesson 00 — Laboratory orientation and trust checks

> Interactive lab: [open the generated notebook](../notebooks/00-laboratory.ipynb).
> Authors edit the [Jupytext source](../notebook_sources/00-laboratory.py).

## Learning objective

Understand the repository contract and establish that every model can execute,
backpropagate, checkpoint, and produce inspectable artifacts before spending
time on real experiments.

## Read first

- `README.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `src/latent_lab/models/outputs.py`
- `src/latent_lab/training/trainer.py`

Answer before running: why does every model return a reconstruction, latent,
and model-specific `extras` dictionary rather than making the trainer know each
architecture?

## Run

```bash
uv sync --locked
uv run pytest
uv run latent-lab train recipes/smoke/fake-ae.yaml --device cpu
uv run latent-lab train recipes/smoke/fake-vae.yaml --device cpu
uv run latent-lab train recipes/smoke/fake-vqvae.yaml --device cpu
```

These use random synthetic images for one epoch. They are integration tests,
not learning results.

Inspect one directory:

```bash
uv run latent-lab inspect <PRINTED_RUN_DIR>
```

## Look for

- All tests pass.
- Each run contains a resolved config, JSONL metrics, best checkpoint, summary,
  and reconstruction figure.
- AE metrics contain reconstruction and optional latent regularization terms.
- VAE metrics expose reconstruction, raw KL, effective KL, weighted KL, and
  beta separately.
- VQ-VAE metrics expose reconstruction, codebook, commitment, perplexity, and
  codes used.

Do not compare the three smoke total losses. Their objective scales and terms
are different.

## Failure signatures

- No checkpoint: validation loss was not finite or the training loop failed.
- Missing VAE KL: the model/objective contract is broken.
- No VQ codebook gradient test: the embeddings might never learn.
- A smoke reconstruction that looks gray is expected; random images have no
  compressible semantic structure and one epoch is deliberately insufficient.

## Advancement gate

Without reading code, explain:

1. What a recipe controls.
2. What a study adds on top of a recipe.
3. Why raw runs are ignored by Git while reports are committed.
4. Why passing a smoke run does not show that a model is scientifically valid.
