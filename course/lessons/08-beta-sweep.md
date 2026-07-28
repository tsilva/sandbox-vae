# Lesson 08 — The VAE rate-distortion experiment

> Interactive lab: [open the generated notebook](../notebooks/08-rate-distortion.ipynb).
> Authors edit the [Jupytext source](../notebook_sources/08-rate-distortion.py).

## Learning objective

Observe directly that beta is not a generic “regularization strength.” It sets
the exchange rate between reconstruction distortion and latent information.

## Predict the four regimes

Before running, fill a table for beta values 0, 0.1, 1, and 4:

| Beta | Reconstruction | Raw KL | Active dimensions | Prior samples |
|---:|---|---|---|---|
| 0 | predict | predict | predict | predict |
| 0.1 | predict | predict | predict | predict |
| 1 | predict | predict | predict | predict |
| 4 | predict | predict | predict | predict |

## Run the exploratory sweep

```bash
uv run latent-lab study studies/vae/vae-001-beta-sweep.yaml --seeds 0
```

The study prints one run directory per variant and an aggregate summary path.
Inspect each run:

```bash
uv run latent-lab inspect <RUN_DIR>
```

## Build the rate-distortion view

For each beta, record:

- Best validation reconstruction loss: distortion
- Best validation raw KL: rate
- Active dimensions from `summary.json`
- A short visual rating of prior samples

Plot mentally or on paper: distortion on one axis and rate on the other. Higher
beta should generally buy lower rate at the cost of higher distortion, though
optimization noise and collapse can make the curve imperfect.

## Common interpretation errors

- Comparing total loss across beta values as if lower were universally better.
  The objective itself changed.
- Calling low KL “good regularization” without checking reconstructions.
- Calling high KL “more expressive” without checking prior samples.
- Choosing beta from one attractive sample grid.

## Confirmation rule

Pick only two beta values that support materially different conclusions, then
repeat them over seeds. You may temporarily copy the study and remove the other
variants, or rerun the full study if compute is unimportant.

## Advancement gate

Describe beta in rate–distortion language. Given a run with excellent
reconstruction, high KL, and poor prior samples, diagnose the model without
using the vague phrase “overfitting.”
