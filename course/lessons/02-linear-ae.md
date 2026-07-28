# Lesson 02 — A genuinely linear autoencoder

## Learning objective

Understand an undercomplete deterministic bottleneck and its relationship to
PCA.

The model is:

\[
z = W_e x + b_e,\qquad \hat{x}=W_dz+b_d
\]

with an eight-dimensional latent \(z\). There are no hidden nonlinearities and
the output activation is explicitly `none`.

## Read and predict

Read:

- `recipes/ae/ae-001-linear.yaml`
- `src/latent_lab/models/ae.py`
- `src/latent_lab/objectives/ae.py`

Predict:

- Whether it will beat the mean-image baseline.
- Which garment details an eight-dimensional linear subspace will lose.
- Whether nearby latent points should decode smoothly.
- Whether samples drawn arbitrarily from \(N(0,I)\) should look like garments.

## Run

```bash
uv run latent-lab train recipes/ae/ae-001-linear.yaml
uv run latent-lab inspect <PRINTED_RUN_DIR>
```

## Inspect in this order

1. `summary.json`: compare validation reconstruction MSE with Lesson 01.
2. `figures/reconstructions.png`: identify structure retained and lost.
3. `figures/latent-space.png`: this is a two-dimensional PCA projection of the
   eight-dimensional learned codes, colored by labels.
4. `figures/interpolation.png`: look for smooth changes and implausible
   intermediate mixtures.
5. `figures/random-latent-samples.png`: arbitrary normal samples have no reason
   to land in populated regions of the AE latent space.

## What the PCA relationship does and does not mean

With linear maps, squared error, and suitable optimization, the learned
subspace spans the same principal subspace as PCA. The individual latent axes
need not numerically match PCA axes: rotations within the subspace reconstruct
equally well.

The AE has learned an encoder and decoder, not a probability distribution over
valid codes.

## Failure signatures

- MSE near the mean baseline: optimization, data, or checkpoint problem.
- Reconstructions far outside the input range: possible with a linear output;
  inspect whether this is small overshoot or instability.
- Random samples that look poor: expected, not a training failure.

## Advancement gate

Draw the tensor shapes from `[B,1,28,28]` through flattening, the
eight-dimensional bottleneck, and reconstruction. Explain why this model can
compress but cannot yet generate by sampling a known prior.

