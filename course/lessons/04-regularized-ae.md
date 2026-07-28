# Lesson 04 — Denoising and sparse autoencoders

## Learning objective

Learn that an autoencoder bottleneck is not defined only by its number of
coordinates. The training task and regularization also determine what the
latent representation must preserve.

## Experiment A: denoising

The model receives corrupted input \(\tilde{x}\) but is trained against clean
target \(x\):

\[
L = \lVert g(f(\tilde{x}))-x\rVert^2
\]

Predict whether denoising training will improve noisy-input output and whether
it might slightly worsen clean-input reconstruction.

```bash
uv run latent-lab study studies/ae/ae-004-denoising.yaml --seeds 0
```

Inspect:

- `reconstructions.png` for clean inputs.
- `corrupted-input-reconstructions.png` for the denoising variant.
- Validation MSE, remembering validation itself uses clean inputs.

The most important denoising evidence is visual because the current shared
validation metric measures clean reconstruction. Record this limitation.

## Experiment B: sparsity

The sparse objective is:

\[
L = L_{\text{reconstruction}} + \lambda\operatorname{mean}(|z|)
\]

Predict the curves for reconstruction MSE and mean absolute latent activation
as \(\lambda\) increases.

```bash
uv run latent-lab study studies/ae/ae-005-sparsity.yaml --seeds 0
```

Inspect named loss terms separately. Total loss cannot tell you whether a run
has low reconstruction error, low activation, or merely a different weighting.

## Failure signatures

- Weighted regularization is exactly zero with nonzero weight: loss wiring is
  broken.
- Very high sparsity pressure and constant reconstructions: the latent has
  become uninformative.
- Denoising output simply copies noise: corruption may not be applied or the
  model may be undertrained.

## Advancement gate

Contrast these three bottlenecks:

- Low latent dimension
- Noisy input with a clean target
- L1 pressure on latent activity

Explain what behavior each one encourages and why they are not interchangeable.

