# Lesson 05 — AE latent geometry and the sampling failure

> Interactive lab: [open the generated notebook](../notebooks/05-ae-geometry.ipynb).
> Authors edit the [Jupytext source](../notebook_sources/05-ae-geometry.py).

## Learning objective

Distinguish three claims that are often conflated:

1. Encoded training examples reconstruct well.
2. Lines between encoded examples decode plausibly.
3. Samples from a simple known distribution decode plausibly.

An ordinary AE optimizes only the first claim.

## Use an existing run

Choose your best nonlinear AE from Lessons 03–04:

```bash
uv run latent-lab inspect <AE_RUN_DIR>
```

Open these together:

- `reconstructions.png`
- `latent-space.png`
- `interpolation.png`
- `random-latent-samples.png`

## Interrogate the figures

For the latent scatter:

- Are class colors locally clustered?
- Are there empty regions?
- Remember that latents larger than two dimensions are projected with PCA; the
  picture can hide separation in discarded dimensions.

For interpolation:

- Do endpoints match recognizable examples?
- Are middle points plausible garments or smooth pixel mixtures?
- A single attractive interpolation is qualitative evidence, not a global
  statement about latent geometry.

For random latent samples:

- The tool draws $z\sim N(0,I)$.
- The AE was never told that encoded examples should follow $N(0,I)$.
- The scale, orientation, density, and holes of the encoded distribution are
  therefore unconstrained.

## The motivating failure

An AE gives us $f(x)$ and $g(z)$, but not a tractable model of
$p(z)$. Sampling a random vector is an out-of-distribution query to the
decoder.

This is the exact problem the next model will address. The VAE will trade some
reconstruction freedom for a posterior that is regularized toward a known
prior.

## Advancement gate

Explain why smooth interpolation does not imply valid random sampling. Your
answer must use the concepts of occupied latent regions, probability density,
and training constraints.
