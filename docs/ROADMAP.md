# Learning Roadmap

This file is the high-level research map. The runnable teaching sequence,
commands, questions, and advancement gates live in
[`course/README.md`](../course/README.md).

## Stage 0 — Laboratory

- Validate data ranges and preprocessing.
- Establish a mean-image reconstruction baseline.
- Overfit a tiny fake dataset.
- Lock the fixed validation grid and artifact contract.

## Stage 1 — Autoencoders

1. Linear undercomplete AE.
2. Nonlinear AE with matched latent capacity.
3. Latent-size rate–distortion sweep.
4. Overcomplete AE and the identity-mapping failure.
5. Denoising and sparse AEs.
6. Convolutional AE.
7. Interpolation and random-latent decoding.

## Stage 2 — Variational autoencoders

1. VAE with KL disabled as an implementation control.
2. Standard VAE.
3. Beta sweep.
4. Prior samples and latent traversals.
5. Per-dimension KL and active units.
6. Posterior collapse.
7. KL warm-up and free bits.

## Stage 3 — VQ-VAE

1. Nearest-neighbor quantization and straight-through gradients.
2. Codebook-size and embedding-size sweeps.
3. Commitment-weight sweep.
4. Usage, perplexity, and dead-code diagnostics.
5. Gradient versus EMA codebook updates.
6. Uniform token samples versus a learned autoregressive prior.

## Stage 4 — Controlled comparison

Compare matched-capacity AE, VAE, and VQ-VAE models using reconstruction,
latent utilization, linear probes, interpolation, robustness, and sample
quality. Do not treat reconstruction error as a complete generative metric.
