# Lesson 12 — Turning VQ-VAE tokens into a generative model

## Learning objective

Learn the separation between representation learning and prior modeling.

The VQ-VAE learns:

\[
x \rightarrow z_e \rightarrow k_{1:H,1:W} \rightarrow \hat{x}
\]

The prior learns:

\[
p(k_1,\dots,k_N)=\prod_{i=1}^{N}p(k_i\mid k_{<i})
\]

where the spatial token grid is flattened in raster order.

## Establish the baseline

Use the chosen VQ-VAE run from Lessons 10–11. Reopen
`uniform-random-token-samples.png`. Those samples answer:

> What if every code at every position is selected independently and uniformly?

The learned prior should answer:

> Which token arrangements resemble those produced by encoded training images?

## Read the prior

- `src/latent_lab/models/prior.py`
- `src/latent_lab/training/prior_trainer.py`

The small GRU prior uses teacher forcing during training. Its input at position
\(i\) is the true previous token; its target is the current token. At sampling
time it must consume its own sampled history.

## Predict

- Initial cross-entropy relative to \(\log K\), the uniform-prior baseline.
- Whether validation perplexity should approach codebook size or effective
  token uncertainty.
- How learned-prior samples should differ from uniform-token samples.
- Why exposure error can accumulate during autoregressive sampling.

## Run

Use the exact best VQ checkpoint:

```bash
uv run latent-lab train-prior \
  recipes/prior/prior-001-gru.yaml \
  <VQ_RUN_DIR>/checkpoint-best.pt

uv run latent-lab inspect <PRINTED_PRIOR_RUN_DIR>
```

Token extraction uses the frozen VQ-VAE. Only the prior is trained.

## Inspect

- Training versus validation cross-entropy.
- Validation perplexity.
- `learned-prior-samples.png`.
- The referenced VQ checkpoint in `resolved-config.yaml`.

Prior perplexity is a next-token prediction metric. Do not confuse it with
VQ codebook usage perplexity:

- Codebook perplexity: diversity of encoder assignments.
- Prior perplexity: uncertainty remaining when predicting the next code from
  previous codes.

## Limitations

The GRU scans a two-dimensional grid as a one-dimensional sequence and is
intentionally small. PixelCNN or a Transformer can model spatial dependencies
more naturally or at greater scale. This lesson is about the factorization and
two-stage generative pipeline, not state-of-the-art samples.

## Advancement gate

Explain why the VQ encoder and decoder can remain frozen while training the
prior, and why improving prior likelihood cannot repair information already
discarded by the VQ-VAE.

