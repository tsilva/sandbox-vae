# Lesson 10 — VQ-VAE quantization and straight-through learning

> Interactive lab: [open the generated notebook](../notebooks/10-vqvae.ipynb).
> Authors edit the [Jupytext source](../notebook_sources/10-vqvae.py).

## Learning objective

Understand how a continuous encoder output becomes a grid of discrete tokens
and how gradients train both the encoder and the codebook.

For each encoder vector $z_e$, select the nearest embedding:

$$
k=\arg\min_j \lVert z_e-e_j\rVert^2,\qquad z_q=e_k
$$

The objective contains:

$$
L = L_{\text{reconstruction}}
+ \lVert \operatorname{sg}[z_e]-e_k\rVert^2
+ \beta\lVert z_e-\operatorname{sg}[e_k]\rVert^2
$$

where `sg` means stop-gradient.

## Read and trace gradients

Read:

- `src/latent_lab/models/quantizer.py`
- `src/latent_lab/models/vqvae.py`
- `src/latent_lab/objectives/vqvae.py`

Before running, identify:

- Which term updates the codebook.
- Which term updates the encoder.
- How reconstruction gradients cross the nondifferentiable nearest-neighbor
  selection.

The straight-through expression uses the quantized value in the forward pass
but gives it the encoder latent's gradient in the backward pass.

## Run

```bash
uv run latent-lab train recipes/vqvae/vqvae-001-basic.yaml
uv run latent-lab inspect <PRINTED_RUN_DIR>
```

## Inspect

- `reconstructions.png`
- `token-maps.png`: each $7\times7$ cell is a discrete code index.
- `codebook-usage.png`
- `diagnostics.json`: codes used, dead codes, and perplexity.
- `uniform-random-token-samples.png`

Codebook perplexity is the exponential entropy of the assignment
distribution. It is an effective vocabulary size, not reconstruction quality.
A 128-entry codebook with perplexity 12 behaves roughly like only a dozen
uniformly used codes.

## The second sampling failure

The VQ-VAE learned a vocabulary and decoder, but no distribution over valid
token arrangements. Uniformly random code grids usually decode incoherently.
This is analogous to choosing random words uniformly and expecting a sentence.

## Advancement gate

Explain why:

1. Argmin blocks ordinary gradients.
2. The straight-through estimator is deliberately biased.
3. Codebook perplexity and codebook size are different.
4. A trained VQ-VAE is not yet a complete generative model.
