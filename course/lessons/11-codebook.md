# Lesson 11 — Codebook capacity, utilization, and commitment

## Learning objective

Treat the codebook as a learned communication channel rather than assuming
that a larger vocabulary automatically creates a richer representation.

## Experiment A: codebook size

Predict reconstruction error, codes used, dead-code count, and perplexity for
codebook sizes 8, 32, 128, and 512.

```bash
uv run latent-lab study studies/vqvae/vqvae-001-codebook-size.yaml --seeds 0
```

Inspect each `diagnostics.json` and `codebook-usage.png`.

Look for:

- Small codebooks saturating most entries.
- Larger codebooks offering capacity the optimizer may never use.
- Lower reconstruction error without proportionally higher effective
  vocabulary.
- A few codes monopolizing assignments.

Dead codes are entries receiving no validation assignments. A dead code
contributes no representational capacity even though it increases the nominal
codebook size.

## Experiment B: commitment weight

The commitment term keeps encoder outputs near the selected embeddings.

```bash
uv run latent-lab study studies/vqvae/vqvae-002-commitment.yaml --seeds 0
```

Compare:

- Raw commitment loss
- Weighted commitment contribution
- Codebook loss
- Reconstruction
- Perplexity and code usage

Changing the weight alters optimization pressure, so compare raw and weighted
terms separately.

## Failure signatures

- Perplexity near 1: nearly all positions choose one code.
- Many codes used once but a few dominate: nominal coverage hides imbalance.
- Good reconstruction with low usage: the spatial grid and decoder may carry
  enough capacity using a small effective vocabulary.
- High commitment weight with poor reconstruction: encoder flexibility may be
  overconstrained.
- Zero codebook gradients: embeddings cannot move; the gradient tests should
  catch this.

## Optional recovery hypotheses

Do not implement these yet. Predict how each might help:

- EMA codebook updates
- Reinitializing dead codes from current encoder outputs
- Smaller codebook
- Lower-dimensional embeddings
- Entropy or usage regularization

## Advancement gate

Given \(K=512\), 40 used codes, and perplexity 9, explain all three numbers and
why none alone establishes representation quality.

