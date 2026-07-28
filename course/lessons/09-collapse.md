# Lesson 09 — Posterior collapse, warm-up, and free bits

> Interactive lab: [open the generated notebook](../notebooks/09-collapse.ipynb).
> Authors edit the [Jupytext source](../notebook_sources/09-collapse.py).

## Learning objective

Recognize when the VAE decoder ignores its latent input and understand two
interventions that alter KL optimization dynamics.

## Collapse signature

Posterior collapse typically combines:

- KL near zero
- Few or zero active dimensions
- Similar posterior parameters across inputs
- Reconstructions that rely on decoder bias or autoregressive context
- Prior samples that may be consistent but uninformative

Low KL alone is not enough; inspect reconstruction and active dimensions.

## Interventions

### KL warm-up

Increase beta from zero to its target over the first ten epochs. Early training
can learn to use $z$ before full prior pressure arrives.

### Free bits

For each latent dimension, optimize:

$$
\max(\lambda, KL_j)
$$

Below $\lambda$, the KL term is constant and supplies no gradient pushing that
dimension closer to zero. This does not force the model to use the dimension;
it removes the reward for compressing it below the allowance.

## Predict and run

Predict which method will retain the most active dimensions and which may have
the best reconstruction.

```bash
uv run latent-lab study studies/vae/vae-002-collapse-remedies.yaml --seeds 0
```

Inspect:

- Beta history in `metrics.jsonl`
- Raw versus effective KL
- `kl-per-dimension.png`
- `summary.json` active dimension count
- Reconstructions and prior samples

With free bits, effective KL can exceed raw KL because the objective includes a
constant floor. Always diagnose activity from raw KL.

## Important limitation

This small Fashion-MNIST MLP may not exhibit catastrophic collapse. A negative
result is still informative: interventions cannot be credited with “fixing”
collapse unless the baseline actually collapsed.

## Advancement gate

Given three hypothetical runs:

- KL 0.02, strong reconstructions
- KL 0.02, constant reconstructions
- KL 15, strong reconstructions but poor prior samples

explain why they require different diagnoses despite sharing some metric
features.
