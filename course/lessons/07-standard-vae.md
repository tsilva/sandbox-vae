# Lesson 07 — ELBO, KL pressure, and a sampleable prior

## Learning objective

Interpret the VAE objective as a rate–distortion tradeoff and inspect whether
the prior is becoming usable.

The implemented minimization objective is:

\[
L =
\underbrace{-\mathbb E_{q(z|x)}[\log p_\theta(x|z)]}_{\text{distortion}}
+
\beta\underbrace{D_{KL}(q_\phi(z|x)\|p(z))}_{\text{rate}}
\]

with \(p(z)=N(0,I)\), Bernoulli observation likelihood, and \(\beta=1\).

## Predict before running

Relative to Lesson 06:

- Will reconstruction improve or worsen?
- Will raw KL rise or fall?
- Will fewer dimensions be active?
- Will random-prior samples become more coherent?

State the causal mechanism, not only the direction.

## Run

```bash
uv run latent-lab train recipes/vae/vae-001-basic.yaml
uv run latent-lab inspect <PRINTED_RUN_DIR>
```

## Read the metrics correctly

- `reconstruction_loss`: summed binary cross-entropy per example.
- `kl_loss`: raw information rate in nats per example.
- `effective_kl_loss`: the value used before beta; it differs from raw KL only
  when free bits are enabled.
- `weighted_kl_loss`: contribution to the optimized total.
- `beta`: pressure applied in that epoch.

The total loss decreasing does not tell you which side of the tradeoff changed.

## Inspect the geometry

- `kl-per-dimension.png`: dimensions above the red threshold are counted as
  active.
- `latent-space.png`: look for a more compact, continuous aggregate geometry.
- `interpolation.png`: inspect continuity.
- `random-latent-samples.png`: compare directly with Lesson 06.

Do not expect perfect images from a small MLP VAE. The question is whether
samples are more consistent with the data distribution, not whether they are
photorealistic.

## Advancement gate

Explain why lowering KL all the way to zero is not necessarily success. Include
the phrases “posterior collapse,” “decoder ignores \(z\),” and “reconstruction
evidence.”

