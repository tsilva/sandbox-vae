# Lesson 06 — VAE distributions and the beta-zero control

## Learning objective

Understand the stochastic encoder and reparameterization trick before adding
KL pressure.

The encoder predicts:

\[
q_\phi(z|x)=\mathcal N\left(\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right)
\]

and samples using:

\[
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu+\exp(0.5\log\sigma^2)\odot\epsilon
\]

## Read and trace

Read:

- `src/latent_lab/models/vae.py`
- `src/latent_lab/objectives/vae.py`
- `recipes/vae/vae-000-kl-off.yaml`

Trace gradients symbolically from reconstruction loss through \(z\) to both
\(\mu\) and `logvar`. Explain why directly sampling from
\(\mathcal N(\mu,\sigma^2)\) through a nondifferentiable sampling operation
would be a problem.

## Why beta zero?

This run computes KL but assigns it zero weight:

\[
L = L_{\text{reconstruction}} + 0\cdot KL
\]

It is a control. It validates the stochastic encoder/decoder path while showing
what happens when nothing regularizes the posterior toward the prior.

Predict:

- Reconstruction quality relative to the standard VAE.
- Raw KL magnitude.
- Number of active latent dimensions.
- Quality of \(N(0,I)\) samples.

## Run

```bash
uv run latent-lab train recipes/vae/vae-000-kl-off.yaml
uv run latent-lab inspect <PRINTED_RUN_DIR>
```

## Inspect

- `validation/reconstruction_loss`
- Raw `validation/kl_loss`
- `validation/weighted_kl_loss`, which must be zero
- `kl-per-dimension.png`
- `random-latent-samples.png`

The BCE reconstruction is summed per sample, not averaged per pixel. Do not
compare its numeric value directly with AE mean MSE.

## Advancement gate

Explain:

1. Why the reparameterization expression is differentiable with respect to
   \(\mu\) and `logvar`.
2. Why a computed but unweighted KL can become large.
3. Why this model still cannot be trusted to decode \(N(0,I)\) samples.

