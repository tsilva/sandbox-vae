# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 07 — ELBO, KL pressure, and a usable prior
#
# **Learning objective:** interpret the VAE objective as a rate–distortion
# tradeoff and test whether prior pressure makes sampling more meaningful.
#
# The implemented minimization objective is:
#
# $$
# L=
# \underbrace{-\mathbb E_{q(z|x)}[\log p_\theta(x|z)]}_{\text{distortion}}
# +\beta\underbrace{D_{KL}(q_\phi(z|x)\|p(z))}_{\text{rate}}
# $$
#
# with $p(z)=N(0,I)$ and $\beta=1$.

# %%
import torch

from latent_lab.config import load_yaml
from latent_lab.course import (
    balanced_class_batch,
    latest_run_dir,
    load_run_summary,
    load_trained_model,
    repository_root,
)
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.training import run_training

ROOT = repository_root()
config = load_yaml(ROOT / "recipes/vae/vae-001-basic.yaml")

# %% [markdown]
# ## Predict the causal tradeoff
#
# Relative to beta zero, predict:
#
# - reconstruction distortion,
# - raw KL,
# - active dimensions,
# - and prior-sample coherence.
#
# State *why* each quantity should move, not only its direction.
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Relative to beta zero, reconstruction should usually worsen because KL now
# charges for input information. Raw KL and active dimensions should decrease as
# posteriors move toward $N(0,I)$. Prior samples should become more coherent
# because decoding $N(0,I)$ is now closer to the latent distribution seen during
# training. Excessive pressure could instead produce posterior collapse.
# </details>

# %%
result = run_training(config, run_root=ROOT / "runs")
run_dir = result.run_dir
standard_summary = load_run_summary(run_dir)
beta_zero_dir = latest_run_dir(ROOT / "runs", "vae/vae-000-kl-off")
beta_zero_summary = load_run_summary(beta_zero_dir)

comparison = {
    "beta-zero": beta_zero_summary["best_validation_metrics"],
    "beta-one": standard_summary["best_validation_metrics"],
}
comparison

# %% [markdown]
# Do not compare this BCE number with the earlier AE MSE. The VAE reconstruction
# term is binary cross-entropy summed per example; the scale and observation
# model changed.

# %%
model, resolved = load_trained_model(run_dir)
_train_loader, validation_loader, spec = build_dataloaders(
    resolved["dataset"], resolved["training"]
)
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
names = class_names(resolved["dataset"]["name"])
with torch.no_grad():
    output = model(inputs)
_ = plot_reconstruction_grid(
    inputs,
    output.reconstruction,
    labels=labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %% [markdown]
# ## Inspect whether the prior became more useful
#
# A more prior-compatible latent can cost reconstruction information. Judge the
# exchange with both metrics and images.

# %%
torch.manual_seed(0)
with torch.no_grad():
    prior_samples = model.decode(torch.randn(16, config["model"]["latent_dim"]))
_ = plot_image_grid(
    prior_samples,
    title="Samples after beta-one prior pressure",
    max_items=16,
)

# %% [markdown]
# ## Low KL is not automatically success
#
# KL approaching zero means $q(z|x)$ approaches the prior for every input. If
# reconstructions remain input-dependent, the model may use little information
# efficiently. If reconstructions become constant, the decoder ignores $z$:
# posterior collapse.
#
# ## Advancement gate
#
# Explain why KL is an information rate, why reducing it can help sampling, and
# why reducing it to zero can destroy the representation. Include
# “posterior collapse,” “decoder ignores $z$,” and “reconstruction evidence.”
