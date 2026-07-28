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
# # Lesson 06 — VAE distributions and the beta-zero control
#
# **Learning objective:** understand reparameterized stochastic encoding before
# adding pressure toward a known prior.
#
# The encoder predicts a distribution:
#
# $$
# q_\phi(z|x)=\mathcal N\left(\mu_\phi(x),
# \operatorname{diag}(\sigma_\phi^2(x))\right)
# $$
#
# and samples through:
#
# $$
# \epsilon\sim\mathcal N(0,I),\qquad
# z=\mu+\exp(0.5\log\sigma^2)\odot\epsilon
# $$

# %%
import torch

from latent_lab.config import load_yaml
from latent_lab.course import (
    balanced_class_batch,
    load_metrics,
    load_trained_model,
    plot_metric_history,
    repository_root,
)
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.training import run_training

ROOT = repository_root()
config = load_yaml(ROOT / "recipes/vae/vae-000-kl-off.yaml")

# %% [markdown]
# ## Poke the reparameterization gradient directly
#
# Predict whether a reconstruction-like loss on $z$ can produce gradients for
# both $\mu$ and `logvar`.
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Yes. Once $\epsilon$ is sampled independently, $z$ is a differentiable
# function of both $\mu$ and `logvar`. A loss on $z$ therefore supplies
# pathwise gradients to both encoder outputs.
# </details>

# %%
torch.manual_seed(0)
mu = torch.tensor([[0.4, -0.2]], requires_grad=True)
logvar = torch.tensor([[0.1, -0.3]], requires_grad=True)
epsilon = torch.randn_like(mu)
z = mu + torch.exp(0.5 * logvar) * epsilon
probe_loss = z.square().mean()
probe_loss.backward()
print("epsilon:", epsilon)
print("gradient dL/dmu:", mu.grad)
print("gradient dL/dlogvar:", logvar.grad)

# %% [markdown]
# Randomness enters through $\epsilon$, which is independent of the encoder
# parameters. For a fixed sampled $\epsilon$, $z$ remains a differentiable
# function of both outputs.
#
# ## Why beta zero?
#
# The control computes KL but gives it zero weight:
#
# $$
# L=L_{\text{reconstruction}}+0\cdot KL
# $$
#
# Predict raw KL, weighted KL, reconstruction quality, active dimensions, and
# $N(0,I)$ sample quality before training.
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Weighted KL must remain exactly zero. Raw KL can grow because nothing rewards
# prior matching, and many latent dimensions may remain active for
# reconstruction. Reconstructions can be strong, while $N(0,I)$ samples remain
# unreliable because that distribution was not enforced.
# </details>

# %%
result = run_training(config, run_root=ROOT / "runs")
run_dir = result.run_dir
print(run_dir)
print(result.best_validation_metrics)

# %%
records = load_metrics(run_dir)
_ = plot_metric_history(
    records,
    [
        "validation/reconstruction_loss",
        "validation/kl_loss",
        "validation/weighted_kl_loss",
    ],
    title="Beta-zero VAE: KL is measured but not optimized",
)

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
# ## Test the unearned prior
#
# A stochastic encoder alone does not make $N(0,I)$ a valid aggregate latent
# distribution. With beta zero, the model is free to move posterior means and
# variances wherever reconstruction finds convenient.

# %%
torch.manual_seed(0)
with torch.no_grad():
    prior_samples = model.decode(torch.randn(16, config["model"]["latent_dim"]))
_ = plot_image_grid(
    prior_samples,
    title="Beta-zero samples from an unconstrained N(0, I) prior",
    max_items=16,
)

# %% [markdown]
# ## Advancement gate
#
# Explain:
#
# 1. Why gradients reach both $\mu$ and `logvar`.
# 2. Why raw KL can grow while weighted KL remains zero.
# 3. Why stochastic encoding alone does not make the prior sampleable.
