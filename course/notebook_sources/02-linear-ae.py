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
# # Lesson 02 — A genuinely linear autoencoder
#
# **Learning objective:** understand an undercomplete deterministic bottleneck,
# its PCA relationship, and the difference between interpolation and sampling.
#
# A linear encoder and decoder with an eight-dimensional bottleneck implement:
#
# $$
# z=W_ex+b_e,\qquad \hat{x}=W_dz+b_d
# $$
#
# With squared error and suitable optimization, the learned reconstruction
# subspace can span the same principal subspace as PCA. Individual latent axes
# remain free to rotate inside that subspace.

# %%
import matplotlib.pyplot as plt
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
from latent_lab.diagnostics.interpolations import linear_interpolation
from latent_lab.training import compute_mean_image, run_training
from latent_lab.training.seeding import resolve_device

ROOT = repository_root()
recipe_path = ROOT / "recipes/ae/ae-001-linear.yaml"
config = load_yaml(recipe_path)
device = resolve_device(config["training"]["device"])
config

# %% [markdown]
# ## Predict before training
#
# 1. Will this model beat the mean-image baseline?
# 2. Which garment details cannot fit in eight linear coordinates?
# 3. Should straight lines between encoded examples decode smoothly?
# 4. Should arbitrary samples from $N(0,I)$ decode into garments?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The model should beat the constant mean by using input-dependent coordinates,
# but eight linear dimensions will lose fine edges and texture. Linear decoding
# makes latent interpolation smooth. Random $N(0,I)$ samples have no reason to
# land in regions occupied by encoded garments because the AE never learned
# that prior.
# </details>

# %% [markdown]
# ## Train one reproducible recipe
#
# This cell uses the same generic trainer as the CLI and returns its exact run
# directory. The checkpoint, metrics, figures, and resolved recipe remain
# durable evidence under `runs/`.

# %%
result = run_training(config, run_root=ROOT / "runs")
run_dir = result.run_dir
print("run directory:", run_dir)
print("best epoch:", result.best_epoch)
print("best validation MSE:", result.best_validation_loss)

# %% [markdown]
# ## Did optimization actually converge?
#
# A final number hides the path taken to reach it. Plot training and validation
# reconstruction loss together and look for underfitting, instability, or a
# widening generalization gap.

# %%
records = load_metrics(run_dir)
_ = plot_metric_history(
    records,
    ["train/reconstruction_loss", "validation/reconstruction_loss"],
    title="Linear AE reconstruction learning curve",
)

# %% [markdown]
# ## Inspect aligned inputs, reconstructions, and errors
#
# Use one example per class so the grid tests more than one convenient batch.

# %%
model, resolved_config = load_trained_model(run_dir, device=device)
train_loader, validation_loader, spec = build_dataloaders(
    resolved_config["dataset"], resolved_config["training"]
)
names = class_names(resolved_config["dataset"]["name"])
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
with torch.no_grad():
    output = model(inputs.to(device))
reconstructions = output.reconstruction.cpu()
latents = output.latent.cpu()

print("input shape:", tuple(inputs.shape))
print("latent shape:", tuple(latents.shape))
print(
    "reconstruction range:",
    (float(reconstructions.min()), float(reconstructions.max())),
)
_ = plot_reconstruction_grid(
    inputs,
    reconstructions,
    labels=labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %% [markdown]
# Identify what is retained: coarse silhouette, location, and broad class
# structure. Identify what is lost: sharp edges, textures, and details that do
# not align with the dominant linear directions.

# %% [markdown]
# ## Compare with the baseline using the same loss semantics
#
# Both values below use Fashion-MNIST in $[0,1]$ and mean squared error over
# channels and pixels, so their numerical comparison is valid.

# %%
mean_image, _ = compute_mean_image(train_loader, torch.device("cpu"))
baseline_squared_error = (
    inputs - mean_image.expand(inputs.shape[0], -1, -1, -1)
).square().flatten(start_dim=1).mean(dim=1)
model_squared_error = (
    inputs - reconstructions
).square().flatten(start_dim=1).mean(dim=1)
print("balanced-batch mean baseline:", float(baseline_squared_error.mean()))
print("balanced-batch linear AE:", float(model_squared_error.mean()))
print(
    "relative reduction:",
    f"{(1 - model_squared_error.mean() / baseline_squared_error.mean()) * 100:.1f}%",
)

# %% [markdown]
# ## Poke the latent geometry
#
# Smooth decoding along a line tests continuity of the decoder. It does **not**
# establish that points on the line—or samples from a convenient distribution—
# have high probability under encoded data.

# %%
with torch.no_grad():
    path = linear_interpolation(
        output.latent[0], output.latent[1], steps=11
    )
    interpolated = model.decode(path).cpu()
_ = plot_image_grid(
    interpolated,
    title="Straight line between two encoded examples",
    max_items=11,
)

# %%
torch.manual_seed(0)
with torch.no_grad():
    random_latents = torch.randn(16, latents.shape[1], device=device)
    random_decoded = model.decode(random_latents).cpu()
_ = plot_image_grid(
    random_decoded,
    title="Arbitrary N(0, I) samples: the AE never learned this prior",
    max_items=16,
)

# %% [markdown]
# ## Advancement gate
#
# Draw the complete shape path:
#
# `B×1×28×28 → B×784 → B×8 → B×784 → B×1×28×28`
#
# Then explain:
#
# 1. Why this model can beat an input-independent baseline.
# 2. Why smooth interpolation is expected from linear maps.
# 3. Why a good reconstruction loss does not define a probability distribution
#    over valid latent codes.
