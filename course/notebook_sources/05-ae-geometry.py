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
# # Lesson 05 — AE latent geometry and the sampling failure
#
# **Learning objective:** distinguish reconstruction, interpolation, and
# sampling claims by probing occupied and unoccupied latent regions.
#
# Keep three claims separate:
#
# 1. Encoded examples reconstruct well.
# 2. Lines between encoded examples decode plausibly.
# 3. Samples from a known distribution decode plausibly.
#
# An ordinary autoencoder directly optimizes only the first.

# %%
import matplotlib.pyplot as plt
import torch

from latent_lab.course import (
    balanced_class_batch,
    latest_run_dir,
    load_trained_model,
    repository_root,
)
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.diagnostics.interpolations import linear_interpolation

ROOT = repository_root()
run_dir = latest_run_dir(
    ROOT / "runs", "ae/ae-002-nonlinearity/nonlinear"
)
model, config = load_trained_model(run_dir)
train_loader, validation_loader, spec = build_dataloaders(
    config["dataset"], config["training"]
)
names = class_names(config["dataset"]["name"])
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
run_dir

# %% [markdown]
# If the previous lesson did not create this run, train
# `recipes/ae/ae-002-nonlinear.yaml` first. Selecting an exact run keeps all
# figures and metrics tied to the same checkpoint.

# %%
with torch.no_grad():
    output = model(inputs)
plot_reconstruction_grid(
    inputs,
    output.reconstruction,
    labels=labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %% [markdown]
# ## Inspect occupied latent regions
#
# Collect encoded validation examples, reduce them to two principal directions
# for visualization, and color by class. Empty regions in this projection are a
# warning, not a complete map of an eight-dimensional space.

# %%
latent_batches = []
label_batches = []
with torch.no_grad():
    for batch_inputs, batch_labels in validation_loader:
        latent_batches.append(model(batch_inputs).latent)
        label_batches.append(batch_labels)
        if sum(batch.shape[0] for batch in latent_batches) >= 2000:
            break
latents = torch.cat(latent_batches)[:2000]
latent_labels = torch.cat(label_batches)[:2000]
centered = latents - latents.mean(dim=0, keepdim=True)
_u, _s, vectors = torch.pca_lowrank(centered, q=2)
coordinates = centered @ vectors[:, :2]

figure, axis = plt.subplots(figsize=(7, 6))
scatter = axis.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=latent_labels,
    cmap="tab10",
    s=8,
    alpha=0.6,
)
axis.set(title="Occupied AE latent regions (2D PCA view)")
figure.colorbar(scatter, ax=axis, label="class")
figure.tight_layout()
figure

# %% [markdown]
# ## Interpolation is not sampling
#
# Predict what will happen in the middle of a line between a shoe and a shirt.
# Smoothness follows from the decoder; probability density does not.

# %%
with torch.no_grad():
    interpolation = linear_interpolation(
        output.latent[0], output.latent[6], 11
    )
    decoded_path = model.decode(interpolation)
plot_image_grid(
    decoded_path,
    title="A straight latent interpolation",
    max_items=11,
)

# %% [markdown]
# Now compare the empirical latent scale with the convenient prior $N(0,I)$.

# %%
print("encoded mean:", latents.mean(dim=0))
print("encoded std:", latents.std(dim=0))
torch.manual_seed(0)
with torch.no_grad():
    random_decoded = model.decode(
        torch.randn(16, latents.shape[1])
    )
plot_image_grid(
    random_decoded,
    title="N(0, I) is an out-of-distribution query for this decoder",
    max_items=16,
)

# %% [markdown]
# ## Advancement gate
#
# Explain why smooth interpolation does not imply valid random sampling. Your
# answer must mention occupied regions, probability density, latent scale, and
# what the training objective did—or did not—constrain.
