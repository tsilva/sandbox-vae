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
# # Lesson 10 — VQ-VAE quantization and straight-through learning
#
# **Learning objective:** trace how a nondifferentiable code choice becomes a
# trainable discrete bottleneck with distinct gradient paths.
#
# For each encoder vector $z_e$, choose its nearest embedding:
#
# $$
# k=\arg\min_j\lVert z_e-e_j\rVert^2,\qquad z_q=e_k
# $$
#
# Argmin has no ordinary useful gradient. VQ-VAE assigns different optimization
# jobs to reconstruction, codebook, and commitment paths.

# %%
import matplotlib.pyplot as plt
import torch

from latent_lab.config import load_yaml
from latent_lab.course import (
    balanced_class_batch,
    load_run_summary,
    load_trained_model,
    repository_root,
)
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.models.quantizer import VectorQuantizer
from latent_lab.training import run_training

ROOT = repository_root()

# %% [markdown]
# ## Trace gradients with a tiny quantizer
#
# Predict which parameters receive gradients from each isolated term:
#
# | Term | Encoder latent | Codebook |
# |---|---|---|
# | Reconstruction through straight-through output | ? | ? |
# | Codebook loss | ? | ? |
# | Commitment loss | ? | ? |

# %%
torch.manual_seed(0)
quantizer = VectorQuantizer(
    codebook_size=4,
    embedding_dim=2,
    commitment_weight=0.25,
)
encoder_latent = torch.randn(1, 2, 2, 2, requires_grad=True)
straight_through, extras = quantizer(encoder_latent)
straight_through.square().mean().backward()
print("reconstruction path → encoder:", encoder_latent.grad.norm())
print("reconstruction path → codebook:", quantizer.codebook.weight.grad)

# %%
quantizer.zero_grad(set_to_none=True)
encoder_latent = torch.randn(1, 2, 2, 2, requires_grad=True)
_straight_through, extras = quantizer(encoder_latent)
extras["codebook_loss"].backward()
print("codebook loss → encoder:", encoder_latent.grad)
print("codebook loss → codebook:", quantizer.codebook.weight.grad.norm())

# %%
quantizer.zero_grad(set_to_none=True)
encoder_latent = torch.randn(1, 2, 2, 2, requires_grad=True)
_straight_through, extras = quantizer(encoder_latent)
extras["commitment_loss"].backward()
print("commitment loss → encoder:", encoder_latent.grad.norm())
print("commitment loss → codebook:", quantizer.codebook.weight.grad)

# %% [markdown]
# The straight-through estimator deliberately uses a biased surrogate gradient:
# the forward value is quantized, while the backward reconstruction gradient
# behaves as though quantization were the identity.

# %% [markdown]
# ## Train the image model
#
# Predict reconstruction quality, token-map structure, codes used, dead codes,
# and perplexity before running.

# %%
config = load_yaml(ROOT / "recipes/vqvae/vqvae-001-basic.yaml")
result = run_training(config, run_root=ROOT / "runs")
run_dir = result.run_dir
summary = load_run_summary(run_dir)
summary

# %%
model, resolved = load_trained_model(run_dir)
_train_loader, validation_loader, spec = build_dataloaders(
    resolved["dataset"], resolved["training"]
)
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
names = class_names(resolved["dataset"]["name"])
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
# ## Inspect the discrete representation

# %%
figure, axes = plt.subplots(2, 5, figsize=(12, 5))
for axis, token_map, label in zip(
    axes.flat,
    output.extras["indices"],
    labels,
    strict=True,
):
    axis.imshow(token_map, cmap="viridis")
    axis.set_title(names[int(label)])
    axis.axis("off")
figure.suptitle("Each 7×7 cell is one discrete code index")
figure.tight_layout()
figure

# %%
diagnostics = summary["codebook"]
print(
    "nominal codes:", diagnostics["codebook_size"],
    "used:", diagnostics["codes_used"],
    "dead:", diagnostics["dead_codes"],
    "perplexity:", diagnostics["perplexity"],
)

# %% [markdown]
# Perplexity is the effective vocabulary size implied by assignment entropy. It
# is neither the nominal codebook size nor a reconstruction-quality metric.
#
# ## Test the second sampling failure
#
# The decoder knows code meanings, but no model has learned which spatial token
# arrangements are likely.

# %%
torch.manual_seed(0)
height, width = output.extras["indices"].shape[1:]
random_indices = torch.randint(
    0, model.quantizer.codebook.num_embeddings, (16, height, width)
)
embeddings = model.quantizer.codebook(random_indices).permute(0, 3, 1, 2)
with torch.no_grad():
    random_images = model.decoder(embeddings)
plot_image_grid(
    random_images,
    title="Uniformly random token grids are not a learned prior",
    max_items=16,
)

# %% [markdown]
# ## Advancement gate
#
# Explain why argmin blocks ordinary gradients, why straight-through learning is
# biased, why codebook size differs from perplexity, and why a trained VQ-VAE is
# not yet a complete generative model.
