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
# # Lesson 01 — Why a reconstruction loss needs a baseline
#
# **Learning objective:** make the number called “MSE” interpretable by comparing
# it with predictions that learn no input-dependent representation.
#
# For image $x$ and reconstruction $\hat{x}$:
#
# $$
# \operatorname{MSE}(x,\hat{x})
# = \frac{1}{CHW}\sum_{c,h,w}(x_{chw}-\hat{x}_{chw})^2
# $$

# %%
import matplotlib.pyplot as plt
import torch

from latent_lab.config import load_yaml
from latent_lab.course import balanced_class_batch, repository_root
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.training import (
    compute_mean_image,
    constant_reconstruction_errors,
)

ROOT = repository_root()
config = load_yaml(ROOT / "recipes/ae/ae-001-linear.yaml")
train_loader, validation_loader, spec = build_dataloaders(
    config["dataset"], config["training"]
)
names = class_names(config["dataset"]["name"])
spec

# %% [markdown]
# ## First look at the dataset
#
# Before discussing loss, establish the data contract:
#
# - One grayscale channel
# - $28\times28$ pixels
# - Pixel values in $[0,1]$
# - Ten garment classes
#
# A class-balanced view prevents the first shuffled batch from defining your
# mental picture of the dataset.

# %%
class_images, class_labels = balanced_class_batch(
    validation_loader, spec.num_classes
)
print(
    "shape:",
    tuple(class_images.shape),
    "range:",
    (float(class_images.min()), float(class_images.max())),
)
plot_image_grid(
    class_images,
    labels=class_labels,
    class_names=names,
    title="One validation example from each Fashion-MNIST class",
)

# %% [markdown]
# ## Predict before revealing the mean
#
# Write down:
#
# 1. Which shared shapes will survive averaging 60,000 garments?
# 2. Will any individual class remain recognizable?
# 3. Why might the result still receive a nonterrible MSE?
#
# > **Prediction:**

# %%
device = torch.device("cpu")
mean_image, train_examples = compute_mean_image(train_loader, device)
print("training examples averaged:", train_examples)
plot_image_grid(mean_image, title="Pixelwise mean training image", max_items=1)

# %% [markdown]
# ## The input-independent reconstruction
#
# Under squared error, the best constant prediction is the training-set mean.
# “Best constant” is the important qualifier: it says nothing about encoding
# the input.

# %%
constant_predictions = mean_image.expand(class_images.shape[0], -1, -1, -1)
plot_reconstruction_grid(
    class_images,
    constant_predictions,
    labels=class_labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %% [markdown]
# Every reconstruction row is identical. If you can infer the input class from
# a prediction, you are using information that the predictor itself did not use.
#
# Inspect the squared-error row. Background pixels dominate the image and are
# predicted well; errors concentrate around class-specific silhouettes and
# details.

# %%
mean_errors = constant_reconstruction_errors(
    validation_loader, mean_image, device
)
black_image = torch.zeros_like(mean_image)
black_errors = constant_reconstruction_errors(
    validation_loader, black_image, device
)

print(f"all-black validation MSE: {black_errors.mean():.6f}")
print(f"mean-image validation MSE: {mean_errors.mean():.6f}")
print(
    "relative improvement over black:",
    f"{(1 - mean_errors.mean() / black_errors.mean()) * 100:.1f}%",
)

# %% [markdown]
# ## Do not let the average hide the distribution
#
# One mean MSE conceals easy and hard examples. Examine the distribution and
# class-conditioned errors before treating it as a complete description.

# %%
validation_labels = torch.cat(
    [labels for _inputs, labels in validation_loader]
)
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(mean_errors.numpy(), bins=40)
axes[0].axvline(float(mean_errors.mean()), color="red", linestyle="--")
axes[0].set(
    title="Per-example mean-image error",
    xlabel="MSE",
    ylabel="Validation examples",
)
class_mse = [
    float(mean_errors[validation_labels == label].mean())
    for label in range(spec.num_classes)
]
axes[1].barh(names, class_mse)
axes[1].set(title="The same baseline is not equally good for every class", xlabel="MSE")
figure.tight_layout()
figure

# %% [markdown]
# ## What you should have learned
#
# The number now has a reference:
#
# - An all-black predictor exploits background sparsity.
# - The mean image is the MSE-optimal predictor among all constant images.
# - Neither predictor contains an encoder or an input-dependent representation.
# - A trained autoencoder must beat these baselines **and** visibly change its
#   output with the input.
#
# ## Advancement gate
#
# Explain why “validation MSE = 0.05” is incomplete without the dataset, pixel
# range, reduction convention, error distribution, and baseline. Then explain
# why beating the mean baseline is necessary but still not sufficient evidence
# of a useful representation.
