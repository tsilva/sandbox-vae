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
# # Lesson 04 — Denoising and sparse autoencoders
#
# **Learning objective:** see how the training task and regularization pressure
# change what a latent representation is rewarded for preserving.
#
# A bottleneck is defined by more than its coordinate count. Changing the
# training task or adding pressure to latent activity changes what information
# the representation is rewarded for preserving.

# %%
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import torch
from IPython.display import Image, display

from latent_lab.config import load_yaml
from latent_lab.course import balanced_class_batch, repository_root
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid
from latent_lab.training import corrupt_inputs

ROOT = repository_root()
denoising_recipe = load_yaml(ROOT / "recipes/ae/ae-003-denoising.yaml")
train_loader, validation_loader, spec = build_dataloaders(
    denoising_recipe["dataset"], denoising_recipe["training"]
)
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
names = class_names(denoising_recipe["dataset"]["name"])

# %% [markdown]
# ## Poke the denoising task before training
#
# The model receives $\tilde{x}$ but the objective compares its output with
# clean $x$:
#
# $$
# L=\lVert g(f(\tilde{x}))-x\rVert^2
# $$
#
# Predict whether this training task should improve corrupted-input output and
# whether it must also improve clean-input MSE.

# %%
torch.manual_seed(0)
corrupted = corrupt_inputs(
    inputs,
    denoising_recipe["training"]["input_corruption"],
)
plot_image_grid(
    inputs,
    labels=labels,
    class_names=names,
    title="Clean targets",
)
plot_image_grid(
    corrupted,
    labels=labels,
    class_names=names,
    title="Inputs seen by the denoising model",
)

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(ROOT / "studies/ae/ae-004-denoising.yaml"),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
denoising_study = load_yaml(ROOT / "studies/ae/ae-004-denoising.yaml")
denoising_summary_path = sorted(
    (ROOT / "runs" / denoising_study["id"]).glob("study-summary-*.json")
)[-1]
denoising_summary = json.loads(denoising_summary_path.read_text())
for record in denoising_summary["records"]:
    print(record["variant"], record["best_validation_metrics"])
    figures = ROOT / record["run_dir"] / "figures"
    display(Image(filename=str(figures / "reconstructions.png")))
    noisy_figure = figures / "corrupted-input-reconstructions.png"
    if noisy_figure.exists():
        display(Image(filename=str(noisy_figure)))

# %% [markdown]
# Clean validation MSE is not the direct metric for the denoising claim. The
# aligned corrupted-input figure is essential evidence. This is an example of a
# metric that is valid but incomplete for the scientific question.

# %% [markdown]
# ## Sparse activity is a different bottleneck
#
# $$
# L=L_{\text{reconstruction}}+
# \lambda\operatorname{mean}(|z|)
# $$
#
# Predict reconstruction MSE and mean absolute latent activation as $\lambda$
# increases. Which should move first?

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(ROOT / "studies/ae/ae-005-sparsity.yaml"),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
sparsity_study = load_yaml(ROOT / "studies/ae/ae-005-sparsity.yaml")
sparsity_summary_path = sorted(
    (ROOT / "runs" / sparsity_study["id"]).glob("study-summary-*.json")
)[-1]
sparsity_summary = json.loads(sparsity_summary_path.read_text())
weights = []
reconstruction = []
activation = []
for record in sparsity_summary["records"]:
    weights.append(float(record["variant"].replace("l1-", "")))
    metrics = record["best_validation_metrics"]
    reconstruction.append(metrics["validation/reconstruction_loss"])
    activation.append(metrics["validation/latent_l1"])

figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(weights, reconstruction, marker="o")
axes[0].set(xscale="symlog", xlabel="L1 weight", ylabel="MSE")
axes[1].plot(weights, activation, marker="o")
axes[1].set(xscale="symlog", xlabel="L1 weight", ylabel="Mean |z|")
figure.suptitle("Raw terms reveal the tradeoff hidden by total loss")
figure.tight_layout()
figure

# %% [markdown]
# ## Advancement gate
#
# Contrast:
#
# - a low-dimensional latent,
# - corrupted input with a clean target,
# - and L1 pressure on latent activity.
#
# State what behavior each encourages and why the three interventions are not
# interchangeable.
