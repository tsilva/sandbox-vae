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
# # Lesson 00 — Trust the laboratory before trusting a result
#
# **Source of truth:** `course/notebook_sources/00-laboratory.py`
#
# This generated notebook is an executable lab. Edit the Jupytext source, then
# rebuild notebooks with `uv run python scripts/build_notebooks.py`.
#
# ## Learning objective
#
# Establish the repository contract: every model must produce a reconstruction,
# a latent representation, and model-specific extras; every objective must be
# finite and backpropagate before a long experiment is worth running.

# %%
from pathlib import Path

import torch

from latent_lab.config import load_yaml
from latent_lab.course import repository_root
from latent_lab.data.datasets import dataset_spec
from latent_lab.models import build_model
from latent_lab.objectives import build_objective

ROOT = repository_root()
torch.manual_seed(0)
ROOT

# %% [markdown]
# ## Predict before running
#
# Why should the trainer receive one common output structure instead of knowing
# separate AE, VAE, and VQ-VAE calling conventions?
#
# Write your answer here before inspecting the next cell:
#
# > **Prediction:**

# %%
recipe_paths = [
    ROOT / "recipes/smoke/fake-ae.yaml",
    ROOT / "recipes/smoke/fake-vae.yaml",
    ROOT / "recipes/smoke/fake-vqvae.yaml",
]

contracts = []
for recipe_path in recipe_paths:
    config = load_yaml(recipe_path)
    spec = dataset_spec(config["dataset"])
    model = build_model(config["model"], spec)
    objective = build_objective(config["objective"])
    inputs = torch.rand(4, *spec.input_shape)
    output = model(inputs)
    losses = objective(output, inputs)
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    gradient_norm = sum(
        float(parameter.grad.norm())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    contracts.append(
        {
            "model": config["model"]["kind"],
            "input": tuple(inputs.shape),
            "reconstruction": tuple(output.reconstruction.shape),
            "latent": tuple(output.latent.shape),
            "extras": sorted(output.extras),
            "losses": sorted(losses),
            "gradient_norm": gradient_norm,
        }
    )

contracts

# %% [markdown]
# ## Interrogate the evidence
#
# For each row, check:
#
# - Reconstruction shape exactly matches input shape.
# - Latent shape reflects the model family.
# - Extras expose only model-specific information.
# - The gradient norm is finite and nonzero.
#
# Passing these checks proves wiring, not learning. Random synthetic images have
# no useful garment structure, and a finite loss does not show that a model
# learned an input-dependent representation.

# %%
assert all(item["input"] == item["reconstruction"] for item in contracts)
assert all(item["gradient_norm"] > 0 for item in contracts)
print("Shape and gradient contracts passed.")

# %% [markdown]
# ## Repository-level trust check
#
# Run the complete fast test suite in a terminal:
#
# ```bash
# uv sync --locked
# uv run pytest
# ```
#
# Then run the smoke recipes if you want to inspect the durable run contract:
#
# ```bash
# uv run latent-lab train recipes/smoke/fake-ae.yaml --device cpu
# uv run latent-lab train recipes/smoke/fake-vae.yaml --device cpu
# uv run latent-lab train recipes/smoke/fake-vqvae.yaml --device cpu
# ```
#
# ## Advancement gate
#
# Explain, without reading code:
#
# 1. What a recipe controls.
# 2. Why a smoke test is weaker than a learning result.
# 3. Why total losses from different model families need not be comparable.
# 4. Why reusable model and training logic belongs in `src/latent_lab`, not here.
