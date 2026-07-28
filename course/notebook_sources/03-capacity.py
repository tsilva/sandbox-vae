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
# # Lesson 03 — Nonlinearity and bottleneck capacity
#
# **Learning objective:** separate representational geometry from the amount of
# information allowed through the latent bottleneck.
#
# Two different changes are often called “more capacity”:
#
# 1. Nonlinear layers change the geometry the model can represent.
# 2. More latent coordinates increase the information that can cross the
#    bottleneck.
#
# These are separate scientific questions, so we probe them in separate studies.

# %%
import json
import subprocess
import sys

import matplotlib.pyplot as plt
from IPython.display import Image, display

from latent_lab.config import load_yaml
from latent_lab.course import repository_root

ROOT = repository_root()
nonlinearity_study = load_yaml(
    ROOT / "studies/ae/ae-002-nonlinearity.yaml"
)
capacity_study = load_yaml(
    ROOT / "studies/ae/ae-003-latent-capacity.yaml"
)
nonlinearity_study

# %% [markdown]
# ## Prediction A — same bottleneck, different geometry
#
# Predict the direction of validation MSE for the linear and nonlinear models.
# Then name the confound that remains even though latent dimension is controlled.
#
# > **Prediction:**

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(ROOT / "studies/ae/ae-002-nonlinearity.yaml"),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
summary_path = sorted(
    (ROOT / "runs" / nonlinearity_study["id"]).glob("study-summary-*.json")
)[-1]
summary = json.loads(summary_path.read_text())
for record in summary["records"]:
    run_summary = json.loads(
        (ROOT / record["run_dir"] / "summary.json").read_text()
    )
    print(
        record["variant"],
        "MSE=", f"{record['primary_metric_value']:.6f}",
        "parameters=", run_summary["parameter_count"],
    )
    display(
        Image(
            filename=str(
                ROOT / record["run_dir"] / "figures/reconstructions.png"
            )
        )
    )

# %% [markdown]
# A lower reconstruction loss establishes that the nonlinear model preserved
# more pixel information under this training setup. It does not establish that
# the representation is more disentangled, robust, or useful downstream.
#
# The comparison also changes parameter count. Record that limitation rather
# than silently calling the result “the effect of ReLU.”

# %% [markdown]
# ## Prediction B — bottleneck size
#
# Sketch validation MSE for latent sizes 2, 8, 32, and 128. Do you expect equal
# improvement from each step?
#
# > **Prediction:**

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(ROOT / "studies/ae/ae-003-latent-capacity.yaml"),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
capacity_summary_path = sorted(
    (ROOT / "runs" / capacity_study["id"]).glob("study-summary-*.json")
)[-1]
capacity_summary = json.loads(capacity_summary_path.read_text())
dimensions = [
    int(record["variant"].split("-")[-1])
    for record in capacity_summary["records"]
]
losses = [
    record["primary_metric_value"]
    for record in capacity_summary["records"]
]
figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(dimensions, losses, marker="o")
axis.set(
    xscale="log",
    xlabel="Latent dimensions",
    ylabel="Validation reconstruction MSE",
    title="Capacity helps reconstruction, usually with diminishing returns",
)
axis.grid(alpha=0.25)
figure.tight_layout()
figure

# %% [markdown]
# ## Deliberately challenge the objective
#
# Imagine a latent dimension of 784 with a sufficiently flexible encoder and
# decoder. Copying the input can become easier, but the bottleneck no longer
# forces compression. A reconstruction objective alone cannot tell whether that
# representation is useful for classification, robustness, or generation.
#
# ## Advancement gate
#
# Explain why “bigger latent is better” is incomplete. Mention:
#
# - the optimized task,
# - compression,
# - parameter-count confounding,
# - downstream usefulness,
# - and the identity-mapping failure mode.
