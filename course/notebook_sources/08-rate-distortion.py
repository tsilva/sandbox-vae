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
# # Lesson 08 — The VAE rate–distortion experiment
#
# **Learning objective:** observe directly how beta prices latent information
# against reconstruction distortion.
#
# Beta is not a generic “regularization strength.” It sets the exchange rate
# between reconstruction distortion and latent information.

# %%
import json
import subprocess
import sys

import matplotlib.pyplot as plt
from IPython.display import Image, display

from latent_lab.config import load_yaml
from latent_lab.course import repository_root

ROOT = repository_root()
study_path = ROOT / "studies/vae/vae-001-beta-sweep.yaml"
study = load_yaml(study_path)
study

# %% [markdown]
# ## Predict four regimes
#
# Fill this table before running:
#
# | Beta | Reconstruction | Raw KL | Active dimensions | Prior samples |
# |---:|---|---|---|---|
# | 0 | | | | |
# | 0.1 | | | | |
# | 1 | | | | |
# | 4 | | | | |
#
# For each prediction, state the causal path from beta to encoder behavior.

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(study_path),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
study_summary_path = sorted(
    (ROOT / "runs" / study["id"]).glob("study-summary-*.json")
)[-1]
study_summary = json.loads(study_summary_path.read_text())
observations = []
for record in study_summary["records"]:
    run_dir = ROOT / record["run_dir"]
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
    metrics = record["best_validation_metrics"]
    observations.append(
        {
            "variant": record["variant"],
            "beta": metrics["validation/beta"],
            "distortion": metrics["validation/reconstruction_loss"],
            "rate": metrics["validation/kl_loss"],
            "active_dimensions": diagnostics["vae_latent"][
                "active_dimensions"
            ],
            "run_dir": run_dir,
        }
    )
observations

# %% [markdown]
# ## Build the rate–distortion view
#
# Moving down means better reconstruction. Moving left means using fewer nats.
# Neither axis alone defines a universally best model.

# %%
figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for item in observations:
    axes[0].scatter(item["rate"], item["distortion"], s=70)
    axes[0].annotate(
        f"β={item['beta']:g}",
        (item["rate"], item["distortion"]),
        xytext=(5, 5),
        textcoords="offset points",
    )
axes[0].set(
    xlabel="Rate: raw KL (nats/example)",
    ylabel="Distortion: reconstruction BCE/example",
    title="Observed rate–distortion frontier",
)
axes[0].grid(alpha=0.25)
axes[1].plot(
    [item["beta"] for item in observations],
    [item["active_dimensions"] for item in observations],
    marker="o",
)
axes[1].set(
    xscale="symlog",
    xlabel="Beta",
    ylabel="Active latent dimensions",
    title="Information can disappear dimension by dimension",
)
axes[1].grid(alpha=0.25)
figure.tight_layout()
figure

# %% [markdown]
# ## Images arbitrate ambiguous metrics
#
# Compare prior samples in beta order. Low rate is useful only when the decoder
# still receives enough input-dependent information to model meaningful
# variation.

# %%
for item in observations:
    print(f"beta={item['beta']:g}")
    display(
        Image(
            filename=str(
                item["run_dir"] / "figures/random-latent-samples.png"
            )
        )
    )

# %% [markdown]
# ## Common mistakes
#
# - Comparing total objective values even though beta changed the objective.
# - Calling low KL “good regularization” without reconstruction evidence.
# - Calling high KL “expressive” without inspecting prior mismatch.
# - Choosing a beta from one attractive image.
#
# ## Advancement gate
#
# Diagnose a run with excellent reconstruction, high KL, and poor prior samples
# using rate, distortion, and aggregate-posterior mismatch—without saying only
# “overfitting.”
