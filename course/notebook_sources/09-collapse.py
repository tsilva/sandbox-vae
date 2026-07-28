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
# # Lesson 09 — Posterior collapse, warm-up, and free bits
#
# **Learning objective:** diagnose posterior collapse from joint metric and
# reconstruction evidence, then understand two optimization interventions.
#
# Posterior collapse is a joint signature:
#
# - KL near zero,
# - few active dimensions,
# - similar posterior parameters across inputs,
# - and reconstructions that do not depend meaningfully on $z$.
#
# Low KL alone is not enough for the diagnosis.

# %%
import json
import subprocess
import sys

import matplotlib.pyplot as plt
from IPython.display import Image, display

from latent_lab.config import load_yaml
from latent_lab.course import load_metrics, repository_root

ROOT = repository_root()
study_path = ROOT / "studies/vae/vae-002-collapse-remedies.yaml"
study = load_yaml(study_path)

# %% [markdown]
# ## Predict the interventions
#
# **KL warm-up:** increase beta gradually so reconstruction can learn to use
# $z$ before full rate pressure arrives.
#
# **Free bits:** optimize a per-dimension floor
#
# $$
# \max(\lambda, KL_j)
# $$
#
# so KL below $\lambda$ supplies no additional compression gradient.
#
# Predict which intervention retains the most active dimensions and which gives
# the lowest distortion.

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
summary_path = sorted(
    (ROOT / "runs" / study["id"]).glob("study-summary-*.json")
)[-1]
summary = json.loads(summary_path.read_text())
observations = []
for record in summary["records"]:
    run_dir = ROOT / record["run_dir"]
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
    metrics = record["best_validation_metrics"]
    observations.append(
        {
            "variant": record["variant"],
            "reconstruction": metrics["validation/reconstruction_loss"],
            "raw_kl": metrics["validation/kl_loss"],
            "effective_kl": metrics["validation/effective_kl_loss"],
            "active": diagnostics["vae_latent"]["active_dimensions"],
            "run_dir": run_dir,
        }
    )
observations

# %% [markdown]
# Effective KL can exceed raw KL under free bits because the objective contains a
# constant floor. Diagnose actual information use from raw per-dimension KL.

# %%
variants = [item["variant"] for item in observations]
positions = range(len(variants))
figure, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(variants, [item["reconstruction"] for item in observations])
axes[0].set(title="Distortion", ylabel="Reconstruction BCE")
axes[1].bar(
    [position - 0.18 for position in positions],
    [item["raw_kl"] for item in observations],
    width=0.36,
    label="raw",
)
axes[1].bar(
    [position + 0.18 for position in positions],
    [item["effective_kl"] for item in observations],
    width=0.36,
    label="effective",
)
axes[1].set_xticks(list(positions), variants)
axes[1].set(title="Raw versus optimized KL", ylabel="Nats")
axes[1].legend()
axes[2].bar(variants, [item["active"] for item in observations])
axes[2].set(title="Active dimensions", ylabel="Count")
for axis in axes:
    axis.tick_params(axis="x", rotation=20)
figure.tight_layout()
figure

# %% [markdown]
# ## Inspect optimization dynamics, not only endpoints

# %%
for item in observations:
    records = load_metrics(item["run_dir"])
    epochs = [record["epoch"] for record in records]
    betas = [record["validation/beta"] for record in records]
    plt.plot(epochs, betas, label=item["variant"])
plt.xlabel("Epoch")
plt.ylabel("Effective beta")
plt.title("Warm-up changes when KL pressure arrives")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# %%
for item in observations:
    print(item["variant"])
    display(
        Image(
            filename=str(item["run_dir"] / "figures/reconstructions.png")
        )
    )
    display(
        Image(
            filename=str(item["run_dir"] / "figures/kl-per-dimension.png")
        )
    )

# %% [markdown]
# This small decoder may not collapse catastrophically. If the immediate-KL
# control still uses its latent, the remedies cannot honestly be credited with
# “fixing collapse.”
#
# ## Advancement gate
#
# Distinguish:
#
# - low KL with strong input-dependent reconstructions,
# - low KL with constant reconstructions,
# - high KL with good reconstruction but poor prior samples.
#
# Give each a different diagnosis.
