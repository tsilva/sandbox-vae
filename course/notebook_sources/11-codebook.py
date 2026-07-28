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
# # Lesson 11 — Codebook capacity, utilization, and commitment
#
# **Learning objective:** distinguish nominal codebook capacity from effective
# usage and observe how commitment pressure changes the channel.
#
# Treat the codebook as a learned communication channel. Nominal vocabulary
# size does not guarantee that optimization uses every symbol—or uses symbols
# evenly.

# %%
import json
import subprocess
import sys

import matplotlib.pyplot as plt

from latent_lab.config import load_yaml
from latent_lab.course import repository_root

ROOT = repository_root()
size_path = ROOT / "studies/vqvae/vqvae-001-codebook-size.yaml"
commitment_path = ROOT / "studies/vqvae/vqvae-002-commitment.yaml"

# %% [markdown]
# ## Prediction A — larger vocabularies
#
# Predict reconstruction MSE, codes used, dead codes, and perplexity for
# $K\in\{8,32,128,512\}$. Distinguish nominal capacity from effective usage.
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Small codebooks should use most entries and may constrain reconstruction.
# Larger codebooks can lower MSE, but utilization and perplexity need not grow
# proportionally; dead-code count should generally rise as nominal capacity
# exceeds what optimization uses.
# </details>

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(size_path),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
size_study = load_yaml(size_path)
size_summary_path = sorted(
    (ROOT / "runs" / size_study["id"]).glob("study-summary-*.json")
)[-1]
size_summary = json.loads(size_summary_path.read_text())
size_observations = []
for record in size_summary["records"]:
    run_dir = ROOT / record["run_dir"]
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())[
        "codebook"
    ]
    size_observations.append(
        {
            "size": diagnostics["codebook_size"],
            "used": diagnostics["codes_used"],
            "dead": diagnostics["dead_codes"],
            "perplexity": diagnostics["perplexity"],
            "mse": record["best_validation_metrics"][
                "validation/reconstruction_loss"
            ],
        }
    )
size_observations

# %%
figure, axes = plt.subplots(1, 3, figsize=(14, 4))
sizes = [item["size"] for item in size_observations]
axes[0].plot(
    sizes, [item["mse"] for item in size_observations], marker="o"
)
axes[0].set(xscale="log", xlabel="Codebook size", ylabel="MSE")
axes[1].plot(
    sizes, [item["used"] for item in size_observations], marker="o", label="used"
)
axes[1].plot(
    sizes,
    [item["perplexity"] for item in size_observations],
    marker="o",
    label="perplexity",
)
axes[1].plot(sizes, sizes, linestyle="--", alpha=0.5, label="nominal")
axes[1].set(xscale="log", yscale="log", xlabel="Codebook size")
axes[1].legend()
axes[2].plot(
    sizes, [item["dead"] for item in size_observations], marker="o"
)
axes[2].set(xscale="log", xlabel="Codebook size", ylabel="Dead codes")
figure.suptitle("More entries do not imply more effective symbols")
figure.tight_layout()

# %% [markdown]
# ## Prediction B — commitment pressure
#
# Too little commitment allows encoder outputs to drift from embeddings; too
# much can constrain reconstruction. Predict raw and weighted commitment terms,
# reconstruction, and usage before running.
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Increasing the weight should pull encoder outputs closer to selected
# embeddings, reducing raw commitment distance after adaptation while
# increasing its optimization importance. Very low pressure may destabilize the
# discrete interface; very high pressure may damage reconstruction or usage.
# The best tradeoff need not sit at an endpoint.
# </details>

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "latent_lab.cli",
        "study",
        str(commitment_path),
        "--seeds",
        "0",
    ],
    cwd=ROOT,
    check=True,
)

# %%
commitment_study = load_yaml(commitment_path)
commitment_summary_path = sorted(
    (ROOT / "runs" / commitment_study["id"]).glob("study-summary-*.json")
)[-1]
commitment_summary = json.loads(commitment_summary_path.read_text())
commitment_observations = []
for record in commitment_summary["records"]:
    metrics = record["best_validation_metrics"]
    commitment_observations.append(
        {
            "weight": float(record["variant"].replace("commitment-", "")),
            "mse": metrics["validation/reconstruction_loss"],
            "raw": metrics["validation/commitment_loss"],
            "weighted": metrics["validation/weighted_commitment_loss"],
            "codebook": metrics["validation/codebook_loss"],
            "perplexity": metrics["validation/codebook_perplexity"],
        }
    )
commitment_observations

# %% [markdown]
# Total loss cannot reveal which component improved. Compare raw commitment
# distance with its weighted contribution, and inspect reconstruction and usage
# separately.

# %%
weights = [item["weight"] for item in commitment_observations]
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
for metric in ("raw", "weighted", "codebook"):
    axes[0].plot(
        weights,
        [item[metric] for item in commitment_observations],
        marker="o",
        label=metric,
    )
axes[0].set(xlabel="Commitment weight", ylabel="Loss component")
axes[0].legend()
axes[1].plot(
    weights,
    [item["mse"] for item in commitment_observations],
    marker="o",
    label="reconstruction MSE",
)
axes[1].plot(
    weights,
    [item["perplexity"] for item in commitment_observations],
    marker="o",
    label="perplexity",
)
axes[1].set(xlabel="Commitment weight")
axes[1].legend()
figure.tight_layout()

# %% [markdown]
# ## Advancement gate
#
# Given $K=512$, 40 used codes, and perplexity 9, explain all three numbers and
# why none alone establishes representation quality.
