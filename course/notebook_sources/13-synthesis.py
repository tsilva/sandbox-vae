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
# # Lesson 13 — Final comparison and teach-back
#
# **Learning objective:** choose AE, VAE, or VQ-VAE from the required latent
# semantics, evidence, and known failure modes.
#
# Choose a model family from requirements and failure modes, not from one
# universal leaderboard.

# %%
from IPython.display import Image, display

from latent_lab.course import (
    latest_run_dir,
    load_run_summary,
    repository_root,
)

ROOT = repository_root()
finalists = {
    "AE": latest_run_dir(
        ROOT / "runs", "ae/ae-002-nonlinearity/nonlinear"
    ),
    "VAE": latest_run_dir(ROOT / "runs", "vae/vae-001-basic"),
    "VQ-VAE": latest_run_dir(ROOT / "runs", "vqvae/vqvae-001-basic"),
    "token prior": latest_run_dir(ROOT / "runs", "prior/prior-001-gru"),
}
finalists

# %% [markdown]
# ## Verify provenance before comparing
#
# Every metric and image must come from the same exact run as its resolved
# configuration and checkpoint. Do not mix a final checkpoint from one variant
# with diagnostics from another.

# %%
summaries = {
    family: load_run_summary(run_dir)
    for family, run_dir in finalists.items()
}
summaries

# %% [markdown]
# ## Build the comparison from mechanisms
#
# Mentally reconstruct this table before reading your old notes:
#
# | Question | AE | VAE | VQ-VAE |
# |---|---|---|---|
# | Latent type | | | |
# | Bottleneck mechanism | | | |
# | Reconstruction evidence | | | |
# | Utilization diagnostic | | | |
# | Can sample directly? | | | |
# | Additional prior required? | | | |
# | Main collapse mode | | | |
# | Best use case | | | |
#
# <details>
# <summary>Reveal a compact comparison</summary>
#
# | Question | AE | VAE | VQ-VAE |
# |---|---|---|---|
# | Latent type | deterministic continuous | stochastic continuous | spatial discrete indices |
# | Bottleneck mechanism | dimension/task/regularization | KL-priced information rate | nearest codebook entry |
# | Utilization diagnostic | latent activity/geometry | per-dimension raw KL | usage, dead codes, perplexity |
# | Can sample directly? | no known prior | approximately from $N(0,I)$ | not coherently from uniform codes |
# | Additional prior required? | yes for generation | usually no | yes, over token grids |
# | Main collapse mode | identity or constant mapping | posterior collapse | codebook collapse/dead codes |
# | Best use case | reconstruction/denoising | continuous probabilistic latents | discrete visual tokens |
# </details>
#
# Total objectives are not comparable across these families. Observation models,
# reductions, and auxiliary loss terms differ.

# %%
for family in ("AE", "VAE", "VQ-VAE"):
    run_dir = finalists[family]
    print(family, run_dir)
    display(Image(filename=str(run_dir / "figures/reconstructions.png")))

# %% [markdown]
# ## Match requirements to evidence
#
# Answer before revealing your prior conclusions:
#
# 1. Which model would you choose for denoising?
# 2. Which supports a deliberately regularized continuous sampling prior?
# 3. Which learns reusable discrete visual tokens?
# 4. Which extra model makes VQ-VAE samples coherent?
# 5. Which first failure metric or figure would you inspect for each?
#
# <details>
# <summary>Reveal one defensible mapping</summary>
#
# 1. A denoising AE for explicit corrupted-input to clean-target reconstruction.
# 2. A VAE for a regularized continuous prior.
# 3. A VQ-VAE for discrete visual tokens.
# 4. An autoregressive token prior for coherent VQ-VAE token arrangements.
# 5. AE: aligned reconstructions and baseline-relative MSE. VAE: reconstruction
#    plus raw/per-dimension KL and prior samples. VQ-VAE: reconstruction,
#    codebook usage/perplexity, token maps, and learned-prior samples.
# </details>

# %% [markdown]
# ## Confirmation discipline
#
# One run is enough to understand mechanics. A close ranking requires repeated
# seeds. Repeat only the finalists whose ordering would change your conclusion,
# then report mean, variation, qualitative consistency, and seed-specific
# failures.
#
# ## Ten-minute teach-back
#
# Explain without notes:
#
# 1. Why a loss needs an input-independent baseline.
# 2. Why an AE reconstructs but lacks a known prior.
# 3. How reparameterization enables VAE gradients.
# 4. Why KL creates a rate–distortion tradeoff.
# 5. How posterior collapse appears in numbers and images.
# 6. How VQ nearest-neighbor selection becomes trainable.
# 7. Why nominal codebook size differs from effective usage.
# 8. Why a VQ-VAE needs a token prior.
#
# ## Completion criterion
#
# You are done when you can predict the major metric and visual changes before a
# beta, bottleneck, or codebook ablation—and respond to a contradictory result
# with one controlled next experiment.
