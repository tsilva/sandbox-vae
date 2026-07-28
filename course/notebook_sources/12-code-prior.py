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
# # Lesson 12 — Turn VQ-VAE tokens into a generative model
#
# **Learning objective:** separate representation learning from modeling a
# probability distribution over valid discrete token arrangements.
#
# Representation learning produces discrete token grids:
#
# $$
# x\rightarrow z_e\rightarrow k_{1:H,1:W}\rightarrow\hat{x}
# $$
#
# A separate prior learns which arrangements are likely:
#
# $$
# p(k_1,\ldots,k_N)=\prod_{i=1}^{N}p(k_i\mid k_{<i})
# $$

# %%
import math

from IPython.display import Image, display

from latent_lab.config import load_yaml
from latent_lab.course import (
    latest_run_dir,
    load_metrics,
    load_run_summary,
    plot_metric_history,
    repository_root,
)
from latent_lab.training import run_code_prior_training

ROOT = repository_root()
vq_run_dir = latest_run_dir(ROOT / "runs", "vqvae/vqvae-001-basic")
vq_checkpoint = vq_run_dir / "checkpoint-best.pt"
vq_summary = load_run_summary(vq_run_dir)
print("chosen VQ-VAE:", vq_run_dir)
display(
    Image(
        filename=str(
            vq_run_dir / "figures/uniform-random-token-samples.png"
        )
    )
)

# %% [markdown]
# Uniform random tokens ask the wrong question: “What if every code at every
# position were independent and equally likely?” The learned prior instead
# models token arrangements emitted by the frozen encoder.
#
# ## Predict before training
#
# Compare initial next-token cross-entropy with the uniform baseline $\log K$.
# Predict validation perplexity and how learned-prior samples will differ from
# uniform-token samples.

# %%
codebook_size = vq_summary["codebook"]["codebook_size"]
print("uniform-prior cross-entropy log(K):", math.log(codebook_size))
print("uniform-prior perplexity K:", codebook_size)

# %%
prior_config = load_yaml(ROOT / "recipes/prior/prior-001-gru.yaml")
result = run_code_prior_training(
    prior_config,
    vq_checkpoint,
    run_root=ROOT / "runs",
)
prior_run_dir = result.run_dir
load_run_summary(prior_run_dir)

# %%
plot_metric_history(
    load_metrics(prior_run_dir),
    ["train/cross_entropy", "validation/cross_entropy"],
    title="Teacher-forced autoregressive token prediction",
)

# %% [markdown]
# ## Compare the two sampling distributions

# %%
print("uniform independent tokens")
display(
    Image(
        filename=str(
            vq_run_dir / "figures/uniform-random-token-samples.png"
        )
    )
)
print("learned autoregressive prior")
display(
    Image(
        filename=str(
            prior_run_dir / "figures/learned-prior-samples.png"
        )
    )
)

# %% [markdown]
# Prior perplexity is uncertainty in the *next token conditioned on previous
# tokens*. Codebook perplexity is diversity of encoder assignments. They answer
# different questions despite sharing a name.
#
# The prior cannot restore visual information already discarded by the frozen
# VQ-VAE. It only models the distribution over the tokens that remain.
#
# ## Advancement gate
#
# Explain why the encoder and decoder can stay frozen, why teacher forcing
# differs from sampling, and why improved prior likelihood cannot repair a
# lossy representation.
