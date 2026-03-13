from __future__ import annotations

from typing import Any

import torch

from sandbox_autoencoders.training.losses import kl_per_dim


def active_latent_count(latents: torch.Tensor, threshold: float = 0.01) -> int:
    return int((latents.var(dim=0) > threshold).sum().item())


def summarize_latents(mu: torch.Tensor | None, logvar: torch.Tensor | None, latent: torch.Tensor) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "active_latents": active_latent_count(latent),
        "latent_mean_abs": float(latent.abs().mean().item()),
    }
    if mu is not None and logvar is not None:
        kl_dims = kl_per_dim(mu, logvar)
        covariance = torch.cov(mu.T) if mu.shape[0] > 1 else torch.eye(mu.shape[1], device=mu.device)
        summary.update(
            {
                "mu_abs": float(mu.abs().mean().item()),
                "logvar_abs": float(logvar.abs().mean().item()),
                "per_dim_mu": mu.mean(dim=0).detach().cpu().tolist(),
                "per_dim_logvar": logvar.mean(dim=0).detach().cpu().tolist(),
                "kl_per_dim": kl_dims.mean(dim=0).detach().cpu().tolist(),
                "posterior_mean": mu.mean(dim=0).detach().cpu().tolist(),
                "posterior_variance": mu.var(dim=0).detach().cpu().tolist(),
                "covariance_trace": float(torch.trace(covariance).item()),
            }
        )
    return summary


def nearest_neighbor_indices(samples: torch.Tensor, reference: torch.Tensor) -> list[int]:
    sample_flat = samples.flatten(start_dim=1)
    ref_flat = reference.flatten(start_dim=1)
    distances = torch.cdist(sample_flat, ref_flat)
    return distances.argmin(dim=1).detach().cpu().tolist()


def factor_predictability(latents: torch.Tensor, factors: torch.Tensor) -> dict[str, float]:
    if factors.numel() == 0:
        return {}
    latents = latents - latents.mean(dim=0, keepdim=True)
    factors = factors - factors.mean(dim=0, keepdim=True)
    numerator = torch.einsum("nd,nf->df", latents, factors).abs()
    denom = latents.std(dim=0).unsqueeze(1) * factors.std(dim=0).unsqueeze(0) * max(1, latents.shape[0])
    corr = torch.nan_to_num(numerator / (denom + 1e-8), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mig_proxy": float(corr.max(dim=0).values.mean().item()),
        "max_factor_correlation": float(corr.max().item()),
    }
