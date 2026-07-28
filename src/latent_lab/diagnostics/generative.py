from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as pyplot
import torch
from torch import Tensor, nn
from torchvision.utils import make_grid

from latent_lab.diagnostics.codebook import code_usage
from latent_lab.diagnostics.interpolations import linear_interpolation


def save_image_grid(
    images: Tensor,
    path: Path,
    *,
    title: str,
    nrow: int,
) -> None:
    grid = make_grid(images.detach().cpu().clamp(0, 1), nrow=nrow, padding=2)
    rendered = grid.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pyplot.subplots(figsize=(max(6, nrow * 1.5), 4))
    axis.imshow(
        rendered.squeeze(-1) if rendered.shape[-1] == 1 else rendered
    )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    pyplot.close(figure)


def _save_latent_scatter(
    latent: Tensor,
    labels: Tensor,
    path: Path,
) -> None:
    latent = latent.float().cpu()
    labels = labels.cpu()
    centered = latent - latent.mean(dim=0, keepdim=True)
    if centered.shape[1] == 1:
        coordinates = torch.cat(
            [centered, torch.zeros_like(centered)], dim=1
        )
        axes_label = ("latent 0", "zero")
    elif centered.shape[1] == 2:
        coordinates = centered
        axes_label = ("latent 0", "latent 1")
    else:
        _u, _s, vectors = torch.pca_lowrank(centered, q=2)
        coordinates = centered @ vectors[:, :2]
        axes_label = ("principal component 1", "principal component 2")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pyplot.subplots(figsize=(7, 6))
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.65,
    )
    axis.set_xlabel(axes_label[0])
    axis.set_ylabel(axes_label[1])
    axis.set_title("Validation latent space (color = class label)")
    figure.colorbar(scatter, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    pyplot.close(figure)


def _save_kl_diagnostics(
    per_dimension_kl: Tensor,
    path: Path,
    active_threshold: float,
) -> dict[str, Any]:
    values = per_dimension_kl.detach().cpu()
    active = values > active_threshold
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pyplot.subplots(figsize=(8, 4))
    axis.bar(torch.arange(values.numel()).numpy(), values.numpy())
    axis.axhline(
        active_threshold,
        color="red",
        linestyle="--",
        label=f"active threshold = {active_threshold:g} nat",
    )
    axis.set_xlabel("Latent dimension")
    axis.set_ylabel("Mean KL (nats)")
    axis.set_title("Information carried by each VAE latent dimension")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    pyplot.close(figure)
    return {
        "kl_per_dimension": values.tolist(),
        "active_threshold": active_threshold,
        "active_dimensions": int(active.sum()),
        "total_dimensions": values.numel(),
        "total_kl": float(values.sum()),
    }


def _save_codebook_diagnostics(
    indices: Tensor,
    codebook_size: int,
    path: Path,
) -> dict[str, Any]:
    usage = code_usage(indices, codebook_size)
    counts = usage["counts"].cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pyplot.subplots(figsize=(10, 4))
    axis.bar(torch.arange(codebook_size).numpy(), counts.numpy())
    axis.set_xlabel("Code index")
    axis.set_ylabel("Validation assignments")
    axis.set_title("VQ-VAE codebook usage")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    pyplot.close(figure)
    return {
        "codebook_size": codebook_size,
        "codes_used": int(usage["codes_used"]),
        "dead_codes": int(codebook_size - int(usage["codes_used"])),
        "perplexity": float(usage["perplexity"]),
        "counts": counts.tolist(),
    }


def _save_token_maps(indices: Tensor, path: Path) -> None:
    count = min(8, indices.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = pyplot.subplots(1, count, figsize=(2 * count, 2.5))
    if count == 1:
        axes = [axes]
    for axis, token_map in zip(axes, indices[:count], strict=True):
        axis.imshow(token_map.detach().cpu(), cmap="viridis")
        axis.axis("off")
    figure.suptitle("Spatial discrete token maps")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    pyplot.close(figure)


@torch.no_grad()
def generate_generative_diagnostics(
    model: nn.Module,
    model_kind: str,
    validation_loader,
    fixed_inputs: Tensor,
    run_dir: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    figures = run_dir / "figures"
    diagnostics_config = config.get("diagnostics", {})
    max_points = int(diagnostics_config.get("max_latent_points", 2000))
    summary: dict[str, Any] = {}

    fixed_output = model(fixed_inputs)
    if model_kind in {"ae", "vae"}:
        latent = (
            fixed_output.extras["mu"]
            if model_kind == "vae"
            else fixed_output.latent
        )
        if latent.shape[0] >= 2:
            interpolated = linear_interpolation(latent[0], latent[1], 11)
            decoded = model.decode(interpolated)
            save_image_grid(
                decoded,
                figures / "interpolation.png",
                title="Decode a straight line between two encoded examples",
                nrow=11,
            )

        random_latent = torch.randn(16, latent.shape[1], device=device)
        random_decoded = model.decode(random_latent)
        title = (
            "Samples from N(0, I): meaningful only if the latent matches this prior"
        )
        save_image_grid(
            random_decoded,
            figures / "random-latent-samples.png",
            title=title,
            nrow=8,
        )

        latent_batches: list[Tensor] = []
        label_batches: list[Tensor] = []
        mu_batches: list[Tensor] = []
        logvar_batches: list[Tensor] = []
        examples = 0
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            representation = (
                output.extras["mu"] if model_kind == "vae" else output.latent
            )
            latent_batches.append(representation.detach().cpu())
            label_batches.append(labels.detach().cpu())
            if model_kind == "vae":
                mu_batches.append(output.extras["mu"].detach().cpu())
                logvar_batches.append(output.extras["logvar"].detach().cpu())
            examples += inputs.shape[0]
            if examples >= max_points:
                break

        all_latent = torch.cat(latent_batches)[:max_points]
        all_labels = torch.cat(label_batches)[:max_points]
        _save_latent_scatter(
            all_latent,
            all_labels,
            figures / "latent-space.png",
        )

        if model_kind == "vae":
            all_mu = torch.cat(mu_batches)[:max_points]
            all_logvar = torch.cat(logvar_batches)[:max_points]
            per_element_kl = -0.5 * (
                1 + all_logvar - all_mu.square() - all_logvar.exp()
            )
            active_threshold = float(
                diagnostics_config.get("active_kl_threshold", 0.01)
            )
            kl_summary = _save_kl_diagnostics(
                per_element_kl.mean(dim=0),
                figures / "kl-per-dimension.png",
                active_threshold,
            )
            summary["vae_latent"] = kl_summary

    if model_kind == "vqvae":
        fixed_indices = fixed_output.extras["indices"]
        _save_token_maps(fixed_indices, figures / "token-maps.png")

        index_batches: list[Tensor] = []
        for inputs, _labels in validation_loader:
            output = model(inputs.to(device))
            index_batches.append(output.extras["indices"].detach().cpu())
        all_indices = torch.cat(index_batches)
        codebook_size = model.quantizer.codebook.num_embeddings
        codebook_summary = _save_codebook_diagnostics(
            all_indices,
            codebook_size,
            figures / "codebook-usage.png",
        )
        summary["codebook"] = codebook_summary

        random_indices = torch.randint(
            0,
            codebook_size,
            (16, fixed_indices.shape[1], fixed_indices.shape[2]),
            device=device,
        )
        embeddings = model.quantizer.codebook(random_indices)
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
        random_decoded = model.decoder(embeddings)
        save_image_grid(
            random_decoded,
            figures / "uniform-random-token-samples.png",
            title="Uniform random codes (not a learned generative prior)",
            nrow=8,
        )

    if summary:
        with (run_dir / "diagnostics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
