from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sandbox_autoencoders.datasets.registry import get_dataset_adapter
from sandbox_autoencoders.models.conv import Autoencoder, ModelOutput, VariationalAutoencoder
from sandbox_autoencoders.training.diagnostics import factor_predictability, nearest_neighbor_indices, summarize_latents
from sandbox_autoencoders.training.losses import apply_free_bits, kl_per_dim, reconstruction_loss
from sandbox_autoencoders.training.schedules import beta_for_epoch
from sandbox_autoencoders.utils.io import ensure_dir, render_recon_grid, save_json, save_tensor_grid, seed_everything, write_text
from sandbox_autoencoders.utils.wandb import WandbConfig, WandbSession


@dataclass
class VariantConfig:
    name: str
    dataset_id: str
    model_type: str
    latent_dim: int
    loss_type: str
    decoder_variant: str = "standard"
    beta: float = 1e-3
    beta_schedule: str = "constant"
    warmup_epochs: int = 0
    free_bits: float = 0.0
    temporal: bool = False
    smoothness_penalty: float = 0.0
    enable_prior_samples: bool = False
    enable_interpolation: bool = False
    enable_traversals: bool = False
    enable_sampling_diagnostics: bool = False
    learning_rate: float = 1e-3
    epochs: int = 3
    notes: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def _build_model(cfg: VariantConfig, sample_shape: tuple[int, ...]) -> nn.Module:
    if cfg.temporal:
        _, channels, size, _ = sample_shape
    else:
        channels, size, _ = sample_shape
    if cfg.model_type == "ae":
        return Autoencoder(channels, size, cfg.latent_dim, decoder_variant=cfg.decoder_variant)
    return VariationalAutoencoder(channels, size, cfg.latent_dim, decoder_variant=cfg.decoder_variant)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _select_inputs(batch: dict[str, torch.Tensor], temporal: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if temporal:
        sequence = batch["sequence"]
        bsz, steps, channels, height, width = sequence.shape
        flat = sequence.view(bsz * steps, channels, height, width)
        return flat, sequence
    return batch["image"], None


def _temporal_penalty(latent: torch.Tensor, sequence_shape: torch.Tensor | None) -> torch.Tensor:
    if sequence_shape is None:
        return torch.tensor(0.0, device=latent.device)
    steps = sequence_shape.shape[1]
    latent_seq = latent.view(sequence_shape.shape[0], steps, -1)
    return (latent_seq[:, 1:] - latent_seq[:, :-1]).pow(2).mean()


def _collect_preview(preview: tuple[torch.Tensor, torch.Tensor] | None, inputs: torch.Tensor, recon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if preview is not None:
        return preview
    return inputs[:8].detach().cpu(), recon[:8].detach().cpu()


def _run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    cfg: VariantConfig,
    beta: float,
    train: bool,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    model.train(mode=train)
    aggregate = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "recon_l1": 0.0,
        "recon_mse": 0.0,
        "kl_loss": 0.0,
        "smoothness_loss": 0.0,
        "count": 0,
    }
    preview = None
    latents = []
    mus = []
    logvars = []
    factors = []
    for batch in loader:
        batch = _to_device(batch, device)
        inputs, sequence = _select_inputs(batch, cfg.temporal)
        output: ModelOutput = model(inputs)
        per_sample_recon = reconstruction_loss(cfg.loss_type, output.recon, output.recon_logits, inputs)
        l1 = (output.recon - inputs).abs().flatten(start_dim=1).mean(dim=1)
        mse = (output.recon - inputs).pow(2).flatten(start_dim=1).mean(dim=1)
        kl_loss = torch.zeros_like(per_sample_recon)
        if output.mu is not None and output.logvar is not None:
            kl_loss = apply_free_bits(kl_per_dim(output.mu, output.logvar), cfg.free_bits)
        smoothness = _temporal_penalty(output.mu if output.mu is not None else output.latent, sequence)
        total_loss = per_sample_recon.mean() + beta * kl_loss.mean() + cfg.smoothness_penalty * smoothness
        if train:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        batch_count = inputs.shape[0]
        aggregate["loss"] += total_loss.item() * batch_count
        aggregate["recon_loss"] += per_sample_recon.mean().item() * batch_count
        aggregate["recon_l1"] += l1.mean().item() * batch_count
        aggregate["recon_mse"] += mse.mean().item() * batch_count
        aggregate["kl_loss"] += kl_loss.mean().item() * batch_count
        aggregate["smoothness_loss"] += smoothness.item() * batch_count
        aggregate["count"] += batch_count
        preview = _collect_preview(preview, inputs, output.recon)
        latents.append(output.latent.detach())
        if output.mu is not None:
            mus.append(output.mu.detach())
        if output.logvar is not None:
            logvars.append(output.logvar.detach())
        if "factors" in batch:
            factors.append(batch["factors"].detach())
    latent_tensor = torch.cat(latents, dim=0)
    mu_tensor = torch.cat(mus, dim=0) if mus else None
    logvar_tensor = torch.cat(logvars, dim=0) if logvars else None
    factor_tensor = torch.cat(factors, dim=0) if factors else torch.empty(0)
    diagnostics = summarize_latents(mu_tensor, logvar_tensor, latent_tensor)
    diagnostics.update(factor_predictability(latent_tensor, factor_tensor) if factor_tensor.numel() else {})
    metrics = {
        "loss": aggregate["loss"] / max(1, aggregate["count"]),
        "recon_loss": aggregate["recon_loss"] / max(1, aggregate["count"]),
        "recon_l1": aggregate["recon_l1"] / max(1, aggregate["count"]),
        "recon_mse": aggregate["recon_mse"] / max(1, aggregate["count"]),
        "kl_loss": aggregate["kl_loss"] / max(1, aggregate["count"]),
        "smoothness_loss": aggregate["smoothness_loss"] / max(1, aggregate["count"]),
    }
    metrics["active_latents"] = diagnostics["active_latents"]
    metrics["mu_abs"] = diagnostics.get("mu_abs", diagnostics["latent_mean_abs"])
    metrics["logvar_abs"] = diagnostics.get("logvar_abs", 0.0)
    return metrics, preview, diagnostics


def _interpolate(model: nn.Module, inputs: torch.Tensor, path: Path) -> None:
    if inputs.shape[0] < 2:
        return
    z0 = model.encode_latent(inputs[:1])
    z1 = model.encode_latent(inputs[1:2])
    steps = torch.linspace(0, 1, steps=8, device=inputs.device).view(-1, 1)
    latents = z0 * (1 - steps) + z1 * steps
    decoded = model.decode_latent(latents)
    save_tensor_grid(path, decoded, nrow=8)


def _traversals(model: nn.Module, latent_dim: int, device: torch.device, path: Path) -> None:
    dims = min(6, latent_dim)
    rows = []
    for dim in range(dims):
        latent = torch.zeros((7, latent_dim), device=device)
        latent[:, dim] = torch.linspace(-2.0, 2.0, steps=7, device=device)
        rows.append(model.decode_latent(latent).detach().cpu())
    save_tensor_grid(path, torch.cat(rows, dim=0), nrow=7)


def _sampling_diagnostics(model: VariationalAutoencoder, reference: torch.Tensor, device: torch.device) -> dict[str, Any]:
    prior = model.sample(min(8, reference.shape[0]), device=device).detach().cpu()
    posterior = model.decode_latent(model.encode_latent(reference.to(device))[: prior.shape[0]]).detach().cpu()
    return {
        "prior_neighbor_indices": nearest_neighbor_indices(prior, reference),
        "posterior_neighbor_indices": nearest_neighbor_indices(posterior, reference),
    }


def _notes_text(cfg: VariantConfig) -> str:
    sections = [
        ("Expected healthy pattern", cfg.notes.get("healthy", "Reconstructions improve while latent usage remains measurable.")),
        ("Common failure pattern", cfg.notes.get("failure", "KL collapses or reconstructions become oversmoothed.")),
        ("What to compare against previous step", cfg.notes.get("compare", "Compare the recon/KL tradeoff and latent usage against the prior experiment.")),
    ]
    lines = [f"# {cfg.name}", ""]
    for title, body in sections:
        lines.extend([f"## {title}", body, ""])
    return "\n".join(lines)


def _copy_best_artifacts(root_dir: Path, best_dir: Path) -> None:
    for name in [
        "recon_grid.png",
        "prior_samples.png",
        "interp_sheet.png",
        "traversal_sheet.png",
        "latent_stats.json",
        "notes.md",
    ]:
        source = best_dir / name
        if source.exists():
            shutil.copyfile(source, root_dir / name)


def execute_experiment(
    experiment_id: str,
    variants: list[VariantConfig],
    *,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    seed: int,
    device_name: str,
    num_workers: int,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_group: str | None,
    wandb_tags: list[str],
) -> dict[str, Any]:
    root_dir = ensure_dir(output_dir / experiment_id)
    all_histories = []
    run_summaries = []
    best_variant_dir = None
    best_val_loss = None
    for index, variant in enumerate(variants):
        run_seed = seed + index
        seed_everything(run_seed)
        variant_dir = ensure_dir(root_dir / variant.name)
        adapter = get_dataset_adapter(variant.dataset_id)
        bundle = adapter.build(data_root=data_root, batch_size=batch_size, seed=run_seed, num_workers=num_workers)
        device = torch.device(device_name)
        model = _build_model(variant, bundle.sample_shape).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=variant.learning_rate)
        wb = WandbSession(
            WandbConfig(
                mode=wandb_mode,
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name or f"{experiment_id}-{variant.name}",
                group=wandb_group or f"{experiment_id}-{variant.dataset_id}",
                tags=sorted(set(wandb_tags + variant.tags)),
                output_dir=variant_dir,
                config={
                    "dataset_id": variant.dataset_id,
                    "model_type": variant.model_type,
                    "latent_dim": variant.latent_dim,
                    "beta_schedule": variant.beta_schedule,
                    "loss_type": variant.loss_type,
                    "decoder_variant": variant.decoder_variant,
                    "seed": run_seed,
                },
            )
        )
        history = []
        latest_preview = None
        latest_diag = {}
        for epoch in range(variant.epochs):
            beta = beta_for_epoch(epoch, variant.epochs, variant.beta, variant.beta_schedule, variant.warmup_epochs)
            train_metrics, _, _ = _run_epoch(model, bundle.train_loader, optimizer, device, variant, beta, train=True)
            with torch.no_grad():
                val_metrics, latest_preview, latest_diag = _run_epoch(model, bundle.val_loader, optimizer, device, variant, beta, train=False)
            epoch_payload = {
                "epoch": epoch + 1,
                "beta": beta,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "train_recon_l1": train_metrics["recon_l1"],
                "val_recon_l1": val_metrics["recon_l1"],
                "train_recon_mse": train_metrics["recon_mse"],
                "val_recon_mse": val_metrics["recon_mse"],
                "train_kl_loss": train_metrics["kl_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
                "beta": beta,
                "active_latents": val_metrics["active_latents"],
                "mu_abs": val_metrics["mu_abs"],
                "logvar_abs": val_metrics["logvar_abs"],
            }
            if variant.temporal:
                epoch_payload["val_temporal_smoothness"] = val_metrics["smoothness_loss"]
            history.append(epoch_payload)
            recon_path = variant_dir / "recon_grid.png"
            render_recon_grid(recon_path, latest_preview[0], latest_preview[1], nrow=min(4, latest_preview[0].shape[0]))
            wb.log_metrics(epoch_payload, step=epoch + 1)
            wb.log_image("recon_grid", recon_path, step=epoch + 1)
        save_json(variant_dir / "history.json", history)
        save_json(variant_dir / "latent_stats.json", latest_diag)
        if latest_preview is not None and variant.enable_interpolation:
            _interpolate(model, latest_preview[0].to(device), variant_dir / "interp_sheet.png")
            wb.log_image("interp_sheet", variant_dir / "interp_sheet.png")
        if variant.enable_traversals:
            _traversals(model, variant.latent_dim, device, variant_dir / "traversal_sheet.png")
            wb.log_image("traversal_sheet", variant_dir / "traversal_sheet.png")
        if variant.enable_prior_samples and isinstance(model, VariationalAutoencoder):
            prior = model.sample(16, device=device).detach().cpu()
            save_tensor_grid(variant_dir / "prior_samples.png", prior, nrow=4)
            wb.log_image("prior_samples", variant_dir / "prior_samples.png")
        summary = {
            "experiment_id": experiment_id,
            "variant": variant.name,
            "dataset_id": variant.dataset_id,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "active_latents": latest_diag["active_latents"],
            "mu_abs": latest_diag.get("mu_abs", 0.0),
            "logvar_abs": latest_diag.get("logvar_abs", 0.0),
            "loss_type": variant.loss_type,
            "decoder_variant": variant.decoder_variant,
            "beta_schedule": variant.beta_schedule,
        }
        if variant.enable_sampling_diagnostics and isinstance(model, VariationalAutoencoder):
            reference = latest_preview[0]
            summary.update(_sampling_diagnostics(model, reference, device))
        save_json(variant_dir / "summary.json", summary)
        write_text(variant_dir / "notes.md", _notes_text(variant))
        wb.finish(summary)
        all_histories.append({"variant": variant.name, "history": history})
        run_summaries.append(summary)
        if best_val_loss is None or summary["final_val_loss"] < best_val_loss:
            best_val_loss = summary["final_val_loss"]
            best_variant_dir = variant_dir
    combined_summary = {"experiment_id": experiment_id, "variants": run_summaries}
    save_json(root_dir / "history.json", all_histories)
    save_json(root_dir / "summary.json", combined_summary)
    if best_variant_dir is not None:
        _copy_best_artifacts(root_dir, best_variant_dir)
    if run_summaries:
        write_text(root_dir / "notes.md", _notes_text(variants[0]))
    return combined_summary
