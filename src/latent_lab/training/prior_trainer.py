from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from latent_lab.config import save_yaml
from latent_lab.data import build_dataloaders
from latent_lab.diagnostics import save_image_grid
from latent_lab.models import build_model
from latent_lab.models.prior import AutoregressiveCodePrior
from latent_lab.training.seeding import resolve_device, seed_everything


@dataclass(frozen=True)
class PriorTrainingResult:
    run_dir: Path
    best_validation_loss: float
    best_validation_perplexity: float


@torch.no_grad()
def _extract_tokens(model, loader, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    batches: list[torch.Tensor] = []
    spatial_shape = None
    model.eval()
    for inputs, _labels in loader:
        indices = model(inputs.to(device)).extras["indices"]
        spatial_shape = (indices.shape[1], indices.shape[2])
        batches.append(indices.flatten(start_dim=1).cpu())
    if spatial_shape is None:
        raise ValueError("Cannot train a prior on an empty dataset")
    return torch.cat(batches), spatial_shape


@torch.no_grad()
def _evaluate_prior(
    prior: AutoregressiveCodePrior,
    loader: DataLoader,
    device: torch.device,
) -> float:
    prior.eval()
    total = 0.0
    tokens_seen = 0
    for (tokens,) in loader:
        tokens = tokens.to(device)
        logits = prior(tokens)
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
            reduction="sum",
        )
        total += float(loss.cpu())
        tokens_seen += tokens.numel()
    return total / tokens_seen


def run_code_prior_training(
    prior_config: dict[str, Any],
    vq_checkpoint: str | Path,
    *,
    run_root: str | Path = "runs",
    device_override: str | None = None,
) -> PriorTrainingResult:
    required = {"id", "model", "training"}
    missing = sorted(required - prior_config.keys())
    if missing:
        raise ValueError(f"Prior recipe is missing: {', '.join(missing)}")

    training = prior_config["training"]
    seed = int(training.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(device_override or str(training.get("device", "auto")))

    checkpoint_path = Path(vq_checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    vq_config = checkpoint["config"]
    if str(vq_config["model"]["kind"]).lower() != "vqvae":
        raise ValueError("The prior requires a VQ-VAE checkpoint")

    train_loader, validation_loader, dataset_spec = build_dataloaders(
        vq_config["dataset"], vq_config["training"]
    )
    vqvae = build_model(vq_config["model"], dataset_spec).to(device)
    vqvae.load_state_dict(checkpoint["model"])
    vqvae.requires_grad_(False)

    train_tokens, spatial_shape = _extract_tokens(vqvae, train_loader, device)
    validation_tokens, validation_shape = _extract_tokens(
        vqvae, validation_loader, device
    )
    if validation_shape != spatial_shape:
        raise ValueError("Training and validation token maps have different shapes")

    batch_size = int(training.get("batch_size", 128))
    generator = torch.Generator().manual_seed(seed)
    token_train_loader = DataLoader(
        TensorDataset(train_tokens),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    token_validation_loader = DataLoader(
        TensorDataset(validation_tokens),
        batch_size=batch_size,
        shuffle=False,
    )

    codebook_size = vqvae.quantizer.codebook.num_embeddings
    prior = AutoregressiveCodePrior(
        codebook_size=codebook_size,
        embedding_dim=int(prior_config["model"].get("embedding_dim", 64)),
        hidden_dim=int(prior_config["model"].get("hidden_dim", 128)),
    ).to(device)
    optimizer = torch.optim.Adam(
        prior.parameters(), lr=float(training.get("learning_rate", 1e-3))
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(run_root) / str(prior_config["id"]) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **prior_config,
        "vq_checkpoint": str(checkpoint_path),
        "token_map_shape": list(spatial_shape),
        "codebook_size": codebook_size,
    }
    save_yaml(resolved, run_dir / "resolved-config.yaml")

    best_loss = float("inf")
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(1, int(training["epochs"]) + 1):
        prior.train()
        train_total = 0.0
        tokens_seen = 0
        for (tokens,) in token_train_loader:
            tokens = tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = prior(tokens)
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.reshape(-1),
                reduction="sum",
            )
            mean_loss = loss / tokens.numel()
            mean_loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu())
            tokens_seen += tokens.numel()

        train_loss = train_total / tokens_seen
        validation_loss = _evaluate_prior(
            prior, token_validation_loader, device
        )
        record = {
            "epoch": epoch,
            "train/cross_entropy": train_loss,
            "train/perplexity": math.exp(min(train_loss, 20)),
            "validation/cross_entropy": validation_loss,
            "validation/perplexity": math.exp(min(validation_loss, 20)),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model": prior.state_dict(),
                    "config": resolved,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                run_dir / "checkpoint-best.pt",
            )

    best_checkpoint = torch.load(
        run_dir / "checkpoint-best.pt",
        map_location=device,
        weights_only=False,
    )
    prior.load_state_dict(best_checkpoint["model"])
    prior.eval()
    sampled_tokens = prior.sample(
        16,
        spatial_shape[0] * spatial_shape[1],
        temperature=float(prior_config.get("sampling", {}).get("temperature", 1.0)),
        device=device,
    ).view(16, *spatial_shape)
    embeddings = vqvae.quantizer.codebook(sampled_tokens)
    embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        samples = vqvae.decoder(embeddings)
    save_image_grid(
        samples,
        run_dir / "figures" / "learned-prior-samples.png",
        title="Samples from the learned autoregressive code prior",
        nrow=8,
    )

    perplexity = math.exp(min(best_loss, 20))
    summary = {
        "prior_recipe_id": prior_config["id"],
        "vq_checkpoint": str(checkpoint_path),
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_cross_entropy": best_loss,
        "best_validation_perplexity": perplexity,
        "codebook_size": codebook_size,
        "token_map_shape": list(spatial_shape),
        "parameter_count": sum(parameter.numel() for parameter in prior.parameters()),
        "device": str(device),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return PriorTrainingResult(
        run_dir=run_dir,
        best_validation_loss=best_loss,
        best_validation_perplexity=perplexity,
    )

