from __future__ import annotations

import json
import copy
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from latent_lab.config import save_yaml, validate_recipe
from latent_lab.data import build_dataloaders
from latent_lab.diagnostics import (
    generate_generative_diagnostics,
    save_reconstruction_grid,
)
from latent_lab.models import build_model
from latent_lab.objectives import build_objective
from latent_lab.tracking import LocalTracker
from latent_lab.training.checkpointing import save_checkpoint
from latent_lab.training.evaluator import evaluate_epoch
from latent_lab.training.seeding import resolve_device, seed_everything


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    best_validation_loss: float
    best_epoch: int
    best_validation_metrics: dict[str, float]


def _run_directory(root: Path, recipe_id: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / recipe_id / f"{timestamp}-seed-{seed}"
    suffix = 1
    while candidate.exists():
        candidate = root / recipe_id / f"{timestamp}-seed-{seed}-{suffix}"
        suffix += 1
    return candidate


def _corrupt_inputs(inputs: torch.Tensor, config: dict[str, Any] | None):
    if not config:
        return inputs
    kind = str(config.get("kind", "none")).lower()
    if kind == "none":
        return inputs
    if kind == "gaussian":
        standard_deviation = float(config.get("standard_deviation", 0.25))
        corrupted = inputs + standard_deviation * torch.randn_like(inputs)
        return corrupted.clamp(0, 1) if config.get("clamp", True) else corrupted
    if kind == "mask":
        probability = float(config.get("probability", 0.25))
        mask = torch.rand_like(inputs) >= probability
        return inputs * mask
    raise ValueError(f"Unsupported input corruption: {kind}")


def _effective_beta(objective_config: dict[str, Any], epoch: int) -> float | None:
    if str(objective_config["kind"]).lower() != "vae":
        return None
    target = float(objective_config.get("beta", 1.0))
    warmup_epochs = int(objective_config.get("kl_warmup_epochs", 0))
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, epoch / warmup_epochs)


def run_training(
    config: dict[str, Any],
    *,
    run_root: str | Path = "runs",
    device_override: str | None = None,
) -> TrainingResult:
    validate_recipe(config)
    training = config["training"]
    seed = int(training.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(device_override or str(training.get("device", "auto")))

    train_loader, validation_loader, dataset_spec = build_dataloaders(
        config["dataset"], training
    )
    model = build_model(config["model"], dataset_spec).to(device)
    optimizer_name = str(training.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
    )

    run_dir = _run_directory(Path(run_root), str(config["id"]), seed)
    run_dir.mkdir(parents=True)
    save_yaml(config, run_dir / "resolved-config.yaml")
    tracker = LocalTracker(run_dir)
    fixed_inputs = next(iter(validation_loader))[0].to(device)
    corruption_config = training.get("input_corruption")

    best_loss = float("inf")
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    warmup_epochs = int(config["objective"].get("kl_warmup_epochs", 0))
    checkpoint_selection_start_epoch = max(1, warmup_epochs)
    for epoch in range(1, int(training["epochs"]) + 1):
        objective = build_objective(
            config["objective"],
            beta_override=_effective_beta(config["objective"], epoch),
        )
        model.train()
        totals: dict[str, float] = defaultdict(float)
        examples = 0
        for inputs, _labels in train_loader:
            clean_inputs = inputs.to(device)
            model_inputs = _corrupt_inputs(clean_inputs, corruption_config)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(model(model_inputs), clean_inputs)
            losses["loss"].backward()
            optimizer.step()

            batch_size = clean_inputs.shape[0]
            examples += batch_size
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * batch_size

        train_metrics = {
            f"train/{name}": value / examples for name, value in totals.items()
        }
        validation_metrics = {
            f"validation/{name}": value
            for name, value in evaluate_epoch(
                model, validation_loader, objective, device
            ).items()
        }
        tracker.log(
            {"epoch": epoch, **train_metrics, **validation_metrics}
        )

        validation_loss = validation_metrics["validation/loss"]
        if (
            epoch >= checkpoint_selection_start_epoch
            and validation_loss < best_loss
        ):
            best_loss = validation_loss
            best_epoch = epoch
            best_validation_metrics = copy.deepcopy(validation_metrics)
            save_checkpoint(
                run_dir / "checkpoint-best.pt",
                model,
                optimizer,
                epoch,
                validation_loss,
                config,
            )

    checkpoint = torch.load(
        run_dir / "checkpoint-best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        fixed_reconstructions = model(fixed_inputs).reconstruction
    save_reconstruction_grid(
        fixed_inputs,
        fixed_reconstructions,
        run_dir / "figures" / "reconstructions.png",
    )
    if corruption_config:
        with torch.no_grad():
            corrupted = _corrupt_inputs(fixed_inputs, corruption_config)
            denoised = model(corrupted).reconstruction
        save_reconstruction_grid(
            corrupted,
            denoised,
            run_dir / "figures" / "corrupted-input-reconstructions.png",
        )

    diagnostic_summary = generate_generative_diagnostics(
        model,
        str(config["model"]["kind"]).lower(),
        validation_loader,
        fixed_inputs,
        run_dir,
        device,
        config,
    )

    summary = {
        "recipe_id": config["id"],
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_metrics": best_validation_metrics,
        "checkpoint_selection_start_epoch": checkpoint_selection_start_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **diagnostic_summary,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return TrainingResult(
        run_dir=run_dir,
        best_validation_loss=best_loss,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
    )
