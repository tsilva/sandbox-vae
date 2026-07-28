from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from latent_lab.config import save_yaml, validate_recipe
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import per_example_mse, save_reconstruction_grid
from latent_lab.training.seeding import resolve_device, seed_everything


@dataclass(frozen=True)
class BaselineResult:
    run_dir: Path
    validation_mse: float


@torch.no_grad()
def compute_mean_image(
    train_loader,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute the pixelwise training-set mean without retaining the dataset."""

    pixel_sum = None
    examples = 0
    for inputs, _labels in train_loader:
        inputs = inputs.to(device)
        batch_sum = inputs.sum(dim=0, keepdim=True)
        pixel_sum = batch_sum if pixel_sum is None else pixel_sum + batch_sum
        examples += inputs.shape[0]
    if pixel_sum is None or examples == 0:
        raise ValueError("Cannot compute a mean image from an empty loader")
    return pixel_sum / examples, examples


@torch.no_grad()
def constant_reconstruction_errors(
    data_loader,
    prediction: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return per-example MSE for a prediction that ignores its input."""

    errors = []
    for inputs, _labels in data_loader:
        inputs = inputs.to(device)
        predictions = prediction.expand(inputs.shape[0], -1, -1, -1)
        errors.append(per_example_mse(inputs, predictions).cpu())
    if not errors:
        raise ValueError(
            "Cannot evaluate a constant prediction on an empty loader"
        )
    return torch.cat(errors)


@torch.no_grad()
def run_mean_image_baseline(
    config: dict[str, Any],
    *,
    run_root: str | Path = "runs",
    device_override: str | None = None,
) -> BaselineResult:
    validate_recipe(config)
    training = config["training"]
    seed = int(training.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(device_override or str(training.get("device", "auto")))
    train_loader, validation_loader, _spec = build_dataloaders(
        config["dataset"], training
    )

    mean_image, examples = compute_mean_image(train_loader, device)
    fixed_inputs = None
    fixed_labels = None
    for inputs, labels in validation_loader:
        if fixed_inputs is None:
            fixed_inputs = inputs.to(device)
            fixed_labels = labels
            break
    validation_errors = constant_reconstruction_errors(
        validation_loader, mean_image, device
    )
    validation_mse = float(validation_errors.mean())
    validation_examples = validation_errors.numel()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_name = str(config["dataset"]["name"])
    run_dir = Path(run_root) / "baselines" / dataset_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "resolved-config.yaml")
    assert fixed_inputs is not None
    assert fixed_labels is not None
    predictions = mean_image.expand(fixed_inputs.shape[0], -1, -1, -1)
    save_reconstruction_grid(
        fixed_inputs,
        predictions,
        run_dir / "figures" / "mean-image-baseline.png",
        labels=fixed_labels,
        class_names=class_names(dataset_name),
        include_error=True,
    )
    summary = {
        "baseline": "mean_training_image",
        "dataset": dataset_name,
        "validation_mse": validation_mse,
        "train_examples": examples,
        "validation_examples": validation_examples,
        "device": str(device),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return BaselineResult(run_dir=run_dir, validation_mse=validation_mse)
