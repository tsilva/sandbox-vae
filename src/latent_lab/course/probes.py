from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as pyplot
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

from latent_lab.config import load_yaml
from latent_lab.data.datasets import dataset_spec
from latent_lab.models import build_model


def repository_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook or project subdirectory."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "course" / "curriculum.yaml").is_file()
        ):
            return candidate
    raise FileNotFoundError(f"Could not find the latent-lab root from {current}")


@torch.no_grad()
def balanced_class_batch(
    data_loader,
    num_classes: int,
    *,
    examples_per_class: int = 1,
) -> tuple[Tensor, Tensor]:
    """Collect a deterministic class-balanced batch from an ordered loader."""

    if num_classes <= 0 or examples_per_class <= 0:
        raise ValueError("Class and example counts must be positive")
    collected: dict[int, list[Tensor]] = {
        label: [] for label in range(num_classes)
    }
    for inputs, labels in data_loader:
        for image, label_tensor in zip(inputs, labels, strict=True):
            label = int(label_tensor)
            if label in collected and len(collected[label]) < examples_per_class:
                collected[label].append(image.detach().cpu())
        if all(len(images) == examples_per_class for images in collected.values()):
            break
    missing = [
        label
        for label, images in collected.items()
        if len(images) != examples_per_class
    ]
    if missing:
        raise ValueError(f"Loader did not contain requested classes: {missing}")
    images = []
    labels = []
    for label in range(num_classes):
        for image in collected[label]:
            images.append(image)
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def latest_run_dir(run_root: str | Path, experiment_id: str) -> Path:
    """Return the lexicographically latest timestamped run for an experiment."""

    experiment_dir = Path(run_root) / experiment_id
    candidates = (
        sorted(
            candidate
            for candidate in experiment_dir.iterdir()
            if candidate.is_dir() and (candidate / "summary.json").is_file()
        )
        if experiment_dir.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {experiment_dir}")
    return candidates[-1]


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"No summary.json found in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.is_file():
        return []
    return [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_trained_model(
    run_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct a model from a run's resolved config and best checkpoint."""

    run_path = Path(run_dir)
    config = load_yaml(run_path / "resolved-config.yaml")
    checkpoint_path = run_path / "checkpoint-best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint-best.pt found in {run_path}")
    resolved_device = torch.device(device)
    model = build_model(config["model"], dataset_spec(config["dataset"]))
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(resolved_device)
    model.eval()
    return model, config


def plot_metric_history(
    records: Iterable[dict[str, Any]],
    metric_names: Sequence[str],
    *,
    title: str = "Metric history",
) -> Figure:
    """Plot selected metrics from JSONL records on a shared epoch axis."""

    history = list(records)
    if not history:
        raise ValueError("At least one metric record is required")
    figure, axis = pyplot.subplots(figsize=(8, 4.5))
    epochs = [
        record.get("epoch", index + 1)
        for index, record in enumerate(history)
    ]
    plotted = 0
    for metric_name in metric_names:
        values = [record.get(metric_name) for record in history]
        if any(value is None for value in values):
            continue
        axis.plot(epochs, values, label=metric_name)
        plotted += 1
    if plotted == 0:
        available = sorted(
            {key for record in history for key in record if key != "epoch"}
        )
        raise KeyError(
            f"None of {list(metric_names)} were present; available={available}"
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure
