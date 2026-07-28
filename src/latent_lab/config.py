from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def save_yaml(value: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {key}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def with_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for key, value in overrides.items():
        set_dotted(resolved, key, value)
    return resolved


def validate_recipe(config: dict[str, Any]) -> None:
    required = {"id", "dataset", "model", "objective", "training"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Recipe is missing required keys: {', '.join(missing)}")

    if int(config["training"].get("epochs", 0)) < 1:
        raise ValueError("training.epochs must be at least 1")
    if int(config["training"].get("batch_size", 0)) < 1:
        raise ValueError("training.batch_size must be at least 1")
    model_kind = str(config["model"].get("kind", "")).lower()
    objective_kind = str(config["objective"].get("kind", "")).lower()
    if model_kind != objective_kind:
        raise ValueError(
            f"model.kind ({model_kind}) and objective.kind "
            f"({objective_kind}) must match"
        )
    warmup_epochs = int(config["objective"].get("kl_warmup_epochs", 0))
    if warmup_epochs > int(config["training"]["epochs"]):
        raise ValueError(
            "objective.kl_warmup_epochs cannot exceed training.epochs"
        )

