from __future__ import annotations

import math

from torch import nn


def activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def output_activation(name: str) -> nn.Module | None:
    name = name.lower()
    if name == "none":
        return None
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported output activation: {name}")


def mlp(
    dimensions: list[int],
    activation_name: str,
    *,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            layers.append(activation(activation_name))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


def flattened_size(input_shape: tuple[int, int, int]) -> int:
    return math.prod(input_shape)
