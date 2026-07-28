from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from latent_lab.objectives import Objective


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    objective: Objective,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    for inputs, _labels in batches:
        inputs = inputs.to(device)
        losses = objective(model(inputs), inputs)
        batch_size = inputs.shape[0]
        examples += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / examples for name, value in totals.items()}

