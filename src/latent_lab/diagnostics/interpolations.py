from __future__ import annotations

import torch
from torch import Tensor


def linear_interpolation(start: Tensor, end: Tensor, steps: int) -> Tensor:
    weights = torch.linspace(0, 1, steps, device=start.device)
    shape = (steps,) + (1,) * start.ndim
    weights = weights.view(shape)
    return (1 - weights) * start.unsqueeze(0) + weights * end.unsqueeze(0)

