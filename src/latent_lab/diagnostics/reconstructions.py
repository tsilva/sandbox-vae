from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as pyplot
import torch
from torch import Tensor
from torchvision.utils import make_grid


def save_reconstruction_grid(
    inputs: Tensor,
    reconstructions: Tensor,
    path: Path,
    max_items: int = 8,
) -> None:
    count = min(max_items, inputs.shape[0])
    comparison = torch.cat(
        [inputs[:count].detach().cpu(), reconstructions[:count].detach().cpu()]
    )
    grid = make_grid(comparison, nrow=count, padding=2)
    image = grid.permute(1, 2, 0).clamp(0, 1).numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pyplot.subplots(figsize=(1.8 * count, 4))
    axis.imshow(image.squeeze(-1) if image.shape[-1] == 1 else image)
    axis.set_title("Originals (top) / reconstructions (bottom)")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    pyplot.close(figure)

