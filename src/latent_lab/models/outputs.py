from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class LatentModelOutput:
    reconstruction: Tensor
    latent: Tensor
    extras: dict[str, Tensor] = field(default_factory=dict)

