from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateDistortionPoint:
    capacity: float
    reconstruction_loss: float
    label: str

