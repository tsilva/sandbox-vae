from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


Batch = dict[str, torch.Tensor]


@dataclass
class DatasetBundle:
    dataset_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    sample_shape: tuple[int, ...]
    normalization: dict[str, Any]
    factor_names: list[str]


class DatasetAdapter(ABC):
    dataset_id: str
    default_batch_size: int = 32

    @abstractmethod
    def build(self, data_root: Path, batch_size: int, seed: int, num_workers: int) -> DatasetBundle:
        raise NotImplementedError
