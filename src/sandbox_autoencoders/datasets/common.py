from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from sandbox_autoencoders.datasets.base import Batch
from sandbox_autoencoders.utils.io import is_test_mode


@dataclass
class SimpleTensorDataset(Dataset):
    images: torch.Tensor
    factors: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Batch:
        item: Batch = {"image": self.images[index]}
        if self.factors is not None:
            item["factors"] = self.factors[index]
        return item


class SequenceDataset(Dataset):
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, index: int) -> Batch:
        return {"sequence": self.sequences[index]}


def make_loader(dataset: Dataset, batch_size: int, seed: int, num_workers: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def split_dataset(dataset: Dataset, seed: int, val_fraction: float = 0.1) -> tuple[Dataset, Dataset]:
    val_size = max(1, int(math.ceil(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def ensure_npz(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(url, path)
    return path


def synthetic_images(count: int, channels: int, size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((count, channels, size, size), generator=generator)


def synthetic_sequences(count: int, steps: int, channels: int, size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((count, steps, channels, size, size), generator=generator)


def synthetic_factors(count: int, dims: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low=0, high=6, size=(count, dims), generator=generator).float()


def small_count(default: int) -> int:
    return min(default, 48) if is_test_mode() else default


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False, encoding="latin1") as handle:
        return handle[key]
