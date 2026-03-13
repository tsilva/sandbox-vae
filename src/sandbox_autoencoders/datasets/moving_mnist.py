from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from sandbox_autoencoders.datasets.base import DatasetAdapter, DatasetBundle
from sandbox_autoencoders.datasets.common import SequenceDataset, make_loader, small_count, synthetic_sequences
from sandbox_autoencoders.utils.io import is_test_mode


class MovingMNISTDataset(Dataset):
    def __init__(self, root: Path, count: int, seed: int, steps: int = 8, size: int = 32):
        transform = transforms.Compose([transforms.Resize((size // 2, size // 2)), transforms.ToTensor()])
        self.mnist = datasets.MNIST(root=root, train=True, transform=transform, download=True)
        self.count = count
        self.steps = steps
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        digit_index = torch.randint(0, len(self.mnist), (1,), generator=generator).item()
        digit, _ = self.mnist[digit_index]
        velocity = torch.randint(1, 4, (2,), generator=generator).float() * torch.tensor([1.0, -1.0])
        pos = torch.randint(0, self.size // 2, (2,), generator=generator).float()
        frames = []
        for _ in range(self.steps):
            canvas = torch.zeros((1, self.size, self.size))
            x, y = pos.int().tolist()
            canvas[:, y : y + digit.shape[-2], x : x + digit.shape[-1]] = digit
            pos = pos + velocity
            for axis in (0, 1):
                limit = self.size - digit.shape[-1]
                if pos[axis] <= 0 or pos[axis] >= limit:
                    velocity[axis] *= -1
                    pos[axis] = pos[axis].clamp(0, limit)
            frames.append(canvas)
        return {"sequence": torch.stack(frames, dim=0)}


class MovingMNISTAdapter(DatasetAdapter):
    dataset_id = "moving_mnist"

    def build(self, data_root: Path, batch_size: int, seed: int, num_workers: int) -> DatasetBundle:
        if is_test_mode():
            train_ds = SequenceDataset(synthetic_sequences(small_count(24), 8, 1, 32, seed))
            val_ds = SequenceDataset(synthetic_sequences(small_count(12), 8, 1, 32, seed + 1))
        else:
            train_ds = MovingMNISTDataset(data_root / "moving_mnist", count=512, seed=seed)
            val_ds = MovingMNISTDataset(data_root / "moving_mnist", count=128, seed=seed + 1)
        return DatasetBundle(
            dataset_id=self.dataset_id,
            train_loader=make_loader(train_ds, batch_size, seed, num_workers, shuffle=True),
            val_loader=make_loader(val_ds, batch_size, seed + 1, num_workers, shuffle=False),
            sample_shape=(8, 1, 32, 32),
            normalization={"range": [0.0, 1.0], "mean": [0.0], "std": [1.0]},
            factor_names=[],
        )
