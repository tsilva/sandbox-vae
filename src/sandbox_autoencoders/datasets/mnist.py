from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

from sandbox_autoencoders.datasets.base import DatasetAdapter, DatasetBundle
from sandbox_autoencoders.datasets.common import SimpleTensorDataset, make_loader, small_count, split_dataset, synthetic_images
from sandbox_autoencoders.utils.io import is_test_mode


class MNISTAdapter(DatasetAdapter):
    dataset_id = "mnist"

    def build(self, data_root: Path, batch_size: int, seed: int, num_workers: int) -> DatasetBundle:
        if is_test_mode():
            train_ds = SimpleTensorDataset(synthetic_images(small_count(32), 1, 32, seed))
            val_ds = SimpleTensorDataset(synthetic_images(small_count(16), 1, 32, seed + 1))
        else:
            transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            train_raw = datasets.MNIST(root=data_root / "mnist", train=True, transform=transform, download=True)
            test_raw = datasets.MNIST(root=data_root / "mnist", train=False, transform=transform, download=True)
            train_ds, val_holdout = split_dataset(train_raw, seed=seed, val_fraction=0.1)
            val_ds = ConcatDataset([val_holdout, test_raw])
        return DatasetBundle(
            dataset_id=self.dataset_id,
            train_loader=make_loader(train_ds, batch_size, seed, num_workers, shuffle=True),
            val_loader=make_loader(val_ds, batch_size, seed + 1, num_workers, shuffle=False),
            sample_shape=(1, 32, 32),
            normalization={"range": [0.0, 1.0], "mean": [0.1307], "std": [0.3081]},
            factor_names=[],
        )
