from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms

from sandbox_autoencoders.datasets.base import DatasetAdapter, DatasetBundle
from sandbox_autoencoders.datasets.common import SimpleTensorDataset, make_loader, small_count, synthetic_images
from sandbox_autoencoders.utils.io import is_test_mode


class CelebAAdapter(DatasetAdapter):
    dataset_id = "celeba"

    def build(self, data_root: Path, batch_size: int, seed: int, num_workers: int) -> DatasetBundle:
        if is_test_mode():
            train_ds = SimpleTensorDataset(synthetic_images(small_count(36), 3, 64, seed))
            val_ds = SimpleTensorDataset(synthetic_images(small_count(18), 3, 64, seed + 1))
        else:
            transform = transforms.Compose(
                [
                    transforms.CenterCrop(178),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                ]
            )
            train_ds = datasets.CelebA(root=data_root / "celeba", split="train", transform=transform, download=True)
            val_ds = datasets.CelebA(root=data_root / "celeba", split="valid", transform=transform, download=True)
        return DatasetBundle(
            dataset_id=self.dataset_id,
            train_loader=make_loader(train_ds, batch_size, seed, num_workers, shuffle=True),
            val_loader=make_loader(val_ds, batch_size, seed + 1, num_workers, shuffle=False),
            sample_shape=(3, 64, 64),
            normalization={"range": [0.0, 1.0], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            factor_names=[],
        )
