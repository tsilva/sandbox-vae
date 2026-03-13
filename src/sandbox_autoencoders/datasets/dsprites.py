from __future__ import annotations

from pathlib import Path

import torch

from sandbox_autoencoders.datasets.base import DatasetAdapter, DatasetBundle
from sandbox_autoencoders.datasets.common import (
    SimpleTensorDataset,
    ensure_npz,
    load_npz_array,
    make_loader,
    small_count,
    split_dataset,
    synthetic_factors,
    synthetic_images,
)
from sandbox_autoencoders.utils.io import is_test_mode


class DSpritesAdapter(DatasetAdapter):
    dataset_id = "dsprites"
    factor_names = ["shape", "scale", "orientation", "pos_x", "pos_y"]
    source_url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def build(self, data_root: Path, batch_size: int, seed: int, num_workers: int) -> DatasetBundle:
        if is_test_mode():
            train_ds = SimpleTensorDataset(
                synthetic_images(small_count(40), 1, 64, seed),
                factors=synthetic_factors(small_count(40), len(self.factor_names), seed),
            )
            val_ds = SimpleTensorDataset(
                synthetic_images(small_count(20), 1, 64, seed + 1),
                factors=synthetic_factors(small_count(20), len(self.factor_names), seed + 1),
            )
        else:
            npz_path = ensure_npz(
                data_root / "dsprites" / "dsprites.npz",
                self.source_url,
            )
            images = torch.from_numpy(load_npz_array(npz_path, "imgs")).float().unsqueeze(1)
            factors = torch.from_numpy(load_npz_array(npz_path, "latents_classes")[:, 1:6]).float()
            dataset = SimpleTensorDataset(images, factors=factors)
            train_ds, val_ds = split_dataset(dataset, seed=seed, val_fraction=0.1)
        return DatasetBundle(
            dataset_id=self.dataset_id,
            train_loader=make_loader(train_ds, batch_size, seed, num_workers, shuffle=True),
            val_loader=make_loader(val_ds, batch_size, seed + 1, num_workers, shuffle=False),
            sample_shape=(1, 64, 64),
            normalization={"range": [0.0, 1.0], "mean": [0.0], "std": [1.0]},
            factor_names=list(self.factor_names),
        )
