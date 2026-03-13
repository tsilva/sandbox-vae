from __future__ import annotations

from sandbox_autoencoders.datasets.base import DatasetAdapter
from sandbox_autoencoders.datasets.celeba import CelebAAdapter
from sandbox_autoencoders.datasets.dsprites import DSpritesAdapter
from sandbox_autoencoders.datasets.mnist import MNISTAdapter
from sandbox_autoencoders.datasets.moving_mnist import MovingMNISTAdapter


DATASET_REGISTRY: dict[str, DatasetAdapter] = {
    "mnist": MNISTAdapter(),
    "dsprites": DSpritesAdapter(),
    "celeba": CelebAAdapter(),
    "moving_mnist": MovingMNISTAdapter(),
}


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset adapter: {dataset_id}") from exc
