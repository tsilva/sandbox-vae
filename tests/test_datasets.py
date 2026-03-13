from sandbox_autoencoders.datasets.registry import DATASET_REGISTRY


def test_dataset_adapters_expose_expected_shapes(tmp_path):
    expected = {
        "mnist": (1, 32, 32),
        "dsprites": (1, 64, 64),
        "celeba": (3, 64, 64),
        "moving_mnist": (8, 1, 32, 32),
    }
    for dataset_id, adapter in DATASET_REGISTRY.items():
        bundle = adapter.build(tmp_path, batch_size=4, seed=3, num_workers=0)
        assert bundle.sample_shape == expected[dataset_id]
        assert bundle.train_loader.dataset is not None
        assert bundle.val_loader.dataset is not None
        assert "range" in bundle.normalization
