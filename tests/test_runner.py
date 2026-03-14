import torch

from sandbox_autoencoders.training.runner import _to_device


def test_to_device_accepts_torchvision_style_batch():
    images = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([1, 2, 3, 4])

    batch = _to_device([images, labels], torch.device("cpu"))

    assert torch.equal(batch["image"], images)
    assert torch.equal(batch["target"], labels)
