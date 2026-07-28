from pathlib import Path

import matplotlib.pyplot as pyplot
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from latent_lab.course import (
    balanced_class_batch,
    latest_run_dir,
    plot_metric_history,
)
from latent_lab.diagnostics import (
    per_example_mse,
    plot_image_grid,
    plot_reconstruction_grid,
)
from latent_lab.training import (
    compute_mean_image,
    constant_reconstruction_errors,
)


def test_mean_image_and_constant_errors_are_exposed_as_pure_probes() -> None:
    images = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1)
    labels = torch.zeros(3, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    mean_image, examples = compute_mean_image(loader, torch.device("cpu"))
    errors = constant_reconstruction_errors(
        loader, mean_image, torch.device("cpu")
    )

    assert examples == 3
    assert torch.equal(mean_image, torch.tensor([[[[0.5]]]]))
    assert torch.allclose(errors, torch.tensor([0.25, 0.0, 0.25]))


def test_balanced_class_batch_orders_examples_by_class() -> None:
    images = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
    labels = torch.tensor([2, 0, 1, 2, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    selected_images, selected_labels = balanced_class_batch(loader, 3)

    assert selected_labels.tolist() == [0, 1, 2]
    assert selected_images.flatten().tolist() == [1.0, 2.0, 0.0]


def test_reconstruction_figures_return_notebook_displayable_figures() -> None:
    inputs = torch.zeros(3, 1, 4, 4)
    reconstructions = torch.ones_like(inputs) * 0.5
    labels = torch.tensor([0, 1, 2])

    errors = per_example_mse(inputs, reconstructions)
    comparison = plot_reconstruction_grid(
        inputs,
        reconstructions,
        labels=labels,
        class_names=("zero", "one", "two"),
        include_error=True,
    )
    examples = plot_image_grid(
        inputs,
        labels=labels,
        class_names=("zero", "one", "two"),
    )

    assert torch.allclose(errors, torch.full((3,), 0.25))
    assert len(comparison.axes) == 9
    assert len(examples.axes) == 3
    pyplot.close(comparison)
    pyplot.close(examples)


def test_metric_history_rejects_unknown_metrics() -> None:
    with pytest.raises(KeyError, match="available"):
        plot_metric_history([{"epoch": 1, "validation/loss": 0.5}], ["missing"])


def test_latest_run_dir_requires_a_completed_run(tmp_path: Path) -> None:
    incomplete = tmp_path / "ae/demo" / "20260101T000000Z-seed-0"
    complete = tmp_path / "ae/demo" / "20260102T000000Z-seed-0"
    incomplete.mkdir(parents=True)
    complete.mkdir()
    (complete / "summary.json").write_text("{}", encoding="utf-8")

    assert latest_run_dir(tmp_path, "ae/demo") == complete
