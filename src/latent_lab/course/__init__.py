"""Notebook-facing helpers for inspecting experiments without duplicating logic."""

from latent_lab.course.probes import (
    balanced_class_batch,
    latest_run_dir,
    load_metrics,
    load_run_summary,
    load_trained_model,
    plot_metric_history,
    repository_root,
)

__all__ = [
    "balanced_class_batch",
    "latest_run_dir",
    "load_metrics",
    "load_run_summary",
    "load_trained_model",
    "plot_metric_history",
    "repository_root",
]
