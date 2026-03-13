from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def is_test_mode() -> bool:
    return os.environ.get("SANDBOX_AUTOENCODERS_TEST_MODE", "0") == "1"


def mini_sweep_mode() -> bool:
    return is_test_mode() or os.environ.get("SANDBOX_AUTOENCODERS_MINI_SWEEP", "0") == "1"


def save_tensor_grid(path: str | Path, images: torch.Tensor, nrow: int = 4) -> None:
    images = images.detach().cpu().clamp(0.0, 1.0)
    grid = make_grid(images, nrow=nrow, padding=2)
    save_image(grid, str(path))


def render_recon_grid(path: str | Path, originals: torch.Tensor, reconstructions: torch.Tensor, nrow: int = 4) -> None:
    originals = originals.detach().cpu().clamp(0.0, 1.0)
    reconstructions = reconstructions.detach().cpu().clamp(0.0, 1.0)
    pair_grid = torch.cat([originals, reconstructions], dim=0)
    save_tensor_grid(path, pair_grid, nrow=nrow)
