from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb


@dataclass
class WandbConfig:
    mode: str
    project: str
    entity: str | None
    run_name: str | None
    group: str | None
    tags: list[str]
    output_dir: Path
    config: dict[str, Any]


class WandbSession:
    def __init__(self, cfg: WandbConfig):
        self.enabled = cfg.mode != "disabled"
        self._run = None
        if self.enabled:
            self._run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=cfg.run_name,
                group=cfg.group,
                tags=cfg.tags,
                dir=str(cfg.output_dir),
                config=cfg.config,
                mode=cfg.mode,
                reinit=True,
            )

    def log_metrics(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            wandb.log(payload, step=step)

    def log_image(self, key: str, image_path: Path, step: int | None = None) -> None:
        if self._run is not None and image_path.exists():
            wandb.log({key: wandb.Image(str(image_path))}, step=step)

    def log_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        if self._run is not None and rows:
            columns = list(rows[0].keys())
            table = wandb.Table(columns=columns)
            for row in rows:
                table.add_data(*[row.get(column) for column in columns])
            wandb.log({key: table})

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if self._run is not None:
            if summary:
                for key, value in summary.items():
                    self._run.summary[key] = value
            self._run.finish()
