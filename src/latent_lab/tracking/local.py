from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalTracker:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

