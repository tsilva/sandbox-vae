#!/usr/bin/env python3
"""Execute generated course notebooks without modifying committed copies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "course" / "notebooks"
PROFILES = {
    "smoke": ("00",),
    "foundations": ("00", "01", "02"),
    "all": tuple(f"{index:02d}" for index in range(14)),
}


def _resolve_notebooks(lesson_ids: tuple[str, ...]) -> list[Path]:
    resolved = []
    for lesson_id in lesson_ids:
        matches = sorted(NOTEBOOK_DIR.glob(f"{lesson_id}-*.ipynb"))
        if len(matches) != 1:
            raise SystemExit(
                f"Expected one generated notebook for lesson {lesson_id}, "
                f"found {len(matches)}"
            )
        resolved.append(matches[0])
    return resolved


def execute(
    lesson_ids: tuple[str, ...],
    *,
    output_dir: Path,
    timeout: int,
) -> None:
    notebooks = _resolve_notebooks(lesson_ids)
    output_dir.mkdir(parents=True, exist_ok=False)
    for notebook_path in notebooks:
        notebook = nbformat.read(notebook_path, as_version=4)
        NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        output_path = output_dir / notebook_path.name
        nbformat.write(notebook, output_path)
        print(f"executed {notebook_path.name} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute generated lessons and store output-bearing copies under runs/."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="smoke",
        help="Named lesson set (default: smoke).",
    )
    selection.add_argument(
        "--lessons",
        nargs="+",
        metavar="ID",
        help="Explicit lesson IDs, for example --lessons 00 01.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory; defaults to a timestamp under runs/.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum seconds per cell (default: 3600).",
    )
    args = parser.parse_args()
    lesson_ids = (
        tuple(f"{int(lesson_id):02d}" for lesson_id in args.lessons)
        if args.lessons
        else PROFILES[args.profile]
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        ROOT / "runs" / "notebook-executions" / timestamp
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    execute(lesson_ids, output_dir=output_dir, timeout=args.timeout)


if __name__ == "__main__":
    main()
