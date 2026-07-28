#!/usr/bin/env python3
"""Build deterministic Jupyter notebooks from canonical Jupytext sources."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import jupytext
import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "course" / "notebook_sources"
OUTPUT_DIR = ROOT / "course" / "notebooks"


def _cell_id(source_name: str, index: int, cell) -> str:
    content = f"{source_name}\0{index}\0{cell.cell_type}\0{cell.source}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def render_notebook(source_path: Path) -> str:
    notebook = jupytext.read(source_path, fmt="py:percent")
    notebook.nbformat = 4
    notebook.nbformat_minor = 5
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "latent_lab": {
            "generated": True,
            "source": str(source_path.relative_to(ROOT)),
        },
    }
    for index, cell in enumerate(notebook.cells):
        cell.id = _cell_id(source_path.name, index, cell)
        tags = sorted(set(cell.metadata.get("tags", [])))
        cell.metadata = {"tags": tags} if tags else {}
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    rendered = nbformat.writes(notebook, version=4)
    return rendered.rstrip() + "\n"


def _expected_notebooks() -> dict[Path, str]:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Notebook source directory does not exist: {SOURCE_DIR}")
    sources = sorted(SOURCE_DIR.glob("[0-9][0-9]-*.py"))
    if not sources:
        raise SystemExit(f"No Jupytext lesson sources found in {SOURCE_DIR}")
    return {
        OUTPUT_DIR / f"{source.stem}.ipynb": render_notebook(source)
        for source in sources
    }


def build(*, check: bool) -> int:
    expected = _expected_notebooks()
    existing = set(OUTPUT_DIR.glob("*.ipynb")) if OUTPUT_DIR.is_dir() else set()
    stale = sorted(existing - set(expected))
    mismatches = []
    for output_path, rendered in expected.items():
        actual = (
            output_path.read_text(encoding="utf-8")
            if output_path.is_file()
            else None
        )
        if actual != rendered:
            mismatches.append(output_path)

    if check:
        if mismatches or stale:
            for output_path in mismatches:
                print(
                    f"out-of-date: {output_path.relative_to(ROOT)}",
                    file=sys.stderr,
                )
            for output_path in stale:
                print(f"stale: {output_path.relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(f"{len(expected)} generated notebooks are up to date")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output_path in stale:
        output_path.unlink()
    for output_path in mismatches:
        output_path.write_text(expected[output_path], encoding="utf-8")
    print(
        f"built {len(expected)} notebooks "
        f"({len(mismatches)} updated, {len(stale)} removed)"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic .ipynb files from Jupytext sources."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated notebooks are missing, stale, or out of date.",
    )
    args = parser.parse_args()
    raise SystemExit(build(check=args.check))


if __name__ == "__main__":
    main()
