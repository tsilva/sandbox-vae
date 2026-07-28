from __future__ import annotations

import argparse
import copy
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latent_lab.config import load_yaml, with_overrides
from latent_lab.training import (
    run_code_prior_training,
    run_mean_image_baseline,
    run_training,
)


def _train(args: argparse.Namespace) -> None:
    config = load_yaml(args.recipe)
    result = run_training(
        config,
        run_root=args.run_root,
        device_override=args.device,
    )
    print(f"run_dir={result.run_dir}")
    print(f"best_epoch={result.best_epoch}")
    print(f"best_validation_loss={result.best_validation_loss:.6f}")


def _baseline(args: argparse.Namespace) -> None:
    config = load_yaml(args.recipe)
    result = run_mean_image_baseline(
        config,
        run_root=args.run_root,
        device_override=args.device,
    )
    print(f"run_dir={result.run_dir}")
    print(f"mean_image_validation_mse={result.validation_mse:.6f}")


def _train_prior(args: argparse.Namespace) -> None:
    config = load_yaml(args.recipe)
    result = run_code_prior_training(
        config,
        args.vq_checkpoint,
        run_root=args.run_root,
        device_override=args.device,
    )
    print(f"run_dir={result.run_dir}")
    print(
        f"best_validation_cross_entropy={result.best_validation_loss:.6f}"
    )
    print(
        f"best_validation_perplexity={result.best_validation_perplexity:.3f}"
    )


def _study(args: argparse.Namespace) -> None:
    study = load_yaml(args.study)
    baseline = load_yaml(study["baseline"])
    seeds = (
        args.seeds
        if args.seeds is not None
        else study.get("seeds", [baseline["training"].get("seed", 0)])
    )
    variants: list[dict[str, Any]] = study.get("variants", [])
    if not variants:
        variants = [{"name": "baseline", "overrides": {}}]
    primary_metric = str(study.get("primary_metric", "validation/loss"))

    records: list[dict[str, Any]] = []
    for variant in variants:
        name = str(variant["name"])
        overrides = variant.get("overrides", {})
        for seed in seeds:
            config = with_overrides(baseline, overrides)
            config = copy.deepcopy(config)
            config["id"] = f"{study['id']}/{name}"
            config["training"]["seed"] = int(seed)
            result = run_training(
                config,
                run_root=args.run_root,
                device_override=args.device,
            )
            print(
                f"{name} seed={seed} "
                f"{primary_metric}="
                f"{result.best_validation_metrics[primary_metric]:.6f} "
                f"run_dir={result.run_dir}"
            )
            records.append(
                {
                    "variant": name,
                    "seed": int(seed),
                    "best_validation_loss": result.best_validation_loss,
                    "best_validation_metrics": result.best_validation_metrics,
                    "primary_metric": primary_metric,
                    "primary_metric_value": result.best_validation_metrics[
                        primary_metric
                    ],
                    "run_dir": str(result.run_dir),
                }
            )

    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record["variant"], []).append(
            record["primary_metric_value"]
        )
    aggregates = {
        name: {
            "runs": len(losses),
            "primary_metric": primary_metric,
            "mean": statistics.mean(losses),
            "standard_deviation": (
                statistics.stdev(losses) if len(losses) > 1 else 0.0
            ),
        }
        for name, losses in grouped.items()
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_dir = args.run_root / str(study["id"])
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_path = study_dir / f"study-summary-{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "study_id": study["id"],
                "question": study.get("question"),
                "hypothesis": study.get("hypothesis"),
                "primary_metric": primary_metric,
                "records": records,
                "aggregates": aggregates,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"study_summary={summary_path}")


def _inspect(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"No summary.json found in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if metrics_path.exists():
        lines = [
            line
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            print("\nlast_epoch_metrics:")
            print(json.dumps(json.loads(lines[-1]), indent=2, sort_keys=True))
    figures = sorted((run_dir / "figures").glob("*.png"))
    if figures:
        print("\nfigures:")
        for figure in figures:
            print(f"- {figure}")


def _find_course_manifest() -> Path:
    for root in [Path.cwd(), *Path.cwd().parents]:
        candidate = root / "course" / "curriculum.yaml"
        if candidate.exists():
            return candidate
    raise SystemExit("Run this command from inside the latent-lab repository.")


def _course(args: argparse.Namespace) -> None:
    manifest = load_yaml(_find_course_manifest())
    lessons = manifest["lessons"]
    if args.lesson is None:
        print(f"{manifest['title']}\n")
        for lesson in lessons:
            print(
                f"{lesson['id']:>2}  {lesson['title']} "
                f"({lesson['estimated_minutes']} min)"
            )
        print("\nUse `uv run latent-lab course <id>` to open a lesson.")
        return

    lesson = next(
        (item for item in lessons if str(item["id"]) == str(args.lesson)),
        None,
    )
    if lesson is None:
        raise SystemExit(f"Unknown lesson: {args.lesson}")
    manifest_path = _find_course_manifest()
    guide = manifest_path.parent / lesson["guide"]
    notebook_source = manifest_path.parent / lesson["notebook_source"]
    notebook = manifest_path.parent / lesson["notebook"]
    print(f"{lesson['id']} — {lesson['title']}")
    print(f"guide={guide}")
    print(f"notebook_source={notebook_source}")
    print(f"notebook={notebook}")
    print(f"estimated_minutes={lesson['estimated_minutes']}")
    print("\ncommands:")
    for command in lesson.get("commands", []):
        print(f"  {command}")
    print("\n---\n")
    print(guide.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent-lab",
        description="Run inspectable latent-model experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one recipe.")
    train.add_argument("recipe", type=Path)
    train.add_argument("--run-root", type=Path, default=Path("runs"))
    train.add_argument("--device", default=None)
    train.set_defaults(func=_train)

    baseline = subparsers.add_parser(
        "baseline", help="Evaluate the mean-image reconstruction baseline."
    )
    baseline.add_argument("recipe", type=Path)
    baseline.add_argument("--run-root", type=Path, default=Path("runs"))
    baseline.add_argument("--device", default=None)
    baseline.set_defaults(func=_baseline)

    prior = subparsers.add_parser(
        "train-prior",
        help="Train an autoregressive prior over a VQ-VAE checkpoint's codes.",
    )
    prior.add_argument("recipe", type=Path)
    prior.add_argument("vq_checkpoint", type=Path)
    prior.add_argument("--run-root", type=Path, default=Path("runs"))
    prior.add_argument("--device", default=None)
    prior.set_defaults(func=_train_prior)

    study = subparsers.add_parser("study", help="Run all variants in a study.")
    study.add_argument("study", type=Path)
    study.add_argument("--run-root", type=Path, default=Path("runs"))
    study.add_argument("--device", default=None)
    study.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override the study's seed list, e.g. --seeds 0 or --seeds 0 1 2.",
    )
    study.set_defaults(func=_study)

    inspect = subparsers.add_parser(
        "inspect", help="Print a run summary, final metrics, and figure paths."
    )
    inspect.add_argument("run_dir", type=Path)
    inspect.set_defaults(func=_inspect)

    course = subparsers.add_parser(
        "course", help="List the curriculum or show one lesson."
    )
    course.add_argument("lesson", nargs="?", default=None)
    course.set_defaults(func=_course)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
