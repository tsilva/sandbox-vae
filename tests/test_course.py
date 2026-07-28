import subprocess
import sys
from pathlib import Path

import nbformat
import torch

from latent_lab.config import load_yaml, validate_recipe, with_overrides
from latent_lab.data.datasets import dataset_spec
from latent_lab.models import build_model
from latent_lab.objectives import build_objective


ROOT = Path(__file__).parents[1]


def test_course_manifest_points_to_existing_guides() -> None:
    manifest = load_yaml(ROOT / "course" / "curriculum.yaml")
    lessons = manifest["lessons"]
    assert [lesson["id"] for lesson in lessons] == [
        f"{index:02d}" for index in range(14)
    ]
    for lesson in lessons:
        assert (ROOT / "course" / lesson["guide"]).is_file()
        source = ROOT / "course" / lesson["notebook_source"]
        notebook_path = ROOT / "course" / lesson["notebook"]
        assert source.is_file()
        assert notebook_path.is_file()
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["latent_lab"]["source"] == str(
            source.relative_to(ROOT)
        )
        assert all(
            not cell.get("outputs")
            for cell in notebook.cells
            if cell.cell_type == "code"
        )
        assert lesson["commands"]


def test_generated_notebooks_are_deterministic_and_current() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_notebooks.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_notebook_sources_preserve_the_learning_loop() -> None:
    sources = sorted((ROOT / "course/notebook_sources").glob("[0-9][0-9]-*.py"))
    assert len(sources) == 14
    for source in sources:
        content = source.read_text(encoding="utf-8")
        normalized = content.lower()
        assert "learning objective" in normalized
        assert "predict" in normalized
        assert (
            "advancement gate" in normalized
            or "completion criterion" in normalized
        )


def test_all_latent_model_recipes_build_and_backpropagate() -> None:
    recipe_paths = [
        *sorted((ROOT / "recipes" / "ae").glob("*.yaml")),
        *sorted((ROOT / "recipes" / "vae").glob("*.yaml")),
        *sorted((ROOT / "recipes" / "vqvae").glob("*.yaml")),
        *sorted((ROOT / "recipes" / "smoke").glob("*.yaml")),
    ]
    assert recipe_paths
    for path in recipe_paths:
        config = load_yaml(path)
        validate_recipe(config)
        spec = dataset_spec(config["dataset"])
        model = build_model(config["model"], spec)
        objective = build_objective(config["objective"])
        inputs = torch.rand(2, *spec.input_shape)
        losses = objective(model(inputs), inputs)
        assert torch.isfinite(losses["loss"]), path
        losses["loss"].backward()


def test_all_studies_resolve_their_variants() -> None:
    study_paths = sorted((ROOT / "studies").glob("*/*.yaml"))
    assert study_paths
    for path in study_paths:
        study = load_yaml(path)
        baseline_path = ROOT / study["baseline"]
        assert baseline_path.is_file(), path
        baseline = load_yaml(baseline_path)
        validate_recipe(baseline)
        for variant in study["variants"]:
            resolved = with_overrides(
                baseline, variant.get("overrides", {})
            )
            validate_recipe(resolved)
