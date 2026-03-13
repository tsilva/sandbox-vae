from pathlib import Path

from sandbox_autoencoders.experiments.exp_02_vanilla_vae_baseline import main


def _run_once(output_dir: Path):
    summary = main(
        [
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(output_dir / "data"),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--seed",
            "11",
            "--wandb-mode",
            "disabled",
        ]
    )
    return summary["variants"][0]["final_val_loss"]


def test_exp02_reproducible_on_cpu(tmp_path):
    first = _run_once(tmp_path / "run_a")
    second = _run_once(tmp_path / "run_b")
    assert abs(first - second) < 1e-8
