from sandbox_autoencoders.experiments.shared import run_experiment_main


def main(argv: list[str] | None = None):
    return run_experiment_main("exp_06_kl_stability_and_collapse", argv)


if __name__ == "__main__":
    main()
