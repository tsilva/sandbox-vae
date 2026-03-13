from sandbox_autoencoders.experiments.shared import run_experiment_main


def main(argv: list[str] | None = None):
    return run_experiment_main("exp_01_autoencoder_baseline", argv)


if __name__ == "__main__":
    main()
