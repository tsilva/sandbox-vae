from sandbox_autoencoders.experiments.shared import run_experiment_main


def main(argv: list[str] | None = None):
    return run_experiment_main("exp_08_real_image_vae", argv)


if __name__ == "__main__":
    main()
