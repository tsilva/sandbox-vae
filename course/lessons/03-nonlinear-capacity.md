# Lesson 03 — Nonlinearity and bottleneck capacity

## Learning objective

Separate two sources of model capacity:

1. The geometry representable by nonlinear transformations.
2. The amount of information that fits through the latent bottleneck.

## Experiment A: nonlinearity

Predict whether adding hidden ReLU layers at the same latent dimension will
lower validation MSE. Also predict whether the comparison is perfectly fair:
the nonlinear model has many more parameters.

```bash
uv run latent-lab study studies/ae/ae-002-nonlinearity.yaml --seeds 0
```

Open the printed study summary and both run directories. Compare:

- Parameter count
- Validation reconstruction MSE
- Reconstruction grids
- Latent scatter

The experiment isolates architecture choice but not parameter count. Say that
explicitly in the worksheet.

## Experiment B: bottleneck size

Before running, sketch expected validation MSE against latent sizes
2, 8, 32, and 128.

```bash
uv run latent-lab study studies/ae/ae-003-latent-capacity.yaml --seeds 0
```

Look for diminishing returns rather than assuming every additional latent
dimension is equally valuable. Inspect the latent-2 run directly: because its
latent is already two-dimensional, the scatter plot is not a PCA projection.

## Interpret carefully

- Lower reconstruction error means less pixel information was discarded.
- It does not prove the representation is more useful for classification,
  disentangled, robust, or sampleable.
- A large or overcomplete latent can make the task close to identity copying.
- A tiny latent can impose useful abstraction or simply destroy information.

## Optional confirmation

After choosing the two most scientifically interesting variants:

```bash
uv run latent-lab study studies/ae/ae-002-nonlinearity.yaml --seeds 0 1 2
```

Do not repeat every variant automatically. Confirm the comparison that would
change your conclusion.

## Advancement gate

Explain why “bigger latent is better” is incomplete. Your answer must mention
the task objective, compression, downstream usefulness, and the danger of an
identity mapping.

