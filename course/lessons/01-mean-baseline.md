# Lesson 01 — Reconstruction and the mean-image baseline

> Interactive lab: [open the generated notebook](../notebooks/01-mean-baseline.ipynb).
> Authors edit the [Jupytext source](../notebook_sources/01-mean-baseline.py).

## Learning objective

Learn why a loss number becomes meaningful only relative to a baseline.

For image $x$ and reconstruction $\hat{x}$, the mean-squared error is:

$$
\operatorname{MSE}(x,\hat{x}) =
\frac{1}{CHW}\sum_{c,h,w}(x_{chw}-\hat{x}_{chw})^2
$$

Under MSE, a model that ignores its input can minimize expected loss by
predicting the mean training image.

## Predict before running

Write down:

- What the Fashion-MNIST mean image will resemble.
- Whether different classes will be recognizable.
- Why this baseline can achieve a nonterrible MSE despite learning no
  input-dependent representation.

## Run

```bash
uv run latent-lab baseline recipes/ae/ae-001-linear.yaml
uv run latent-lab inspect <PRINTED_RUN_DIR>
```

## Inspect

- `summary.json`: record `validation_mse`.
- `figures/mean-image-baseline.png`: originals are on top; the same mean
  prediction appears below every image.

This number is the first reference point. A trained autoencoder that cannot
beat it has not demonstrated useful reconstruction.

## Think experimentally

MSE is sensitive to pixel distance, not semantic correctness. A blurry average
can score better than a sharp image shifted one pixel. Therefore:

- Numeric reconstruction error answers a narrow pixel-level question.
- Fixed reconstruction grids expose blur, lost shape, and class confusion.
- Later generative models need additional diagnostics.

## Advancement gate

Explain why “validation MSE = 0.05” is not interpretable without the input
range, reduction convention, dataset, and baseline. Record the baseline in a
worksheet for use in Lesson 02.
