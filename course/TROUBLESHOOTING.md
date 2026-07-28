# Course Troubleshooting

## The command cannot find a recipe or course lesson

Run commands from the repository root:

```bash
cd ~/repos/tsilva/sandbox-vae
```

## Fashion-MNIST download fails

Confirm network access and retry. The dataset is cached under `data/` after the
first successful download.

## MPS produces an unsupported-operation error

Force CPU for the affected command:

```bash
uv run latent-lab train <RECIPE> --device cpu
```

Record the device change in the worksheet. Do not compare throughput across
devices as though it were a model-quality result.

## A run seems slow after the last epoch

Post-training diagnostics traverse validation data and render latent, KL, or
codebook figures. This is expected. The CLI prints the run directory only after
artifacts finish.

## AE, VAE, and VQ-VAE losses have very different magnitudes

Expected:

- AE and VQ-VAE recipes use mean pixel MSE.
- VAE recipes use summed-per-example binary cross-entropy plus KL.

Compare named components under matched conventions, never unrelated totals.

## VAE KL is almost zero

Check reconstruction quality and `kl-per-dimension.png`. Low KL plus inactive
dimensions and input-insensitive reconstructions suggests collapse. Low KL
alone is not a diagnosis.

## VQ-VAE reports many dead codes

Inspect global `diagnostics.json`, not only the per-batch training metric. Then
test one variable at a time: codebook size, commitment weight, update method,
or dead-code reinitialization.

## I lost a printed run directory

Find recent summaries:

```bash
find runs -name summary.json -print | sort
```

## A result contradicts the lesson prediction

Do not tune immediately. Check:

1. Resolved config
2. Input and output ranges
3. Named loss components
4. Correct checkpoint and figures
5. Whether the baseline actually exhibited the claimed failure

Then propose the smallest experiment that distinguishes implementation error,
optimization failure, and a false hypothesis.

