# Lesson 13 — Final comparison and teach-back

## Learning objective

Choose among AE, VAE, and VQ-VAE based on representation and generation
requirements rather than one universal leaderboard.

## First verify the laboratory

```bash
uv run pytest
uv lock --check
git diff --check
```

## Select finalists

Choose:

- One deterministic AE
- One VAE beta/remedy configuration
- One VQ-VAE codebook/commitment configuration
- The learned prior paired with that exact VQ-VAE

Record exact run directories and resolved configs. Do not compare a final
checkpoint from one configuration with diagnostics from another.

## Build the comparison

Complete this table:

| Question | AE | VAE | VQ-VAE |
|---|---|---|---|
| Latent type | | | |
| Encoder output | | | |
| Bottleneck mechanism | | | |
| Reconstruction evidence | | | |
| Utilization diagnostic | | | |
| Can sample directly? | | | |
| Additional prior required? | | | |
| Main collapse mode | | | |
| Best use case | | | |

Do not compare total objectives numerically. For pixel reconstruction, compare
only runs with the same data range, observation model, and reduction.

## Confirm only important claims

For any conclusion that depends on a small numerical difference, rerun the two
relevant variants:

```bash
uv run latent-lab study <STUDY_FILE> --seeds 0 1 2
```

Report the mean, variation, qualitative consistency, and any seed-specific
failure. A single run is sufficient for understanding mechanics but weak
evidence for close model rankings.

## Teach-back

Without notes, give a ten-minute explanation covering:

1. Why a deterministic AE can reconstruct but lacks a known sampling prior.
2. How reparameterization enables VAE gradients.
3. Why KL creates a rate–distortion tradeoff.
4. How posterior collapse appears in metrics and images.
5. How VQ nearest-neighbor selection becomes trainable.
6. Why codebook size differs from effective usage.
7. Why VQ-VAE requires a token prior for coherent generation.

Then answer:

- Which model would you choose for denoising?
- Which for smooth continuous interpolation?
- Which for learning reusable discrete visual tokens?
- Which failure metric would you inspect first for each?

## Completion criterion

The course is complete when you can predict the major metric and visual changes
before running a beta, bottleneck, or codebook ablation—and can explain a
contradictory result by proposing a controlled next experiment rather than
changing several things at once.

