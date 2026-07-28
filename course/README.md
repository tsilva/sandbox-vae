# From Bottlenecks to Discrete Latents

This is an experiment-first course on autoencoders, variational
autoencoders, and VQ-VAEs. The goal is not to finish every command quickly.
The goal is to be able to predict what a run should do, notice when it does
something else, and explain why.

## How to use the course

Work through one lesson at a time:

1. Read only that lesson.
2. Write down your prediction before running anything.
3. Run the exact command.
4. Inspect metrics and every named figure.
5. Complete a copy of [WORKSHEET.md](WORKSHEET.md).
6. Explain the result without looking at the lesson.
7. Continue only when you pass the lesson's advancement gate.

List the curriculum at any time:

```bash
uv run latent-lab course
uv run latent-lab course 00
```

When you want interactive guidance, tell Codex:

```text
Start lesson 03 with me. Do not give me the conclusions before I make my
predictions. Here is the run directory from the previous lesson: ...
```

After a run, provide its printed `run_dir`. The run contains the exact config,
metrics, checkpoint, summary, and figures needed for discussion.

Use [GLOSSARY.md](GLOSSARY.md) for terminology and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) when a command or result behaves
unexpectedly.

## Course map

| Module | Lessons | Central question |
|---|---:|---|
| Laboratory | 00–01 | What counts as evidence that reconstruction learned something? |
| Autoencoders | 02–05 | What does a deterministic bottleneck preserve, and what does it fail to define? |
| VAEs | 06–09 | How does KL pressure create a sampleable continuous latent space? |
| VQ-VAEs | 10–12 | How do discrete codes learn, collapse, and become generative? |
| Synthesis | 13 | Which representation should we choose, and why? |

The exploratory pass uses one seed. Lesson 13 repeats only the important
finalists over three seeds. This separates learning/debugging from evidence
strong enough to support a conclusion.

## Evidence hierarchy

Treat evidence in this order:

1. Tensor, range, and gradient tests
2. Tiny-overfit behavior
3. Training and validation loss components
4. Fixed reconstructions
5. Model-specific diagnostics
6. Repeated-seed confirmation

A visually attractive sample does not override broken loss semantics. A lower
total loss does not imply a better model when the objectives differ.

## Expected artifacts

Training creates:

```text
runs/<recipe>/<timestamp>-seed-<seed>/
├── resolved-config.yaml
├── metrics.jsonl
├── summary.json
├── diagnostics.json          # VAE and VQ-VAE when applicable
├── checkpoint-best.pt
└── figures/
```

Use:

```bash
uv run latent-lab inspect <RUN_DIR>
```

The inspection command prints the best metrics, model-specific summary, final
epoch metrics, and all figure paths.
