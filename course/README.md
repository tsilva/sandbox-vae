# From Bottlenecks to Discrete Latents

This is an executable, experiment-first course on autoencoders, variational
autoencoders, and VQ-VAEs. The goal is not to finish every notebook quickly.
The goal is to predict what a model should do, expose the relevant tensors and
images, notice contradictions, and explain the observed dynamics.

## How to use the course

Work through one lesson at a time:

1. Open the generated notebook.
2. Pause at every question, reason mentally, then expand its answer check.
3. Run one cell at a time.
4. Inspect tensors, metrics, reconstructions, error maps, and diagnostics.
5. Reconcile any result that contradicts your prediction.
6. Complete a copy of [WORKSHEET.md](WORKSHEET.md).
7. Continue only when you can pass the advancement gate without notes.

Start JupyterLab at Lesson 00:

```bash
uv sync --locked
uv run jupyter lab course/notebooks/00-laboratory.ipynb
```

The generated notebooks intentionally contain no saved outputs. Running them
locally produces your own evidence and may make the tracked notebook appear
modified. Rebuilding restores the deterministic, output-free distribution
form.

List the curriculum at any time:

```bash
uv run latent-lab course
uv run latent-lab course 00
```

The lesson command prints three paths:

- `guide`: concise Markdown reference.
- `notebook_source`: canonical Jupytext `py:percent` source.
- `notebook`: generated `.ipynb` for learners and GitHub.

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

## Notebook authoring contract

Edit only `course/notebook_sources/*.py`. Cells use Jupytext percent markers:

```python
# %% [markdown]
# ## Predict before running
# What result should this mechanism produce?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# State the causal expectation here.
# </details>

# %%
result = run_training(config)
```

Build and verify all generated notebooks with:

```bash
uv run python scripts/build_notebooks.py
uv run python scripts/build_notebooks.py --check
```

The builder assigns content-derived cell IDs, normalizes kernel metadata,
removes outputs and execution counts, detects stale notebooks, and writes
stable notebook JSON. Tests enforce one source and generated notebook per
curriculum lesson.

Conversion and execution are deliberately separate. Conversion must be
deterministic; training execution produces machine- and run-specific evidence.

Execute output-bearing copies without modifying committed notebooks:

```bash
uv run python scripts/execute_notebooks.py --profile smoke
uv run python scripts/execute_notebooks.py --profile foundations
uv run python scripts/execute_notebooks.py --lessons 06 07
```

Executed copies are written beneath ignored `runs/notebook-executions/`.

Reusable computation belongs in `src/latent_lab`. Notebook cells may configure
experiments, call the trainer, and interrogate returned tensors, but must not
define alternative models, objectives, or training loops.

Question cells must not contain blank fields that ask learners to type into the
distributed notebook. Use a concise question followed by a collapsed
`<details>` reasoning check. Learners may take notes wherever they prefer.

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

The notebooks use the same underlying probes and run artifacts, but render
them inline. Reconstruction labs show class-balanced inputs, aligned
reconstructions, per-example loss, and squared-error maps.
