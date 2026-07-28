# Project Instructions

- Preserve the distinction between studies (scientific questions), recipes
  (executable configurations), reports (conclusions), and runs (generated
  evidence).
- Keep model-specific math in model/objective modules; keep the trainer generic.
- Do not put reusable training or model logic in notebooks.
- Treat `course/notebook_sources/*.py` as the canonical Jupytext lessons. Do not
  hand-edit generated `course/notebooks/*.ipynb`; rebuild them with
  `uv run python scripts/build_notebooks.py`.
- Keep notebook conversion deterministic and output-free. Generated training
  evidence belongs in ignored `runs/`, not committed notebook cells.
- Every new model or objective needs shape, gradient, and tiny-overfit tests.
- Every experiment changes one declared variable unless the report explicitly
  justifies a coupled change.
- Use `uv` and keep `uv.lock` committed.
- Preserve `[tool.uv]` supply-chain constraints and the seven-day release delay.
- Do not commit datasets, raw run directories, or checkpoints.
