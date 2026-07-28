# Project Instructions

- Preserve the distinction between studies (scientific questions), recipes
  (executable configurations), reports (conclusions), and runs (generated
  evidence).
- Keep model-specific math in model/objective modules; keep the trainer generic.
- Do not put reusable training or model logic in notebooks.
- Every new model or objective needs shape, gradient, and tiny-overfit tests.
- Every experiment changes one declared variable unless the report explicitly
  justifies a coupled change.
- Use `uv` and keep `uv.lock` committed.
- Preserve `[tool.uv]` supply-chain constraints and the seven-day release delay.
- Do not commit datasets, raw run directories, or checkpoints.
