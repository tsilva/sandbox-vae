# Canonical notebook sources

These Jupytext `py:percent` files are the source of truth for the executable
course. Edit them instead of the generated `.ipynb` files.

Build and verify:

```bash
uv run python scripts/build_notebooks.py
uv run python scripts/build_notebooks.py --check
```

Every lesson must preserve the learning loop:

1. State one learning objective.
2. Ask for a prediction before revealing evidence.
3. Expose relevant tensors, metrics, and images.
4. Change one declared experimental variable.
5. Name limitations and failure signatures.
6. End with an advancement gate.

Keep reusable model, objective, training, and plotting logic in
`src/latent_lab`.
