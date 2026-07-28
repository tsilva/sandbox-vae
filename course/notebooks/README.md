# Generated Jupyter notebooks

The `.ipynb` files in this directory are deterministic builds of the canonical
Jupytext sources in `course/notebook_sources/`.

Do not hand-edit them. Rebuild with:

```bash
uv run python scripts/build_notebooks.py
```

The committed notebooks have no execution outputs. Learners generate evidence
locally by running cells; raw runs and checkpoints remain under ignored
`runs/`.

To execute a copy without modifying this directory:

```bash
uv run python scripts/execute_notebooks.py --profile smoke
```
