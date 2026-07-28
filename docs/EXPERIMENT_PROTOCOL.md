# Experiment Protocol

Each study must state:

1. The question.
2. A falsifiable hypothesis.
3. The baseline recipe.
4. The single independent variable.
5. The controlled variables.
6. The evidence needed to answer the question.
7. The decision made from the result.

Before trusting a new model:

- Confirm tensor shapes and ranges.
- Confirm the intended parameters receive gradients.
- Overfit a tiny dataset.
- Inspect a fixed reconstruction batch.
- Save the fully resolved configuration.

Use one seed while debugging. Use three or more seeds only when promoting a
result into a durable conclusion.

Do not compare total losses across model families without decomposing them.
AE, VAE, and VQ-VAE total objectives contain different terms and scales.

