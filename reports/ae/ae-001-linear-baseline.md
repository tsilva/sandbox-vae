# AE-001 — Linear baseline

## Question

How much better is a linear undercomplete autoencoder than predicting the mean
training image?

## Hypothesis

The latent bottleneck will capture recurring Fashion-MNIST structure and beat
the mean-image validation MSE.

## Controlled variables

- Fashion-MNIST split and `[0, 1]` input range
- Mean-squared reconstruction error
- Training duration and optimizer

## Results

Pending.

## Interpretation

Pending.

## Decision

Do not begin the nonlinearity study until the implementation passes the tiny
overfit test and the fixed reconstruction grid is visually plausible.

