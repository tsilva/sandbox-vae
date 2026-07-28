# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 01 — Why a reconstruction loss needs a baseline
#
# **Learning objective:** make the number called “MSE” interpretable by comparing
# it with predictions that learn no input-dependent representation.
#
# For image $x$ and reconstruction $\hat{x}$:
#
# $$
# \operatorname{MSE}(x,\hat{x})
# = \frac{1}{CHW}\sum_{c,h,w}(x_{chw}-\hat{x}_{chw})^2
# $$
#
# **No model is trained in this lesson.** We are constructing deliberately weak
# predictors so that later model losses have a reference point. The central
# question is:
#
# > How well can we score while learning nothing about the particular input?

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional

from latent_lab.config import load_yaml
from latent_lab.course import balanced_class_batch, repository_root
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.training import (
    compute_mean_image,
    constant_reconstruction_errors,
)

ROOT = repository_root()
config = load_yaml(ROOT / "recipes/ae/ae-001-linear.yaml")
train_loader, validation_loader, spec = build_dataloaders(
    config["dataset"], config["training"]
)
names = class_names(config["dataset"]["name"])
spec

# %% [markdown]
# ## Begin with one number, not 784 pixels
#
# MSE follows three operations:
#
# 1. Find each prediction error: $x-\hat{x}$.
# 2. Square it, making every error nonnegative and penalizing large misses more.
# 3. Average the squared errors.
#
# Consider a “two-pixel image”:
#
# $$
# x=[0,1],\qquad \hat{x}=[0,0.5]
# $$
#
# The first pixel is exact. The second misses by $0.5$, so its squared error is
# $0.25$. Averaging both pixels gives $\operatorname{MSE}=0.125$.
#
# Predict the MSE below before running the cell.

# %%
toy_target = torch.tensor([0.0, 1.0])
toy_prediction = torch.tensor([0.0, 0.5])
toy_errors = toy_target - toy_prediction
toy_squared_errors = toy_errors.square()
toy_mse = toy_squared_errors.mean()

print("errors:", toy_errors.tolist())
print("squared errors:", toy_squared_errors.tolist())
print("MSE:", float(toy_mse))

# %% [markdown]
# Fashion-MNIST has $1\times28\times28=784$ pixel values per image. Its MSE is
# the same calculation over 784 values instead of two.
#
# This averaging convention matters. A **mean** pixel loss and a **sum** pixel
# loss describe the same errors on very different numeric scales.
#
# <details>
# <summary>Quick check: what happens if every pixel misses by 0.1?</summary>
#
# Every squared error is $0.1^2=0.01$, so their mean is also $0.01$. Image size
# does not change mean MSE when every pixel has the same error.
# </details>

# %% [markdown]
# ## First look at the dataset
#
# Before discussing loss, establish the data contract:
#
# - One grayscale channel
# - $28\times28$ pixels
# - Pixel values in $[0,1]$
# - Ten garment classes
#
# A class-balanced view prevents the first shuffled batch from defining your
# mental picture of the dataset.

# %%
class_images, class_labels = balanced_class_batch(
    validation_loader, spec.num_classes
)
print(
    "shape:",
    tuple(class_images.shape),
    "range:",
    (float(class_images.min()), float(class_images.max())),
)
_ = plot_image_grid(
    class_images,
    labels=class_labels,
    class_names=names,
    title="One validation example from each Fashion-MNIST class",
)

# %% [markdown]
# The printed shape is `[10, 1, 28, 28]`:
#
# - `10`: one selected example for each class
# - `1`: one grayscale channel
# - `28, 28`: image height and width
#
# The pictures also reveal a dataset shortcut: much of every image is black
# background. A predictor can be correct on many pixels without recognizing a
# garment.
#
# **Stop and explain:** why would “predict black everywhere” be a stronger
# baseline here than it would be for photographs that fill the frame?
#
# <details>
# <summary>Reveal the reasoning</summary>
#
# Fashion-MNIST centers a small grayscale object on a black canvas, so many
# target pixels are exactly or nearly zero. Predicting zero gets those pixels
# right without identifying the object. In a photograph that fills the frame,
# far fewer pixels would be correctly predicted by a single black value.
# </details>

# %% [markdown]
# ## Predict before revealing the mean
#
# Write down:
#
# 1. Which shared shapes will survive averaging 60,000 garments?
# 2. Will any individual class remain recognizable?
# 3. Why might the result still receive a nonterrible MSE?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The average should preserve a centered, vertically oriented garment-like
# glow, with brighter regions where many classes overlap. Class-specific edges
# should blur together, so no single class should be reliably recognizable.
# The score can remain nonterrible because common black background and common
# object locations account for many pixels.
# </details>

# %% [markdown]
# ## Why the mean is the best constant prediction
#
# “Constant” means the prediction cannot change when the input changes. Imagine
# choosing one value $a$ for one fixed pixel location. Across $N$ training
# images, that pixel contains values $x_1,\ldots,x_N$. Its average squared loss
# is:
#
# $$
# L(a)=\frac{1}{N}\sum_{i=1}^{N}(x_i-a)^2
# $$
#
# Differentiate with respect to the prediction:
#
# $$
# \frac{dL}{da}
# =\frac{2}{N}\sum_{i=1}^{N}(a-x_i)
# $$
#
# At the minimum the derivative is zero:
#
# $$
# Na-\sum_i x_i=0
# \quad\Longrightarrow\quad
# a=\frac{1}{N}\sum_i x_i
# $$
#
# So the best constant value for each pixel is that pixel’s training-set mean.
# Applying this argument independently at all 784 locations produces the mean
# training image.
#
# This is not a fact about neural networks. It follows directly from squared
# error.

# %%
device = torch.device("cpu")
mean_image, train_examples = compute_mean_image(train_loader, device)
print("training examples averaged:", train_examples)
_ = plot_image_grid(
    mean_image,
    title="Pixelwise mean training image",
    max_items=1,
)

# %% [markdown]
# The mean was computed from **training images only**. Later cells evaluate it
# on the separate validation set. Computing the predictor from validation
# images would leak information from the data used to judge it.
#
# ## Verify the derivation at one pixel
#
# The center pixel takes different values across garments. We can try every
# constant prediction from 0 to 1 and measure its training MSE. The curve should
# reach its minimum at the empirical mean.

# %%
center_values = torch.cat(
    [inputs[:, 0, 14, 14] for inputs, _labels in train_loader]
)
candidate_values = torch.linspace(0, 1, 101)
candidate_losses = (
    center_values[:, None] - candidate_values[None, :]
).square().mean(dim=0)
best_candidate = candidate_values[candidate_losses.argmin()]
empirical_mean = center_values.mean()

figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(candidate_values, candidate_losses)
axis.axvline(
    float(empirical_mean),
    color="red",
    linestyle="--",
    label=f"pixel mean = {empirical_mean:.3f}",
)
axis.set(
    xlabel="Constant prediction for center pixel",
    ylabel="Training MSE at that pixel",
    title="Squared error is minimized by the mean",
)
axis.legend()
axis.grid(alpha=0.25)
figure.tight_layout()

print("best value on the grid:", float(best_candidate))
print("empirical pixel mean:", float(empirical_mean))

# %% [markdown]
# The grid search and the derivative agree. The tiny difference, if any, comes
# from testing candidate values only in increments of 0.01.
#
# <details>
# <summary>Why is the mean image blurry?</summary>
#
# A pixel is bright only when some garment occupies that location. Averaging
# shoes, trousers, shirts, and bags mixes mutually incompatible shapes. The
# result represents where Fashion-MNIST objects are commonly bright, not a
# coherent garment.
# </details>

# %% [markdown]
# ## The input-independent reconstruction
#
# Under squared error, the best constant prediction is the training-set mean.
# “Best constant” is the important qualifier: it says nothing about encoding
# the input.

# %%
constant_predictions = mean_image.expand(class_images.shape[0], -1, -1, -1)
_ = plot_reconstruction_grid(
    class_images,
    constant_predictions,
    labels=class_labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %% [markdown]
# Every reconstruction row is identical. If you can infer the input class from
# a prediction, you are using information that the predictor itself did not use.
#
# Read the three rows as:
#
# 1. **Input:** what the predictor was asked to reconstruct.
# 2. **Reconstruction:** what it predicted.
# 3. **Squared error:** where the prediction and target disagree; brighter means
#    a larger contribution to MSE.
#
# Inspect the squared-error row. Background pixels dominate the image and are
# predicted well; errors concentrate around class-specific silhouettes and
# details.

# %%
print(
    "largest difference between any two baseline predictions:",
    float(
        (
            constant_predictions
            - constant_predictions[0:1]
        ).abs().max()
    ),
)

# %% [markdown]
# That value is exactly zero. The prediction is invariant to the input.
#
# This is the critical failure the scalar loss does not announce: a baseline
# can receive a finite, apparently respectable score while discarding every bit
# of information about which example it was given.
#
# ## Why black is not as absurd as it sounds
#
# Let us measure how many validation pixels are nearly black, then separate the
# mean-image baseline’s squared error over dark and non-dark target pixels.

# %%
dark_threshold = 0.05
dark_pixels = 0
bright_pixels = 0
dark_squared_error = 0.0
bright_squared_error = 0.0

for inputs, _labels in validation_loader:
    predictions = mean_image.expand(inputs.shape[0], -1, -1, -1)
    squared_error = (inputs - predictions).square()
    dark_mask = inputs <= dark_threshold
    bright_mask = ~dark_mask
    dark_pixels += int(dark_mask.sum())
    bright_pixels += int(bright_mask.sum())
    dark_squared_error += float(squared_error[dark_mask].sum())
    bright_squared_error += float(squared_error[bright_mask].sum())

total_pixels = dark_pixels + bright_pixels
print(f"nearly-black target pixels: {dark_pixels / total_pixels:.1%}")
print(
    "contribution to overall MSE from nearly-black targets:",
    f"{dark_squared_error / total_pixels:.6f}",
)
print(
    "contribution to overall MSE from other targets:",
    f"{bright_squared_error / total_pixels:.6f}",
)

# %% [markdown]
# The two contributions add to the overall mean-image MSE. Notice the
# denominators: both are divided by **all** validation pixels, so they describe
# how much each region contributes to the final average.
#
# A dark-background dataset gives an all-black predictor many easy correct
# pixels. It still makes large errors on garment pixels, so the mean image can
# improve substantially by predicting where garments tend to occur. Neither can
# determine which garment is present.

# %%
mean_errors = constant_reconstruction_errors(
    validation_loader, mean_image, device
)
black_image = torch.zeros_like(mean_image)
black_errors = constant_reconstruction_errors(
    validation_loader, black_image, device
)

print(f"all-black validation MSE: {black_errors.mean():.6f}")
print(f"mean-image validation MSE: {mean_errors.mean():.6f}")
print(
    "relative improvement over black:",
    f"{(1 - mean_errors.mean() / black_errors.mean()) * 100:.1f}%",
)

# %% [markdown]
# The comparison gives each number meaning:
#
# - All-black asks how far background sparsity alone can take us.
# - Mean-image asks how well the best input-independent MSE predictor performs.
# - A future autoencoder must improve on the mean-image reference using
#   information from its input.
#
# <details>
# <summary>Does beating the all-black baseline prove the model uses its input?</summary>
#
# No. The mean image beats black while remaining constant. Beating a weaker
# baseline does not establish input dependence.
# </details>
#
# <details>
# <summary>Does beating the mean baseline prove a useful latent representation?</summary>
#
# Not by itself. It shows improved pixel prediction under the same evaluation
# contract. We must also inspect whether reconstructions change with inputs,
# whether meaningful structure is retained, and what the bottleneck encodes.
# </details>

# %% [markdown]
# ## Do not let the average hide the distribution
#
# One mean MSE conceals easy and hard examples. Examine the distribution and
# class-conditioned errors before treating it as a complete description.

# %%
validation_labels = torch.cat(
    [labels for _inputs, labels in validation_loader]
)
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(mean_errors.numpy(), bins=40)
axes[0].axvline(float(mean_errors.mean()), color="red", linestyle="--")
axes[0].set(
    title="Per-example mean-image error",
    xlabel="MSE",
    ylabel="Validation examples",
)
class_mse = [
    float(mean_errors[validation_labels == label].mean())
    for label in range(spec.num_classes)
]
axes[1].barh(names, class_mse)
axes[1].set(
    title="The same baseline is not equally good for every class",
    xlabel="MSE",
)
figure.tight_layout()

# %% [markdown]
# The histogram asks whether the reported mean describes most examples or hides
# a wide spread. The class bars ask which shapes resemble the dataset-wide
# average most closely.
#
# Keep observation and interpretation separate:
#
# - **Observation:** classes have different average pixel MSE.
# - **Interpretation:** the constant image happens to approximate some class
#   silhouettes better; it has not learned those class concepts.
#
# ## A boundary of MSE: location matters more than meaning
#
# MSE compares pixels at matching coordinates. Shift a recognizable garment by
# one pixel and many formerly aligned pixels become errors, even though a human
# still sees the same object.

# %%
original = class_images[0:1]
shifted = functional.pad(
    original[:, :, :, :-1],
    (1, 0, 0, 0),
)
shifted_mse = (original - shifted).square().mean()

_ = plot_reconstruction_grid(
    original,
    shifted,
    max_items=1,
    include_error=True,
)
print("MSE after a one-pixel horizontal shift:", float(shifted_mse))

# %% [markdown]
# This does not make MSE wrong. It identifies the narrow question MSE answers:
#
# > How accurately did the prediction reproduce pixel intensities at the same
# > locations?
#
# It does **not** directly answer whether two images depict the same semantic
# object. That is why the course always pairs numeric reconstruction error with
# aligned visual evidence.
#
# ## Build the conclusion yourself
#
# Complete these sentences before opening the checks:
#
# 1. The all-black baseline performs better than I might expect because …
# 2. The mean image beats other constant images because …
# 3. The mean-image MSE does not prove representation learning because …
# 4. A trained autoencoder will provide stronger evidence if …
#
# <details>
# <summary>Self-check</summary>
#
# 1. Most Fashion-MNIST pixels are dark background.
# 2. Squared error at each pixel is minimized by that pixel’s training mean.
# 3. The same prediction is emitted for every input; no input information is
#    encoded.
# 4. It beats the baseline under identical loss semantics **and** produces
#    visibly input-dependent reconstructions that retain meaningful structure.
# </details>

# %% [markdown]
# ## What you should have learned
#
# The number now has a reference:
#
# - An all-black predictor exploits background sparsity.
# - The mean image is the MSE-optimal predictor among all constant images.
# - Neither predictor contains an encoder or an input-dependent representation.
# - A trained autoencoder must beat these baselines **and** visibly change its
#   output with the input.
#
# ## Advancement gate
#
# Use these as self-check questions. Reason through them, then expand the check:
#
# 1. Manually compute MSE for two two-pixel vectors.
# 2. Derive why the mean minimizes constant-prediction squared error.
# 3. Explain how Fashion-MNIST background helps the black baseline and why the
#    per-pixel mean still improves on it.
# 4. Explain why “validation MSE = 0.05” is incomplete without the dataset,
#    pixel range, reduction convention, error distribution, and baseline.
# 5. Name one numeric and one visual observation that would establish stronger
#    input-dependent reconstruction evidence in Lesson 02.
#
# <details>
# <summary>Reveal the advancement check</summary>
#
# 1. Subtract corresponding values, square each difference, and average them.
# 2. For one pixel, setting the derivative of average squared loss to zero gives
#    the empirical pixel mean; the full mean image applies this independently
#    at every location.
# 3. Dark background supplies many easy correct pixels to black. The mean image
#    further predicts common object locations, reducing error while remaining
#    input-independent.
# 4. The number needs its dataset, pixel range, per-pixel mean reduction,
#    distribution across examples/classes, and input-independent reference
#    losses.
# 5. Numeric: validation MSE below the mean-image baseline under identical loss
#    semantics. Visual: aligned outputs change with their inputs and preserve
#    recognizable input-specific structure.
# </details>
#
# If any answer feels vague, return to the probe that produced its evidence
# rather than memorizing the summary.
