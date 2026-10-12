# JiT-Inspired Diffusion Model for CIFAR-10

A reduced-scale, JiT-inspired diffusion model for CIFAR-10 image generation, implemented in PyTorch. This project explores patch-based Transformer denoising, compares Patch-4 and Patch-8 configurations, and includes an ablation study on time embedding.

> **Project scope:** This is an educational, reduced-scale implementation inspired by *“Back to Basics: Let Denoising Generative Models Denoise”*. It is **not** an exact reproduction of the original paper's ImageNet-scale experiments.

## Highlights

- CIFAR-10 image loading and preprocessing
- Patch-based Transformer denoiser
- Clean-image (`x`) prediction with a velocity-space MSE objective
- Patch-4 vs. Patch-8 comparison
- Time-embedding ablation
- Training-loss visualization
- Generated-image and denoising visualizations
- Model checkpoint saving

## Model and Training Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 |
| Image size | 32 × 32 RGB |
| Patch sizes | 4 and 8 |
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Feed-forward dimension | 512 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Batch size | 128 |
| Training epochs | 10 per configuration |

## Results

The table below records the results from the current notebook run.

| Experiment | Initial Training Loss | Final Training Loss | Test Loss |
|---|---:|---:|---:|
| Patch-4 (with time embedding) | 0.358753 | 0.211430 | 0.204914 |
| Patch-8 (with time embedding) | 0.432867 | 0.206700 | 0.201217 |
| Patch-4 (without time embedding) | N/A | 0.216322 | 0.208269 |

**Observation:** Patch-8 achieved the lowest recorded test loss (`0.201217`) in this run. Removing time embedding increased the Patch-4 test loss from `0.204914` to `0.208269`. These results come from one recorded run per configuration and should not be treated as statistically conclusive.

## Method Overview

Given a clean image `x` and Gaussian noise `eps`, the notebook constructs a noisy input:

```text
z_t = t * x + (1 - t) * eps
```

The model predicts the clean image:

```text
x_hat = f_theta(z_t, t)
```

The implementation derives a velocity estimate and target using a denominator clamp:

```text
v_hat = (x_hat - z_t) / max(1 - t, 0.05)
v     = (x - z_t) / max(1 - t, 0.05)
```

Training minimizes mean-squared error between the predicted and target velocity values. See the notebook for the exact implementation details.

## Repository Contents

Suggested repository structure:

```text
jit-cifar10-diffusion/
├── JiT Project.ipynb
├── README.md
├── requirements.txt
├── checkpoints/             # Optional; model weights may be large
└── results/                  # Optional; figures and experiment outputs
```

The exact file structure may differ depending on how the notebook is exported and organized.

## Run in Google Colab

1. Upload `JiT Project.ipynb` to Google Colab, or open the notebook from your GitHub repository.
2. In Colab, select **Runtime → Change runtime type** and choose a GPU if one is available.
3. Run the notebook cells in order.
4. Allow the CIFAR-10 dataset to download when prompted by the notebook.
5. Review the training curves, test-loss results, generated samples, denoising visualizations, and ablation results.

**Important:** The notebook's dependencies and execution order are the source of truth. If you move code into separate Python files, update the setup instructions accordingly.

## Reproducibility Notes

- The reported results are from the current notebook run.
- The recorded configuration uses 10 epochs and batch size 128.
- The results can vary due to random initialization, data order, and stochastic time/noise sampling.
- For a stronger comparison, run each configuration with multiple random seeds and report the mean and standard deviation.
- No FID benchmark is reported in the current results.

## Comparison with the Original JiT Paper

The original JiT paper evaluates larger models on high-resolution ImageNet settings and reports generative metrics such as FID and Inception Score. This project uses a compact model on 32 × 32 CIFAR-10 images and reports test v-loss.

Because the dataset, model scale, training setup, and metrics differ, the values in this repository cannot be directly compared numerically with the original paper's FID or Inception Score. This repository should be described as **JiT-inspired**, not as an exact reproduction.

## Limitations

- Small model and low-resolution images
- Only 10 training epochs per configuration
- One recorded run per configuration
- Generated samples may be blurry or lack detail
- No standardized FID evaluation in the current experiment

## Future Work

- Train for more epochs
- Repeat experiments with multiple random seeds
- Report mean and standard deviation
- Save and compare fixed-seed sample grids
- Evaluate generated samples with a documented FID protocol
- Improve training and sampling configurations

## Reference

Li, T., and He, K., *“Back to Basics: Let Denoising Generative Models Denoise,”* arXiv:2511.13720v2, 2026.

---

If you use or extend this project, please preserve the distinction between the reduced-scale implementation and the original paper's full-scale experiments.
