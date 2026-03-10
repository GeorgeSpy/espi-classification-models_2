# Repository Scope and Thesis Mapping

This note clarifies how the public repositories map to the thesis workflow.

## What this repository contains

`espi-classification-models_2` is the **classification and evaluation repository**. It contains:

- Random Forest training
- CNN baseline training
- LOBO / LODO robustness analysis
- utility scripts for CNN labels CSV generation and validation
- figure generation for robustness analysis

## What this repository does not contain

The thesis also includes two additional technical components that are maintained separately:

- the pseudo-noisy data generator,
- the DnCNN-ECA denoising stage.

Those components are intentionally isolated in separate repositories because they correspond to a different stage of the experimental workflow.

## Thesis-level interpretation

At thesis level, the complete workflow is:

1. Generate realistic pseudo-noisy ESPI pairs.
2. Train or apply the DnCNN-ECA denoising stage.
3. Extract or prepare classification inputs.
4. Train and evaluate the classification models.

This repository corresponds primarily to steps **3** and **4**, plus the final robustness analysis.

## Practical guidance for readers

If a reader wants to understand only the final modal-classification results, this repository is the correct entry point.

If a reader wants to reproduce the full denoising-and-classification thesis pipeline, they should also consult the separate pseudo-noisy-generator and DnCNN-ECA repositories linked in the main `README.md`.
