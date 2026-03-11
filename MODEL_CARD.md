# Model Card - ESPI Classification Models

## Overview

This repository contains two main model families used for vibration-mode classification from ESPI-derived data:

1. **Random Forest models**
   - Pattern-only Random Forest baseline
   - Hybrid Random Forest with physics-informed frequency priors
2. **CNN baselines**
   - SimpleCNN
   - ResNet-18 adapted to single-channel ESPI inputs

The repository should be understood as the **classification and evaluation component** of the broader thesis workflow.

## Task

The task is supervised classification of vibration modes in bouzouki-family instruments from ESPI measurements or derived representations.

## Reported thesis metrics

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Hybrid Random Forest | **97.85%** | **95.15%** |
| Pattern-only Random Forest | 90.15% | 69.91% |
| CNN baseline (ResNet-18) | 93.76% | 88.11% |

Additional robustness figures reported in the thesis:

- LOBO: 91.83% +/- 8.90%
- LODO: 66.31% +/- 44.11%
- CNN MC-LOBO: 67.68% +/- 2.20%

## Inputs

### Random Forest branch

Feature CSV with class metadata and numeric descriptors. The hybrid variant uses both pattern descriptors and physics-informed priors. The public training script uses whichever valid feature columns are present in the supplied CSV.

### CNN branch

Image labels CSV with at least:

- `path`
- `label`

and preferably:

- `freq_hz`
- `dataset_id`

## Intended use

These models are intended for:

- academic research,
- reproducible thesis support material,
- comparative evaluation of physics-informed versus image-based baselines,
- ESPI-based modal classification experiments.

They are not intended to be treated as a production-ready diagnostic system without project-specific data validation and calibration.

## Limitations

The main limitations are the following:

- the raw ESPI dataset is not included in the public repository,
- the pseudo-noisy generator and the DnCNN-ECA denoiser are separate repositories,
- robustness under strong domain shift remains substantially more difficult than the standard split,
- public reproducibility depends on the user providing compatible CSV inputs and dataset structure.

## Ethical and scientific notes

This repository is designed for scientific transparency and reproducibility. Reported numbers should be interpreted in the context of the thesis protocol, dataset composition, and the separation of preprocessing, denoising, and classification code across multiple repositories.