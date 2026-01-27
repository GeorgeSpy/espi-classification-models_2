# Reproducibility Guide

This document describes how to reproduce the results from the thesis.

## Prerequisites

### A. Environment Setup

```bash
# Create virtual environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### B. Dataset Contract

The dataset is not included in this repository due to size constraints.
Contact the author for access to the ESPI measurement data.

Expected structure:
```
data/
├── features/
│   └── labels_features.csv    # Combined features and labels for RF
├── raw/                       # Raw ESPI measurements (images) for CNN
│   ├── clean/                 # Reference clean images
│   └── noisy/                 # Noisy inputs
└── splits/                    # (Optional) Pre-defined split indices
```

## 1. Reproduce Hybrid RF (Main Model)

**Goal:** Train and evaluate the Random Forest Hybrid model (Accuracy: 97.85%).

```bash
python src/rf_train_complete.py \
    --data data/features/labels_features.csv \
    --output results/rf_hybrid \
    --n-estimators 100 \
    --seed 42
```
*Output: Metrics saved to `results/rf_hybrid/metrics.json`*

## 2. Reproduce Pattern-only RF

**Goal:** Train baseline Pattern-only model (Accuracy: 90.15%).

```bash
python src/rf_train_complete.py \
    --data data/features/labels_features.csv \
    --output results/rf_pattern \
    --features pattern_only \
    --n-estimators 100 \
    --seed 42
```

## 3. Reproduce CNN Baseline (ResNet-18)

**Goal:** Train CNN reference model (Accuracy: 93.76%).

```bash
python src/train_espi_cnn_baselines.py \
    --data-root data/raw \
    --output-dir results/cnn_baseline \
    --arch resnet18 \
    --epochs 50 \
    --batch-size 32
```

## 4. Robustness Evaluation (LOBO/LODO)

**Goal:** Verify model stability across frequencies and datasets.

```bash
# Run LOBO and LODO analysis
python src/rf_lodo_lobo.py \
    --data data/features/labels_features.csv \
    --output results/robustness \
    --bin-width 5 \
    --n-estimators 600
```
*Expected Outputs:*
- **LOBO:** ~91.8% Accuracy
- **LODO:** ~66.3% Accuracy (High variance is expected)

## 5. MC-LOBO Stress Test (Exploratory)

> **Note:** This is a stress-test for domain shift, not a primary performance metric.

```bash
python src/train_espi_cnn_baselines.py \
    --mode mc-lobo \
    --data-root data/raw \
    --output-dir results/cnn_mclobo
```

## 6. Generate Figures

```bash
python src/create_rf_robustness_figures.py \
    --results results/robustness \
    --output figures/
```
