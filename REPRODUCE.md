# Reproducibility Guide

This document describes how to reproduce the results from the thesis.

## Prerequisites

### Dataset Preparation

The ESPI measurement data is **not included** in this repository. Before running any scripts:

1. Obtain the raw ESPI data (W01/W02/W03 PhaseOut folders)
2. Generate labels CSV:
   ```bash
   python src/make_espi_labels_csv.py generate \
     --roots path/to/W01_PhaseOut path/to/W02_PhaseOut path/to/W03_PhaseOut \
     --out_csv data/labels_images.csv
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

### CNN MC-LOBO Stress Testing

For the CNN stress-test mentioned in the thesis (67.68% ± 2.2%):

```bash
python src/train_espi_cnn_baselines_mclobo.py --labels_csv data/labels_images.csv --run_dir results/cnn_mclobo
```

## 6. Generate Figures

```bash
python src/create_rf_robustness_figures.py \
    --results results/robustness \
    --output figures/
```
