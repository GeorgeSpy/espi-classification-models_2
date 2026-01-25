# Reproducibility Guide

This document describes how to reproduce the results from the thesis.

## Prerequisites

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository due to size constraints.
Contact the author for access to the ESPI measurement data.

Expected structure:
```
data/
├── features/
│   └── labels_features.csv    # Combined features and labels
└── raw/                       # Raw ESPI measurements (optional)
```

## 1) Train Hybrid RF (Main Model)

```bash
python src/rf_train_complete.py \
    --data data/features/labels_features.csv \
    --output results/rf_hybrid \
    --n-estimators 100 \
    --seed 42
```

Expected output: **97.85% accuracy**, **95.15% Macro-F1**

## 2) Robustness Evaluation (LOBO/LODO)

```bash
python src/rf_lodo_lobo.py \
    --data data/features/labels_features.csv \
    --output results/robustness \
    --bin-width 5 \
    --n-estimators 600
```

Expected outputs:
- **LOBO:** 91.83% ± 8.9% accuracy
- **LODO:** 66.31% ± 44.11% accuracy

## 3) Robustness Analysis & Figures

```bash
python src/rf_robustness_analysis.py \
    --data data/features/labels_features.csv \
    --output results/analysis

python src/create_rf_robustness_figures.py \
    --results results/robustness \
    --output figures/
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_estimators` | 100-600 | Higher for robustness tests |
| `class_weight` | balanced | Handles 30.9:1 imbalance |
| `random_state` | 42 | For reproducibility |
| `test_size` | 0.2 | Standard 80/20 split |
| `bin_width` | 5 Hz | LOBO frequency bins |

## Verified Results

All KPIs in README.md have been verified with these scripts.
See `docs/` for detailed analysis documentation.
