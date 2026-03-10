# Reproducibility Guide

This document describes how to run the **public scripts** contained in this repository. It is intentionally aligned with the current codebase, not with older internal script names from the thesis development phase.

## 1. Environment setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

On Windows, use the activation command shown in `README.md`.

## 2. Required inputs

This repository uses two different input formats.

### 2.1 CNN branch: image labels CSV

The CNN scripts expect a CSV with the following schema:

```text
path,freq_hz,dataset_id,label
```

Required columns:

- `path`
- `label`

Recommended columns:

- `freq_hz` for LOBO-style evaluation
- `dataset_id` for LODO-style evaluation

### 2.2 Random Forest branch: feature CSV

The Random Forest scripts expect a feature table with:

- class metadata columns such as `class_id` and `class_name`,
- numeric feature columns for the model input,
- optional metadata columns such as `freq_hz`, `dataset`, `path`, and similar identifiers.

`rf_train_complete.py` uses **all non-excluded numeric feature columns** present in the CSV. Therefore, the difference between a hybrid run and a pattern-only run is determined by the contents of the input feature table.

## 3. Generate the CNN labels CSV

Use the public helper script to scan PhaseOut folders and build the image labels CSV:

```bash
python src/make_espi_labels_csv.py generate \
  --roots path/to/W01_PhaseOut path/to/W02_PhaseOut path/to/W03_PhaseOut \
  --out_csv data/labels_images.csv \
  --subdir phase_unwrapped_npy \
  --ext npy \
  --frames-per-dir 3
```

Validate the generated file before training:

```bash
python src/make_espi_labels_csv.py validate \
  --csv data/labels_images.csv
```

## 4. Train the Hybrid Random Forest model

Use a feature CSV that already contains both pattern features and physics-informed frequency priors:

```bash
python src/rf_train_complete.py \
  --data data/features/labels_features_hybrid.csv \
  --output results/rf_hybrid \
  --n-estimators 100 \
  --seed 42
```

The main JSON output is written to:

```text
results/rf_hybrid/rf_results.json
```

## 5. Train the Pattern-only Random Forest baseline

Use a feature CSV that contains only the pattern features required for the baseline:

```bash
python src/rf_train_complete.py \
  --data data/features/labels_features_pattern_only.csv \
  --output results/rf_pattern_only \
  --n-estimators 100 \
  --seed 42
```

There is **no `--features` switch** in the public script. Pattern-only vs hybrid behavior is controlled by the feature columns present in the CSV you pass.

## 6. Run LOBO / LODO robustness analysis

```bash
python src/rf_lodo_lobo.py \
  --data data/features/labels_features_pattern_only.csv \
  --output results/rf_robustness \
  --bin-width 5 \
  --n-estimators 600 \
  --seed 42
```

Important note: the robustness script automatically excludes columns matching frequency-like leakage patterns such as `freq`, `freq_hz`, `level_db`, and `dist_*`.

Expected output files:

```text
results/rf_robustness/lobo_results.json
results/rf_robustness/lodo_results.json
```

## 7. Train the CNN baseline (ResNet-18)

```bash
python src/train_espi_cnn_baselines.py \
  --labels_csv data/labels_images.csv \
  --run_dir results/cnn_resnet18 \
  --model resnet18 \
  --img_size 256 \
  --epochs 50 \
  --batch_size 64 \
  --lr 3e-4 \
  --freeze_until layer2 \
  --augment strong
```

Main outputs:

```text
results/cnn_resnet18/metrics.json
results/cnn_resnet18/best.pt
results/cnn_resnet18/cm_test.png
```

## 8. Train the SimpleCNN baseline

```bash
python src/train_espi_cnn_baselines.py \
  --labels_csv data/labels_images.csv \
  --run_dir results/cnn_simple \
  --model simple \
  --img_size 256 \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-3 \
  --augment strong
```

## 9. Run the CNN MC-LOBO stress test

```bash
python src/train_espi_cnn_baselines_mclobo.py \
  --labels_csv data/labels_images.csv \
  --run_dir results/cnn_mclobo \
  --model resnet18 \
  --lobo_per_class_pct 0.20 \
  --epochs 50 \
  --batch_size 64 \
  --lr 3e-4
```

This script is intended for exploratory stress testing under stronger domain-shift conditions.

## 10. Generate robustness figures

```bash
python src/create_rf_robustness_figures.py \
  --results results/rf_robustness \
  --output figures
```

## 11. Scope clarification

This repository does **not** reproduce the full thesis pipeline by itself. The pseudo-noisy generator and the DnCNN-ECA denoiser are maintained as separate repositories. The public scripts here should therefore be interpreted as the **classification and evaluation layer** of the overall thesis workflow.
