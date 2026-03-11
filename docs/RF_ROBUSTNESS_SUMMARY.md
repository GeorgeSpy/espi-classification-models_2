# Supplementary Technical Summary of Random Forest Generalization, Stability, and Error Analysis

This document provides supplementary technical material supporting the Random Forest results reported in the thesis. While Section 4.5 of the thesis focuses on the comparative evaluation of Random Forest and CNN classifiers on the common ESPI dataset, the present note summarizes the additional evidence used to support the interpretation of the Pattern-only Random Forest model, including confidence intervals, cross-validation stability, error patterns, and out-of-domain validation behavior.

## Robustness Analysis for PatternOnly Random Forest

### Main Results

**PatternOnly Random Forest performance:**

- **Accuracy:** 90.15% (95% CI: 89.14% - 91.05%)
- **Macro-F1:** 69.91% (95% CI: 66.71% - 72.66%)
- **Weighted-F1:** 88.53% (95% CI: 87.12% - 89.94%)

**Statistical significance vs baseline:**

- **Accuracy improvement:** +34.55% (`p < 0.001`)
- **Macro-F1 improvement:** +19.31% (`p < 0.001`)
- **Effect size (Cohen's d):** 2.34

---

## Statistical Significance Testing

### Bootstrap Confidence Intervals

- **Accuracy:** 90.15% +/- 0.96%
- **Macro-F1:** 69.91% +/- 2.98%
- **Cross-validation stability:** `sigma^2 = 0.0008`

### Key Findings

- The improvement over baseline is highly significant.
- The estimated effect size is large.
- Three-fold grouped cross-validation shows strong stability.

---

## Feature Importance Analysis

### Top-5 Most Important Features

1. **`freq_hz.1`:** 55.98% importance, dominant frequency-driven predictor
2. **`diag_ratio`:** 4.66%, diagonal symmetry measure
3. **`valid_px`:** 3.55%, valid-pixel ratio
4. **`lapz`:** 3.20%, normalized Laplacian
5. **`grad_cv`:** 3.14%, gradient coefficient of variation

### Feature Categories

- **Frequency-based:** 55.98% (1 feature)
- **Symmetry features:** 7.58% (2 features)
- **Topological features:** 6.32% (2 features)
- **Nodal features:** 5.76% (2 features)
- **Gradient features:** 5.95% (2 features)

### Feature Stability

- **Cross-seed correlation:** 0.075
- **Consistently stable features:** 10/16 (62.5%)
- **Most stable features:** `freq_hz.1`, `diag_ratio`, `valid_px`, `lapz`, `grad_cv`

---

## Cross-Validation Stability

### 3-Fold StratifiedGroupKFold Results

| Fold | Accuracy | Macro-F1 | Weighted-F1 |
|------|----------|----------|-------------|
| 1 | 90.8% | 70.2% | 89.1% |
| 2 | 89.2% | 68.9% | 87.8% |
| 3 | 91.1% | 70.6% | 89.9% |
| **Mean +/- Std** | **90.4% +/- 0.9%** | **69.9% +/- 0.9%** | **88.9% +/- 1.1%** |

### Stability Metrics

- **Coefficient of variation:** 1.0%
- **Range:** 1.9%
- **Grouped CV:** reduces leakage between datasets

---

## Error Analysis and Confusion Patterns

### Key Error Patterns

1. **Minority-class confusion**
   - `mode_(1,1)H`: 69% -> `other_unknown`
   - `mode_(1,1)T`: 64% -> `other_unknown`
   - `mode_(2,1)`: 31% -> `mode_(1,2)`
2. **Majority-class stability**
   - `mode_higher`: 100% accuracy
   - `other_unknown`: 98.6% accuracy

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Error Rate |
|-------|-----------|--------|----------|---------|------------|
| mode_(1,1)H | 95.3% | 30.7% | 46.4% | 199 | 69.3% |
| mode_(1,1)T | 87.3% | 36.3% | 51.2% | 171 | 63.7% |
| mode_(1,2) | 69.6% | 67.0% | 68.3% | 106 | 33.0% |
| mode_(2,1) | 100% | 44.8% | 61.9% | 58 | 55.2% |
| mode_higher | 100% | 100% | 100% | 1,115 | 0% |
| other_unknown | 85.7% | 98.6% | 91.7% | 1,794 | 1.4% |

---

## Generalization Analysis

### Leave-One-Dataset-Out (LODO)

| Test Dataset | Accuracy | Macro-F1 | Interpretation |
|--------------|----------|----------|----------------|
| W01 | 83.7% | 56.8% | Good generalization |
| W02 | 15.3% | 8.2% | Poor generalization |
| W03 | 100% | 100% | Perfect generalization on a small test set |

### Leave-One-Band-Out (LOBO)

- High performance is observed across most low- and high-frequency regions.
- Weak regions remain concentrated around specific minority-class frequency bands.
- The most difficult bins are `150-155 Hz`, `175-180 Hz`, `315-320 Hz`, and `320-325 Hz`.

---

## Computational Efficiency

### Performance Metrics

- **Training time:** about 45 seconds for 3,443 samples and 16 features
- **Inference time:** about 0.8 ms per sample
- **Model size:** about 50 MB
- **Batch throughput:** about 1,000 samples per second

### Comparison with Deep Learning

- **RF training:** about 45 s vs about 2 h for CNN
- **RF inference:** about 0.8 ms vs about 15 ms for CNN
- **RF memory:** about 50 MB vs about 200 MB for CNN
- **RF advantage:** about 18.75x faster inference

---

## Summary and Conclusions

### Key Strengths

1. **Statistical reliability:** 90.15% accuracy with narrow confidence intervals.
2. **Interpretability:** meaningful feature-importance structure beyond the dominant frequency term.
3. **Stability:** low grouped cross-validation variance.
4. **Efficiency:** substantially lower training and inference cost than CNN baselines.

### Limitations

1. **Class imbalance:** minority classes remain difficult.
2. **Frequency dependence:** the model relies heavily on frequency information.
3. **Dataset transfer:** performance on W02 remains weak.

### Scientific Contribution

The results support PatternOnly Random Forest as a computationally efficient, interpretable, and statistically robust baseline for ESPI vibration-mode classification within the thesis workflow.