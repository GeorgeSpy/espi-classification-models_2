# Model Card — ESPI Mode Classifiers (RF & CNN)

## Overview
**Models:** Hybrid RF (Winner) & ResNet-18 (CNN)
**Task:** ESPI mode classification

## Locked Metrics (Verified)
- **Hybrid RF:** 97.85% Acc / 95.15% Macro-F1
- **CNN:** 93.76% Acc / 88.11% Macro-F1
- **LOBO RF:** 91.83% ± 8.9%
- **CNN MC-LOBO:** 67.68% ± 2.2% Acc

## Limitations
- LODO exhibits extreme variance due to board/domain shift.
- Macro-F1 can be unstable in narrow-bin LOBO regimes.

## Reproducibility
- Seeds, splits, configs tracked in REPRODUCE.md
