# ESPI Classification Models for Bouzouki Modal Analysis

This repository contains the public classification and evaluation code accompanying the master's thesis on automated vibration-mode classification in bouzouki-family instruments from Electronic Speckle Pattern Interferometry (ESPI) data.

The repository focuses on the **classification stage** of the pipeline. It includes the final Random Forest models, CNN baselines, dataset utilities for the image-classification branch, and robustness analysis scripts. The pseudo-noisy data generator and the DnCNN-based denoising stage are intentionally maintained in separate repositories.

## Repository scope within the thesis

The full thesis spans three code components:

1. **Pseudo-noisy data generation** for realistic ESPI supervision.
2. **DnCNN-ECA denoising** for ESPI image restoration.
3. **Classification and evaluation**, which is the scope of this repository.

In practical terms, this repository corresponds to the final modal-classification stage built on top of preprocessed or denoised ESPI data. It covers:

- Random Forest baselines and the final Hybrid Random Forest model
- CNN image baselines (`SimpleCNN` and `ResNet-18`)
- LOBO and LODO robustness analysis
- CSV generation and validation for the CNN branch
- Figure generation for robustness analysis

It does **not** contain the training code for the pseudo-noisy generator or the DnCNN-ECA denoiser.

## Related repositories

The thesis codebase is split across the following repositories:

- **ESPI classification and evaluation (this repository)**  
  `https://github.com/GeorgeSpy/espi-classification-models_2`
- **DnCNN-ECA denoising**  
  `https://github.com/GeorgeSpy/ESPI-DnCNN-ECA`
- **Pseudo-noisy data generation**  
  `https://github.com/GeorgeSpy/ESPI-pseydonoisy-generator`

## Locked thesis results

The following headline metrics correspond to the thesis results reported for the public classification repository:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Hybrid Random Forest | **97.85%** | **95.15%** |
| Pattern-only Random Forest | 90.15% | 69.91% |
| CNN baseline (ResNet-18) | 93.76% | 88.11% |

Robustness results reported in the thesis include:

- **LOBO accuracy:** 91.83% ± 8.90%
- **LODO accuracy:** 66.31% ± 44.11%
- **CNN MC-LOBO stress test:** 67.68% ± 2.20%

## Data availability

The raw ESPI measurement data are **not included** in this repository because of size and project-distribution constraints.

To reproduce the experiments, you will need:

- the ESPI image data or derived feature tables,
- a labels CSV for the CNN branch,
- a feature CSV for the Random Forest branch.

The repository provides scripts for creating and validating the image labels CSV used by the CNN baselines.

## Repository layout

```text
.
├── README.md
├── REPRODUCE.md
├── MODEL_CARD.md
├── CITATION.cff
├── requirements.txt
├── docs/
└── src/
    ├── make_espi_labels_csv.py
    ├── rf_train_complete.py
    ├── rf_lodo_lobo.py
    ├── train_espi_cnn_baselines.py
    ├── train_espi_cnn_baselines_mclobo.py
    └── create_rf_robustness_figures.py
```

## Quick start

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (CMD)

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See `REPRODUCE.md` for command-line examples that match the public scripts.

## Notes on reproducibility

The public scripts are intentionally lightweight and expect the user to provide already prepared inputs. In particular:

- `rf_train_complete.py` trains on **all numeric feature columns** present in the input CSV, excluding metadata columns.
- `rf_lodo_lobo.py` applies leakage-safe exclusion rules for frequency-like columns during robustness analysis.
- `train_espi_cnn_baselines.py` expects an image labels CSV with at least `path` and `label`, and preferably `freq_hz` and `dataset_id` as well.

Because the pseudo-noisy generator and denoising stage live in separate repositories, this repository should be understood as the **classification / evaluation repository** of the thesis rather than the full end-to-end pipeline.

## Citation

If you use this repository, please cite the software metadata in `CITATION.cff`. For thesis citation, you may also use the following BibTeX entry:

```bibtex
@mastersthesis{spyridakis2026espi,
  author       = {Spyridakis, Georgios},
  title        = {Automated Classification of Vibration Modes in Bouzouki Family Instruments via Machine Learning Techniques and Advanced Interferometric Image Processing},
  school       = {Hellenic Mediterranean University},
  year         = {2026},
  type         = {Master's Thesis},
  address      = {Greece}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
