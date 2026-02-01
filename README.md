# ESPI Mode Classification (RF & CNN)

Κύριο repo για την τελική ταξινόμηση modes ESPI. Περιλαμβάνει:
1. **Random Forest (Baseline & Hybrid)**: Το τελικό νικητήριο μοντέλο.
2. **CNN (ResNet-18)**: Deep Learning baseline & MC-LOBO stress-tests.
3. **v6.1**: Future work model (frozen backbone).

---

## Locked / Verified Results (Main Thesis KPIs)

**Dataset**
- Final labeled samples: **3,443**
- Extreme imbalance ratio: **30.9:1**

**Performance (Standard split)**
| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| **Hybrid RF (Winner)** | **97.85%** | **95.15%** |
| Pattern-only RF | 90.15% | 69.91% |
| CNN (reference) | 93.76% | 88.11% |

**Robustness**
- **LOBO Accuracy:** **91.83% ± 8.9%** (Robustness Protocol)
- **LODO Accuracy:** **66.31% ± 44.11%** (Robustness Protocol)
- **CNN MC-LOBO pct20:** **67.68% ± 2.2%** (Stress-Test)
> *Note: MC-LOBO is an exploratory stress-test used to evaluate domain shift difficulty, not an official thesis KPI.*

---

## 📁 Data Availability

⚠️ **The raw ESPI measurement data is not included** in this repository due to size constraints.

### To reproduce results:

1. **Contact the author** for access to the ESPI dataset (PhaseOut folders: W01/W02/W03)
2. **Generate the dataset CSV**:
   ```bash
   python src/make_espi_labels_csv.py generate --roots [your_data_paths]
   ```
3. Run training scripts as described in REPRODUCE.md

## Related Repository

For ESPI image preprocessing and denoising (DnCNN-ECA), see:
- [ESPI-DnCNN-ECA](https://github.com/GeorgeSpy/ESPI-DnCNN-ECA) (if exists)

---

## Quickstart

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bat
:: Windows (CMD)
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See **REPRODUCE.md** for exact commands.

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{spyridakis2026espi,
  author       = {Spyridakis, Georgios},
  title        = {Automatic Modal Classification of {ESPI} Images for Musical Instrument Quality Control},
  school       = {Hellenic Mediterranean University, Department of Music Technology and Acoustics},
  year         = {2026},
  type         = {MSc Thesis},
  address      = {Rethymno, Greece},
  note         = {Hybrid Random Forest approach achieving 97.85\% accuracy on bouzouki soundboard modal classification}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
