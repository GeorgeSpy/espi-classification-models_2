#!/usr/bin/env python3
"""
make_espi_labels_csv.py — Generator & Validator for ESPI CNN baselines
======================================================================

What this script does
---------------------
1) **generate**: Scans your PhaseOut folders and creates a CSV with the exact
   columns that `train_espi_cnn_baselines.py` expects.
2) **validate**: Runs consistency checks on an existing CSV (paths, labels,
   optional npy shape checks), before training.

CSV Schema (header)
-------------------
path,freq_hz,dataset_id,label

- **path**: full or relative path to 16‑bit PNG or .npy (grayscale; unwrapped phase or any image you want the CNN to learn).
- **freq_hz** (optional but recommended): float; used for LOBO splits.
- **dataset_id** (optional but recommended): string (e.g., W01/W02/W03); used for LODO splits.
- **label**: integer in {0..5} according to your label_map:
    0: mode_(1,1)H
    1: mode_(1,1)T
    2: mode_(1,2)
    3: mode_(2,1)
    4: mode_higher
    5: other_unknown

Frequency → label mapping (wood, fixed bins)
--------------------------------------------
- (1,1)H:  [155, 175]
- (1,1)T:  [320, 345]
- (1,2):   [500, 525]
- (2,1):   [540, 570]
- higher:  [680, 1500]
- other:   anything else

Windows tip
-----------
Use **forward slashes** in CSV (e.g., `C:/ESPI_TEMP/...`) for robust parsing across tools.

Usage examples
--------------
# Generate CSV by scanning PhaseOut roots (NPY, take 3 frames per freq dir)
python make_espi_labels_csv.py generate \
  --roots C:/ESPI_TEMP/GPU_FULL2/W01_PhaseOut_b18_cs16_ff100 \
          C:/ESPI_TEMP/GPU_FULL2/W02_PhaseOut_b18_cs16_ff100 \
          C:/ESPI_TEMP/GPU_FULL2/W03_PhaseOut_b18_cs16_ff100 \
  --out_csv C:/ESPI_TEMP/THESIS_PACKAGE/FINAL_labels_images.csv \
  --subdir phase_unwrapped_npy --ext npy --frames-per-dir 3

# Same but for PNGs
python make_espi_labels_csv.py generate \
  --roots C:/ESPI_TEMP/GPU_FULL2/W02_PhaseOut_b18_cs16_ff100 \
  --out_csv C:/ESPI_TEMP/THESIS_PACKAGE/FINAL_labels_images_png.csv \
  --subdir phase_unwrapped_png --ext png --frames-per-dir -1

# Validate before training
python make_espi_labels_csv.py validate \
  --csv C:/ESPI_TEMP/THESIS_PACKAGE/FINAL_labels_images.csv

"""
import os
import re
import csv
import json
import argparse
from glob import glob
from typing import List, Tuple, Optional

try:
    import numpy as np
except ImportError:
    np = None  # Only needed for .npy validation


FREQ_RE = re.compile(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*Hz", re.IGNORECASE)
DATASET_RE = re.compile(r"\bW0[1-9]\b", re.IGNORECASE)


def label_from_freq(freq: float) -> int:
    """Map frequency (Hz) to label id using fixed wood bins."""
    if 155 <= freq <= 175:
        return 0  # (1,1)H
    if 320 <= freq <= 345:
        return 1  # (1,1)T
    if 500 <= freq <= 525:
        return 2  # (1,2)
    if 540 <= freq <= 570:
        return 3  # (2,1)
    if 680 <= freq <= 1500:
        return 4  # higher
    return 5      # other_unknown


def find_dataset_id(path: str) -> str:
    """Search upwards in path segments for tokens like W01/W02/W03; fallback to '?'"""
    parts = os.path.normpath(path).split(os.sep)
    for seg in reversed(parts):
        m = DATASET_RE.search(seg)
        if m:
            return m.group(0).upper()
    return "?"


def find_freq_from_path(path: str) -> Optional[float]:
    """Find first '...Hz' token in path segments (handles '0160Hz', '520.0Hz', etc.)."""
    for seg in reversed(os.path.normpath(path).split(os.sep)):
        m = FREQ_RE.search(seg)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def iter_phase_dirs(root: str) -> List[str]:
    # typical pattern: .../####Hz_*db/
    # be permissive: any dir containing 'Hz'
    cand = []
    for d in glob(os.path.join(root, "**"), recursive=True):
        if os.path.isdir(d) and ("Hz" in os.path.basename(d) or "hz" in os.path.basename(d)):
            cand.append(d)
    return sorted(cand)


def list_frames(freq_dir: str, subdir: str, ext: str) -> List[str]:
    p = os.path.join(freq_dir, subdir)
    if not os.path.isdir(p):
        return []
    pattern = os.path.join(p, f"*.{ext.lower()}")
    return sorted(glob(pattern))


def to_forward_slashes(p: str) -> str:
    return p.replace("\\", "/")


def generate_csv(roots: List[str], out_csv: str, subdir: str, ext: str, frames_per_dir: int,
                 require_freq: bool = True, require_dataset: bool = False,
                 limit_per_root: Optional[int] = None) -> int:
    rows = []
    for root in roots:
        freq_dirs = iter_phase_dirs(root)
        if limit_per_root:
            freq_dirs = freq_dirs[:limit_per_root]
        for fdir in freq_dirs:
            freq = find_freq_from_path(fdir)
            if require_freq and freq is None:
                continue
            files = list_frames(fdir, subdir=subdir, ext=ext)
            if not files:
                continue
            if frames_per_dir is None or frames_per_dir < 0:
                take = files
            else:
                take = files[:frames_per_dir]
            dsid = find_dataset_id(fdir) if require_dataset or True else "?"
            label = label_from_freq(freq if freq is not None else -1)
            for fp in take:
                rows.append([
                    to_forward_slashes(os.path.abspath(fp)),
                    f"{freq if freq is not None else ''}",
                    dsid,
                    str(label),
                ])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "freq_hz", "dataset_id", "label"])
        w.writerows(rows)

    # class counts summary
    counts = {i: 0 for i in range(6)}
    for r in rows:
        try:
            counts[int(r[3])] += 1
        except Exception:
            pass
    summary = {
        "total_rows": len(rows),
        "class_counts": counts,
        "roots": roots,
        "subdir": subdir,
        "ext": ext,
        "frames_per_dir": frames_per_dir,
    }
    with open(os.path.splitext(out_csv)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[generate] wrote: {out_csv}  (rows={len(rows)})")
    print("class counts:", counts)
    return len(rows)


def validate_csv(csv_path: str, check_npy: bool = True) -> None:
    import pandas as pd
    import os
    df = pd.read_csv(csv_path)

    # required columns
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    # label range
    bad_labels = df[~df["label"].isin([0, 1, 2, 3, 4, 5])]
    if not bad_labels.empty:
        print(bad_labels.head())
        raise SystemExit(f"Invalid labels in {len(bad_labels)} rows (labels must be 0..5).")

    # path existence
    exists_mask = df["path"].apply(os.path.exists)
    missing_paths = df[~exists_mask]
    print(f"[validate] total rows={len(df)}, missing_files={len(missing_paths)}")
    if not missing_paths.empty:
        print(missing_paths.head())

    # optional: .npy shape/content check
    if check_npy:
        if np is None:
            print("[validate] numpy not available; skipping .npy checks")
        else:
            def ok_npy(p: str) -> bool:
                if not str(p).lower().endswith(".npy"):
                    return True
                try:
                    arr = np.load(p, mmap_mode="r")
                    return (arr.size > 0) and (arr.ndim in (2, 3))
                except Exception:
                    return False
            bad_npy = df[~df["path"].apply(ok_npy)]
            print(f"[validate] problematic .npy rows={len(bad_npy)}")
            if not bad_npy.empty:
                print(bad_npy.head())


def main():
    ap = argparse.ArgumentParser(description="Generate/validate CSV for ESPI CNN baselines")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="scan PhaseOut dirs and write CSV")
    g.add_argument("--roots", nargs="+", type=str, required=True, help="PhaseOut root folders (W01_..., W02_..., W03_...)")
    g.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    g.add_argument("--subdir", type=str, default="phase_unwrapped_npy", help="Sub-folder inside each freq dir (e.g., phase_unwrapped_npy / phase_unwrapped_png)")
    g.add_argument("--ext", type=str, choices=["npy", "png"], default="npy", help="File extension to pick")
    g.add_argument("--frames-per-dir", type=int, default=3, help="How many frames to take per frequency dir (-1 for all)")
    g.add_argument("--require-freq", action="store_true", help="Skip dirs where freq Hz cannot be parsed")
    g.add_argument("--require-dataset", action="store_true", help="Skip dirs where dataset_id cannot be inferred")
    g.add_argument("--limit-per-root", type=int, default=None, help="For quick dry-runs; limit number of freq dirs per root")

    v = sub.add_parser("validate", help="validate an existing CSV")
    v.add_argument("--csv", type=str, required=True, help="CSV path to validate")
    v.add_argument("--no-npy-check", action="store_true", help="Do not inspect .npy files for shape/emptiness")

    args = ap.parse_args()

    if args.cmd == "generate":
        generate_csv(
            roots=args.roots,
            out_csv=args.out_csv,
            subdir=args.subdir,
            ext=args.ext,
            frames_per_dir=args.frames_per_dir,
            require_freq=args.require_freq,
            require_dataset=args.require_dataset,
            limit_per_root=args.limit_per_root,
        )
    elif args.cmd == "validate":
        validate_csv(csv_path=args.csv, check_npy=(not args.no_npy_check))


if __name__ == "__main__":
    main()
