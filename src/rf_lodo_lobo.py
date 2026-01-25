#!/usr/bin/env python3
"""
LODO & LOBO Cross-Validation Analysis

Leave-One-Dataset-Out (LODO) and Leave-One-Bin-Out (LOBO) analysis
for evaluating model robustness to domain and frequency shifts.

Usage:
    python rf_lodo_lobo.py --data path/to/features.csv --output results/
    python rf_lodo_lobo.py --data data/features.csv --output results/ --bin-width 5
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LODO/LOBO cross-validation for ESPI classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Path to input CSV with features and labels"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=600,
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Frequency bin width in Hz for LOBO"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


# Columns that should never be used as features
ALWAYS_EXCLUDE = {
    "class_name", "class_id", "label", "target", "dataset", "set", "split",
    "group_id", "file", "filepath", "path", "relpath", "name", "id"
}

# Patterns for columns that could cause data leakage
LEAKAGE_PATTERNS = [r"^freq$", r"^freq_hz$", r"^level_db$", r"^dist_"]


def is_leakage_column(col_name: str) -> bool:
    """Check if column could cause data leakage."""
    return any(re.match(p, col_name) for p in LEAKAGE_PATTERNS)


def load_and_prepare_data(data_path: Path) -> tuple:
    """
    Load data and prepare features.
    
    Returns:
        df: Full dataframe
        pattern_cols: List of valid pattern feature columns
        y: Labels array
        freq: Frequency values
        datasets: Dataset identifiers
    """
    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))
    
    # Select numeric columns excluding metadata and leakage
    num_cols = [
        c for c in df.columns 
        if c not in ALWAYS_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    pattern_cols = [c for c in num_cols if not is_leakage_column(c)]
    
    # Labels
    y = df["class_name"].astype(str).values
    
    # Frequency column
    if "freq_hz" in df:
        freq = df["freq_hz"].astype(float)
    elif "freq" in df:
        freq = df["freq"].astype(float)
    else:
        raise ValueError("Dataset must contain 'freq_hz' or 'freq' column")
    
    # Dataset identifiers
    if "dataset" in df.columns:
        datasets = df["dataset"].astype(str)
    else:
        datasets = pd.Series(["UNK"] * len(df))
    
    print(f"Loaded {len(df)} samples")
    print(f"Pattern features: {len(pattern_cols)}")
    print(f"Classes: {sorted(np.unique(y))}")
    
    return df, pattern_cols, y, freq, datasets


def lodo_analysis(df, pattern_cols, y, datasets, args) -> list:
    """
    Leave-One-Dataset-Out cross-validation.
    
    Tests model generalization across different measurement sessions/boards.
    """
    print("\n" + "=" * 50)
    print("Leave-One-Dataset-Out (LODO) Analysis")
    print("=" * 50)
    
    X = df[pattern_cols].values
    unique_datasets = sorted(datasets.unique())
    
    print(f"Datasets: {unique_datasets}")
    print(f"Dataset sizes: {datasets.value_counts().to_dict()}")
    
    results = []
    
    for test_dataset in unique_datasets:
        train_mask = datasets != test_dataset
        test_mask = datasets == test_dataset
        
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"  Skipping {test_dataset} - insufficient samples")
            continue
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=args.seed
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            'test_dataset': test_dataset,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'accuracy': float(acc),
            'macro_f1': float(macro_f1)
        })
        
        print(f"  {test_dataset}: Acc={acc:.4f}, F1={macro_f1:.4f} "
              f"(train={train_mask.sum()}, test={test_mask.sum()})")
    
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        std_acc = np.std([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['macro_f1'] for r in results])
        print(f"\nLODO Summary: {avg_acc:.4f} ± {std_acc:.4f} accuracy")
    
    return results


def lobo_analysis(df, pattern_cols, y, freq, args) -> list:
    """
    Leave-One-Bin-Out cross-validation.
    
    Tests model generalization across different frequency ranges.
    """
    print("\n" + "=" * 50)
    print(f"Leave-One-Bin-Out (LOBO) Analysis ({args.bin_width}Hz bins)")
    print("=" * 50)
    
    X = df[pattern_cols].values
    
    # Create frequency bins
    freq_bins = (freq / args.bin_width).apply(np.floor).astype(int)
    unique_bins = sorted(freq_bins.unique())
    
    print(f"Frequency bins: {len(unique_bins)}")
    
    results = []
    
    for test_bin in unique_bins:
        train_mask = freq_bins != test_bin
        test_mask = freq_bins == test_bin
        
        # Skip bins with too few samples
        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=args.seed
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        freq_range = f"{int(test_bin * args.bin_width)}-{int((test_bin + 1) * args.bin_width)}"
        
        results.append({
            'test_bin': int(test_bin),
            'freq_range': freq_range,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'accuracy': float(acc),
            'macro_f1': float(macro_f1)
        })
        
        print(f"  {freq_range}Hz: Acc={acc:.4f}, F1={macro_f1:.4f} "
              f"(n={test_mask.sum()})")
    
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        std_acc = np.std([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['macro_f1'] for r in results])
        print(f"\nLOBO Summary: {avg_acc:.4f} ± {std_acc:.4f} accuracy")
    
    return results


def save_results(lodo_results: list, lobo_results: list, output_dir: Path) -> None:
    """Save analysis results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if lodo_results:
        with open(output_dir / "lodo_results.json", 'w') as f:
            json.dump(lodo_results, f, indent=2)
        print(f"\nLODO results saved to: {output_dir / 'lodo_results.json'}")
    
    if lobo_results:
        with open(output_dir / "lobo_results.json", 'w') as f:
            json.dump(lobo_results, f, indent=2)
        print(f"LOBO results saved to: {output_dir / 'lobo_results.json'}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 50)
    print("LODO & LOBO Cross-Validation Analysis")
    print("=" * 50)
    
    # Load data
    df, pattern_cols, y, freq, datasets = load_and_prepare_data(args.data)
    
    # Run analyses
    lodo_results = lodo_analysis(df, pattern_cols, y, datasets, args)
    lobo_results = lobo_analysis(df, pattern_cols, y, freq, args)
    
    # Save results
    save_results(lodo_results, lobo_results, args.output)
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("Summary Comparison")
    print("=" * 50)
    print("Standard CV (reference): ~90.15% accuracy")
    if lodo_results:
        lodo_acc = np.mean([r['accuracy'] for r in lodo_results])
        print(f"LODO: {lodo_acc:.4f} (tests domain shift)")
    if lobo_results:
        lobo_acc = np.mean([r['accuracy'] for r in lobo_results])
        print(f"LOBO: {lobo_acc:.4f} (tests frequency generalization)")


if __name__ == "__main__":
    main()
