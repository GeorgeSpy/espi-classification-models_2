#!/usr/bin/env python3
"""
RF Robustness Analysis

Bootstrap confidence intervals, confusion matrices, and per-class metrics
for evaluating model reliability.

Usage:
    python rf_robustness_analysis.py --data path/to/features.csv --output results/
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robustness analysis for ESPI RF classifiers",
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
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


# Columns to exclude from features
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
    """Load data and prepare feature sets."""
    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))
    
    # Feature columns
    num_cols = [
        c for c in df.columns 
        if c not in ALWAYS_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    pattern_cols = [c for c in num_cols if not is_leakage_column(c)]
    hybrid_cols = list(num_cols)
    
    # Labels and classes
    y = df["class_name"].astype(str).values
    classes = sorted(np.unique(y))
    
    # Groups for CV
    if "freq_hz" in df:
        freq = df["freq_hz"].astype(float)
    elif "freq" in df:
        freq = df["freq"].astype(float)
    else:
        raise ValueError("Dataset must contain 'freq_hz' or 'freq' column")
    
    dset = df["dataset"] if "dataset" in df.columns else pd.Series(["UNK"] * len(df))
    groups = dset.astype(str) + "_" + (np.floor(freq / 5).astype(int)).astype(str)
    
    print(f"Loaded {len(df)} samples, {len(classes)} classes")
    print(f"Pattern features: {len(pattern_cols)}, Hybrid features: {len(hybrid_cols)}")
    
    return df, pattern_cols, hybrid_cols, y, classes, groups


def bootstrap_ci(y_true: list, y_pred: list, n_bootstrap: int = 1000, 
                 confidence: float = 0.95, seed: int = 42) -> dict:
    """Compute bootstrap confidence intervals for accuracy and macro-F1."""
    np.random.seed(seed)
    n_samples = len(y_true)
    accuracies = []
    macro_f1s = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
        macro_f1s.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
    
    alpha = 1 - confidence
    acc_ci = np.percentile(accuracies, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    f1_ci = np.percentile(macro_f1s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    
    return {
        'accuracy': {'mean': float(np.mean(accuracies)), 'ci': acc_ci.tolist()},
        'macro_f1': {'mean': float(np.mean(macro_f1s)), 'ci': f1_ci.tolist()}
    }


def plot_confusion_matrix(y_true: list, y_pred: list, classes: list, 
                          title: str, save_path: Path) -> None:
    """Plot and save normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{title} - Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def analyze_setup(df: pd.DataFrame, feature_cols: list, y: np.ndarray, 
                  classes: list, groups: pd.Series, setup_name: str,
                  output_dir: Path, args) -> dict:
    """Complete analysis for one feature setup."""
    print(f"\n{'=' * 50}")
    print(f"Analyzing: {setup_name}")
    print(f"{'=' * 50}")
    
    X = df[feature_cols].values
    
    # Cross-validation
    try:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
        splits = list(cv.split(X, y, groups=groups))
    except Exception:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
        splits = list(cv.split(X, y))
    
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=args.seed
        )
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)
        print(f"  Fold {fold}: {len(test_idx)} samples")
    
    # Compute metrics
    acc = accuracy_score(all_y_true, all_y_pred)
    macro_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    
    # Bootstrap CI
    print("  Computing bootstrap confidence intervals...")
    bootstrap_results = bootstrap_ci(all_y_true, all_y_pred, 
                                     n_bootstrap=args.n_bootstrap, seed=args.seed)
    
    # Per-class metrics
    cls_report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm_path = output_dir / f"confusion_matrix_{setup_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(all_y_true, all_y_pred, classes, setup_name, cm_path)
    
    # Results
    results = {
        'setup': setup_name,
        'n_features': len(feature_cols),
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'bootstrap_ci': bootstrap_results,
        'per_class_metrics': cls_report,
        'classes': classes
    }
    
    # Save
    results_path = output_dir / f"robustness_{setup_name.lower().replace(' ', '_')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    acc_ci = bootstrap_results['accuracy']['ci']
    f1_ci = bootstrap_results['macro_f1']['ci']
    print(f"  Accuracy: {acc:.4f} (95% CI: {acc_ci[0]:.4f}-{acc_ci[1]:.4f})")
    print(f"  Macro-F1: {macro_f1:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 50)
    print("RF Robustness Analysis")
    print("=" * 50)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, pattern_cols, hybrid_cols, y, classes, groups = load_and_prepare_data(args.data)
    
    # Analyze both setups
    pattern_results = analyze_setup(
        df, pattern_cols, y, classes, groups,
        "Pattern_Only", args.output, args
    )
    hybrid_results = analyze_setup(
        df, hybrid_cols, y, classes, groups,
        "Hybrid", args.output, args
    )
    
    # Comparison summary
    comparison = {
        'pattern_only': {
            'accuracy': pattern_results['accuracy'],
            'macro_f1': pattern_results['macro_f1'],
            'accuracy_ci': pattern_results['bootstrap_ci']['accuracy']['ci'],
            'macro_f1_ci': pattern_results['bootstrap_ci']['macro_f1']['ci']
        },
        'hybrid': {
            'accuracy': hybrid_results['accuracy'],
            'macro_f1': hybrid_results['macro_f1'],
            'accuracy_ci': hybrid_results['bootstrap_ci']['accuracy']['ci'],
            'macro_f1_ci': hybrid_results['bootstrap_ci']['macro_f1']['ci']
        }
    }
    
    with open(args.output / "robustness_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    p_ci = np.diff(pattern_results['bootstrap_ci']['accuracy']['ci'])[0] / 2
    h_ci = np.diff(hybrid_results['bootstrap_ci']['accuracy']['ci'])[0] / 2
    print(f"Pattern-Only: {pattern_results['accuracy']:.4f} ± {p_ci:.4f}")
    print(f"Hybrid:       {hybrid_results['accuracy']:.4f} ± {h_ci:.4f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
