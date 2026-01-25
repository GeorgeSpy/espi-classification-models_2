#!/usr/bin/env python3
"""
RF Training Script - Complete Pipeline

Train a Random Forest classifier on ESPI features for modal classification.

Usage:
    python rf_train_complete.py --data path/to/features.csv --output results/
    python rf_train_complete.py --data data/labels.csv --output results/ --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier on ESPI features",
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
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds"
    )
    return parser.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate input data."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and labels from dataframe.
    
    Returns:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        feature_names: List of feature column names
    """
    # Columns to exclude from features
    exclude_cols = {
        'id', 'dataset', 'source_file', 'class_id', 'class_name', 
        'material', 'freq_hz.1', 'label', 'target', 'filepath', 'path'
    }
    
    feature_names = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(feature_names)} features")
    
    X = df[feature_names].values
    y = df['class_id'].values
    
    return X, y, feature_names


def train_and_evaluate(X, y, df, args) -> dict:
    """Train RF model and compute metrics."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.seed, 
        stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=args.seed,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Get class names mapping
    class_id_to_name = dict(zip(df['class_id'], df['class_name']))
    unique_classes = sorted(df['class_id'].unique())
    class_names = [class_id_to_name[cid] for cid in unique_classes]
    
    # Metrics
    accuracy = rf.score(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=args.cv_folds, scoring='accuracy')
    print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return {
        "model": rf,
        "accuracy": float(accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }


def save_results(results: dict, feature_names: list, df: pd.DataFrame, 
                 output_dir: Path, model) -> None:
    """Save results and model artifacts."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature importance
    importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nTop 5 Feature Importances:")
    for feat, imp in importance[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    # Save JSON results
    output = {
        "total_samples": int(len(df)),
        "accuracy": results["accuracy"],
        "cv_mean": results["cv_mean"],
        "cv_std": results["cv_std"],
        "class_distribution": df["class_name"].value_counts().to_dict(),
        "feature_importance": {f: float(i) for f, i in importance},
        "confusion_matrix": results["confusion_matrix"]
    }
    
    output_path = output_dir / "rf_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 50)
    print("RF Training - ESPI Mode Classification")
    print("=" * 50)
    
    # Load data
    df = load_data(args.data)
    print(f"\nClass distribution:\n{df['class_name'].value_counts()}")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    
    # Train and evaluate
    results = train_and_evaluate(X, y, df, args)
    
    # Save results
    save_results(results, feature_names, df, args.output, results["model"])
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
