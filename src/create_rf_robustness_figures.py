#!/usr/bin/env python3
"""
Create figures and tables for RF Robustness Analysis (Section 4.5)

Usage:
    python create_rf_robustness_figures.py --results path/to/results --output figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create RF robustness analysis figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results", "-r",
        type=Path,
        required=True,
        help="Directory containing analysis JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures"
    )
    return parser.parse_args()


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_data(results_dir: Path) -> tuple:
    """Load all RF analysis data from results directory."""
    data = {}
    
    # Try to load each file (some may not exist)
    file_mapping = {
        'pattern': 'report_pattern.json',
        'robustness': 'robustness_pattern_only.json',
        'stability': 'feature_stability.json',
        'lodo': 'lodo_results.json',
        'lobo': 'lobo_results.json'
    }
    
    for key, filename in file_mapping.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
            print(f"  Loaded: {filename}")
        else:
            data[key] = None
            print(f"  Not found: {filename}")
    
    return data


def create_confusion_matrix_heatmap(pattern_data: dict, output_dir: Path) -> None:
    """Create normalized confusion matrix heatmap."""
    if not pattern_data or 'confusion_matrix' not in pattern_data:
        print("  Skipping confusion matrix (no data)")
        return
    
    cm = np.array(pattern_data['confusion_matrix']['raw'])
    labels = pattern_data['confusion_matrix']['labels']
    
    # Normalize by true labels
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized Count'})
    plt.title('PatternOnly RF: Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_pattern_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Confusion matrix heatmap")


def create_feature_importance_plot(pattern_data: dict, output_dir: Path) -> None:
    """Create feature importance bar plot."""
    if not pattern_data or 'feature_importances' not in pattern_data:
        print("  Skipping feature importance (no data)")
        return
    
    importances = pattern_data['feature_importances']
    
    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_features)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(features)), values, color='steelblue', alpha=0.7)
    
    # Color top 5 features differently
    for i in range(min(5, len(bars))):
        bars[i].set_color('darkred')
        bars[i].set_alpha(0.8)
    
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('PatternOnly RF: Feature Importance Analysis', fontsize=16, fontweight='bold')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_pattern_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Feature importance plot")


def create_bootstrap_ci_plot(robustness_data: dict, output_dir: Path) -> None:
    """Create bootstrap confidence interval plot."""
    if not robustness_data or 'bootstrap_ci' not in robustness_data:
        print("  Skipping bootstrap CI (no data)")
        return
    
    metrics = ['accuracy', 'macro_f1']
    metric_names = ['Accuracy', 'Macro-F1']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = ax1 if i == 0 else ax2
        ci_data = robustness_data['bootstrap_ci'][metric]
        
        mean_val = ci_data['mean']
        ci_low, ci_high = ci_data['ci']
        
        # Create bar plot with error bars
        ax.bar([name], [mean_val], color='steelblue', alpha=0.7,
               yerr=[[mean_val - ci_low], [ci_high - mean_val]],
               capsize=10)
        
        # Add confidence interval text
        ax.text(0, mean_val + (ci_high - mean_val) + 0.01,
               f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{name} Bootstrap 95% CI', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean value label
        ax.text(0, mean_val, f'{mean_val:.3f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    plt.suptitle('PatternOnly RF: Bootstrap Confidence Intervals', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'bootstrap_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Bootstrap CI plot")


def create_lodo_analysis_plot(lodo_data: list, output_dir: Path) -> None:
    """Create LODO analysis plot."""
    if not lodo_data:
        print("  Skipping LODO plot (no data)")
        return
    
    datasets = [item['test_dataset'] for item in lodo_data]
    accuracies = [item['accuracy'] for item in lodo_data]
    macro_f1s = [item['macro_f1'] for item in lodo_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['steelblue', 'darkred', 'green'][:len(datasets)]
    
    # Accuracy plot
    bars1 = ax1.bar(datasets, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('LODO: Accuracy by Test Dataset', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Macro-F1 plot
    bars2 = ax2.bar(datasets, macro_f1s, color=colors, alpha=0.7)
    ax2.set_ylabel('Macro-F1', fontsize=12)
    ax2.set_title('LODO: Macro-F1 by Test Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, macro_f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Leave-One-Dataset-Out Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'lodo_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] LODO analysis plot")


def create_lobo_frequency_plot(lobo_data: list, output_dir: Path) -> None:
    """Create LOBO analysis by frequency range."""
    if not lobo_data:
        print("  Skipping LOBO plot (no data)")
        return
    
    freq_ranges = [item['freq_range'] for item in lobo_data]
    accuracies = [item['accuracy'] for item in lobo_data]
    
    df = pd.DataFrame({'freq_range': freq_ranges, 'accuracy': accuracies})
    df['freq_start'] = df['freq_range'].str.split('-').str[0].astype(int)
    df = df.sort_values('freq_start')
    
    plt.figure(figsize=(20, 8))
    
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.8 else 'green' for acc in df['accuracy']]
    plt.scatter(df['freq_start'], df['accuracy'], c=colors, alpha=0.7, s=60)
    
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (80%)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Poor (50%)')
    
    plt.xlabel('Frequency Range Start (Hz)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Leave-One-Bin-Out: Accuracy by Frequency Range', fontsize=16, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate problem regions
    for _, row in df[df['accuracy'] < 0.5].iterrows():
        plt.annotate(f"{row['freq_range']}\n{row['accuracy']:.2f}",
                    (row['freq_start'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lobo_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] LOBO frequency plot")


def create_performance_comparison_table(pattern_data: dict, output_dir: Path) -> pd.DataFrame:
    """Create performance comparison table."""
    if not pattern_data or 'classification_report' not in pattern_data:
        print("  Skipping performance table (no data)")
        return None
    
    class_report = pattern_data['classification_report']
    
    data = []
    for class_name, metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            data.append({
                'Class': class_name,
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1-score']:.3f}",
                'Support': int(metrics['support'])
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'per_class_performance.csv', index=False)
    
    # Create markdown table
    markdown_lines = ["# Per-Class Performance Table\n"]
    markdown_lines.append("| " + " | ".join(df.columns) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    
    for _, row in df.iterrows():
        markdown_lines.append("| " + " | ".join(str(val) for val in row) + " |")
    
    with open(output_dir / 'per_class_performance_table.md', 'w') as f:
        f.write("\n".join(markdown_lines))
    
    print("  [OK] Performance table")
    return df


def create_feature_stability_plot(stability_data: dict, output_dir: Path) -> None:
    """Create feature stability visualization."""
    if not stability_data or 'stability_analysis' not in stability_data:
        print("  Skipping stability plot (no data)")
        return
    
    stability_info = stability_data['stability_analysis']
    
    top_features = list(stability_info['top_10_features'].keys())
    top_importances = list(stability_info['top_10_features'].values())
    stability_counts = stability_info['stability_count']
    stability_scores = [stability_counts.get(feat, 0) for feat in top_features]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Feature importance plot
    bars1 = ax1.bar(range(len(top_features)), top_importances, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Features', fontsize=12)
    ax1.set_ylabel('Average Importance', fontsize=12)
    ax1.set_title('Top 10 Features: Average Importance Across Seeds', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(top_features)))
    ax1.set_xticklabels(top_features, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, top_importances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Stability count plot
    colors = ['red' if s < 3 else 'orange' if s < 5 else 'green' for s in stability_scores]
    bars2 = ax2.bar(range(len(top_features)), stability_scores, color=colors, alpha=0.7)
    ax2.set_xlabel('Features', fontsize=12)
    ax2.set_ylabel('Stability Count (out of 5 seeds)', fontsize=12)
    ax2.set_title('Feature Stability Across 5 Random Seeds', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(top_features)))
    ax2.set_xticklabels(top_features, rotation=45, ha='right')
    ax2.set_ylim(0, 5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, stability_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}/5', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Feature Importance Stability Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Feature stability plot")


def main():
    """Main function to create all figures and tables."""
    args = parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Creating RF Robustness Analysis Figures")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    data = load_data(args.results)
    
    # Create figures
    print("\nGenerating figures...")
    create_confusion_matrix_heatmap(data['pattern'], args.output)
    create_feature_importance_plot(data['pattern'], args.output)
    create_bootstrap_ci_plot(data['robustness'], args.output)
    create_lodo_analysis_plot(data['lodo'], args.output)
    create_lobo_frequency_plot(data['lobo'], args.output)
    create_feature_stability_plot(data['stability'], args.output)
    create_performance_comparison_table(data['pattern'], args.output)
    
    # List generated files
    print(f"\n{'=' * 50}")
    print(f"All figures saved to: {args.output}")
    print("Generated files:")
    for file in sorted(args.output.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
