#!/usr/bin/env python3
"""
Execute Features Extraction and Final RF Training Pipeline

Complete pipeline from raw phase data to classification results.
Note: This script orchestrates external scripts that must exist in your project.

Usage:
    python execute_features_rf_final.py --config config.yaml
    python execute_features_rf_final.py --data-roots W01/phase W02/phase W03/phase --output results/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute complete features extraction and RF training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to phase output directories for each dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for features and model"
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path("scripts"),
        help="Directory containing pipeline scripts"
    )
    return parser.parse_args()


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {str(e)}")
        return False


def run_features_extraction(phase_root: Path, dataset_name: str, 
                           scripts_dir: Path) -> bool:
    """Run features extraction for a dataset."""
    if not phase_root.exists():
        print(f"❌ Phase root not found: {phase_root}")
        return False
    
    script = scripts_dir / "phase_nodal_features_min.py"
    if not script.exists():
        print(f"⚠️  Script not found: {script}")
        return False
    
    cmd = [sys.executable, str(script), "--band-root", str(phase_root)]
    return run_command(cmd, f"Features extraction for {dataset_name}")


def run_merge_features(data_roots: list, output_dir: Path, 
                       scripts_dir: Path) -> bool:
    """Run features merging for all datasets."""
    script = scripts_dir / "merge_all_features.py"
    if not script.exists():
        print(f"⚠️  Script not found: {script}")
        return False
    
    cmd = [
        sys.executable, str(script),
        "--roots", *[str(r) for r in data_roots],
        "--out", str(output_dir / "all_features_QCpass.csv")
    ]
    return run_command(cmd, "Features merging")


def run_dedup_and_numeric(output_dir: Path, scripts_dir: Path) -> bool:
    """Run deduplication and numeric features creation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Deduplication
    dedup_script = scripts_dir / "simple_dedup.py"
    if dedup_script.exists():
        cmd = [
            sys.executable, str(dedup_script),
            "--inp", str(output_dir / "all_features_QCpass.csv"),
            "--out", str(output_dir / "all_features_QCpass_dedup.csv")
        ]
        if not run_command(cmd, "Deduplication"):
            return False
    
    # Numeric features
    numeric_script = scripts_dir / "create_numeric_features.py"
    if numeric_script.exists():
        cmd = [
            sys.executable, str(numeric_script),
            "--inp", str(output_dir / "all_features_QCpass_dedup.csv"),
            "--out", str(output_dir / "features_numeric_only.csv")
        ]
        if not run_command(cmd, "Numeric features creation"):
            return False
    
    return True


def run_labels_creation(output_dir: Path, scripts_dir: Path) -> bool:
    """Run labels creation."""
    script = scripts_dir / "create_labels_corrected.py"
    if not script.exists():
        print(f"⚠️  Script not found: {script}")
        return False
    
    cmd = [
        sys.executable, str(script),
        "--inp", str(output_dir / "features_numeric_only.csv"),
        "--out", str(output_dir / "labels_5class.csv")
    ]
    return run_command(cmd, "Labels creation")


def run_final_rf_training(output_dir: Path, scripts_dir: Path) -> bool:
    """Run final RF training with hierarchical approach."""
    script = scripts_dir / "hierarchical_rf_classifier.py"
    if not script.exists():
        # Fallback to our rf_train_complete.py
        script = Path(__file__).parent / "rf_train_complete.py"
    
    if not script.exists():
        print(f"⚠️  No training script found")
        return False
    
    cmd = [
        sys.executable, str(script),
        "--data", str(output_dir / "features_numeric_only.csv"),
        "--output", str(output_dir / "rf_model_final")
    ]
    return run_command(cmd, "Final RF training")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 60)
    print("🧮 FEATURES & RF TRAINING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    features_dir = args.output / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Step 1: Features extraction for each dataset
    print("\n📊 Step 1: Features Extraction")
    features_success = 0
    for data_root in args.data_roots:
        dataset_name = data_root.name
        if run_features_extraction(data_root, dataset_name, args.scripts_dir):
            features_success += 1
    
    if features_success != len(args.data_roots):
        print(f"⚠️  Only {features_success}/{len(args.data_roots)} datasets processed")
    
    # Step 2: Merge features
    print("\n🔗 Step 2: Features Merging")
    if not run_merge_features(args.data_roots, features_dir, args.scripts_dir):
        print("⚠️  Merge step skipped or failed")
    
    # Step 3: Dedup and numeric
    print("\n🔧 Step 3: Deduplication & Numeric Features")
    if not run_dedup_and_numeric(features_dir, args.scripts_dir):
        print("⚠️  Dedup/numeric step skipped or failed")
    
    # Step 4: Labels creation
    print("\n🏷️  Step 4: Labels Creation")
    if not run_labels_creation(features_dir, args.scripts_dir):
        print("⚠️  Labels step skipped or failed")
    
    # Step 5: Final RF training
    print("\n🌲 Step 5: Final RF Training")
    if not run_final_rf_training(features_dir, args.scripts_dir):
        print("⚠️  Training step skipped or failed")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE COMPLETE")
    print(f"⏱️  Total time: {duration/60:.1f} minutes")
    print(f"📁 Results in: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()