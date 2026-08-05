#!/usr/bin/env python3
"""
YOLO Model Evaluation Script for NEU-DET Industrial Defect Detection

Evaluates trained YOLO models with industrial-specific metrics:
  - Standard metrics: mAP@0.5, mAP@0.5:0.95, precision, recall
  - Industrial KPIs: False Negative Rate (FNR), False Positive Rate (FPR)
  - Per-class analysis for defect-specific performance
  - Confusion matrix and misclassification analysis

Usage:
    python evaluate_yolo.py --model outputs/train_xxx/weights/best.pt --data configs/neudet.yaml
    python evaluate_yolo.py --model best.pt --split val --save-txt

Author: MaxML154
Date: 2026-07-27
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)


CLASS_NAMES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]


def compute_industrial_metrics(results):
    """
    Compute industrial-specific metrics from YOLO validation results.

    Industrial priorities:
      - Low FNR (< 1%): Critical - missing a defect is costly
      - Moderate FPR (< 5%): Acceptable - false alarms can be manually filtered
      - High recall per defect class: Ensure all defect types are detected

    Args:
        results: YOLO validation results object

    Returns:
        dict: Industrial metrics
    """
    metrics = {}

    # Extract standard metrics
    if hasattr(results, 'box'):
        box_metrics = results.box
        metrics['mAP50'] = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
        metrics['mAP50_95'] = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
        metrics['precision'] = float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0
        metrics['recall'] = float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0

        # Compute industrial KPIs
        # FNR = 1 - Recall (percentage of defects missed at detection level)
        metrics['FNR'] = (1.0 - metrics['recall']) * 100.0

        # False discovery rate proxy (not true FPR which needs TN count)
        metrics['false_discovery_rate'] = (1.0 - metrics['precision']) * 100.0

        # Per-class metrics
        if hasattr(box_metrics, 'ap_class_index') and len(box_metrics.ap_class_index) > 0:
            per_class = {}
            # Get class names from results if available
            class_names = getattr(results, 'names', None) or CLASS_NAMES
            for idx, cls_id in enumerate(box_metrics.ap_class_index):
                cls_id_int = int(cls_id)
                if isinstance(class_names, dict):
                    cls_name = class_names.get(cls_id_int, f"class_{cls_id_int}")
                elif isinstance(class_names, list) and cls_id_int < len(class_names):
                    cls_name = class_names[cls_id_int]
                else:
                    cls_name = f"class_{cls_id_int}"

                # Use idx (position in metrics arrays) not cls_id
                if hasattr(box_metrics, 'class_result'):
                    try:
                        class_res = box_metrics.class_result(idx)
                        per_class[cls_name] = {
                            'precision': float(class_res[0]) if len(class_res) > 0 else 0.0,
                            'recall': float(class_res[1]) if len(class_res) > 1 else 0.0,
                            'mAP50': float(class_res[2]) if len(class_res) > 2 else 0.0,
                        }
                    except:
                        per_class[cls_name] = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0}
            if per_class:
                metrics['per_class'] = per_class

    return metrics


def print_evaluation_results(metrics, model_path):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # Standard metrics
    print("\nStandard Object Detection Metrics:")
    print(f"  mAP@0.5:        {metrics.get('mAP50', 0.0):.4f}")
    print(f"  mAP@0.5:0.95:   {metrics.get('mAP50_95', 0.0):.4f}")
    print(f"  Precision:      {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:         {metrics.get('recall', 0.0):.4f}")

    # Industrial KPIs
    print("\nIndustrial KPIs:")
    fnr = metrics.get('FNR', 0.0)
    fdr = metrics.get('false_discovery_rate', 0.0)
    print(f"  False Negative Rate (FNR): {fnr:.2f}%  {'✓ PASS' if fnr < 1.0 else '⚠ HIGH' if fnr < 5.0 else '✗ FAIL'}")
    print(f"  False Discovery Rate:      {fdr:.2f}%  {'✓ PASS' if fdr < 5.0 else '⚠ MODERATE' if fdr < 10.0 else '✗ HIGH'}")

    print("\n  Interpretation:")
    print("    FNR < 1%:  Excellent - industrial quality standard")
    print("    FNR < 5%:  Acceptable - suitable for assisted inspection")
    print("    FNR ≥ 5%:  Poor - missing too many defects")

    # Per-class metrics
    if 'per_class' in metrics:
        print("\nPer-Class Performance:")
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12}")
        print("  " + "-" * 51)
        for cls_name, cls_metrics in metrics['per_class'].items():
            print(f"  {cls_name:<15} "
                  f"{cls_metrics['precision']:<12.4f} "
                  f"{cls_metrics['recall']:<12.4f} "
                  f"{cls_metrics['mAP50']:<12.4f}")

    print("=" * 70)


def save_metrics_json(metrics, output_path):
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics saved to: {output_path}")


def evaluate(args):
    """Execute YOLO model evaluation."""

    # Validate data config
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model_path = Path(args.model)

    print("=" * 70)
    print("YOLO Model Evaluation - NEU-DET Industrial Defect Detection")
    print("=" * 70)
    print(f"Model:   {model_path}")
    print(f"Data:    {data_path}")
    print(f"Split:   {args.split}")
    print(f"Device:  {args.device if args.device else 'auto'}")
    print("=" * 70)

    # Load model
    print(f"\nInitializing model: {model_path}")

    # Check if model exists
    if not model_path.exists():
        # Check common locations for trained models
        common_locations = [
            Path('outputs') / 'train' / 'weights' / model_path.name,
            Path('outputs') / 'train_*' / 'weights' / model_path.name,
            Path('models') / model_path.name,
            Path.home() / '.cache' / 'ultralytics' / model_path.name,
        ]

        found = False
        for loc in common_locations:
            if '*' in str(loc):
                # Handle glob pattern
                matches = list(Path('outputs').glob(f'train*/weights/{model_path.name}'))
                if matches:
                    model_path = matches[0]
                    found = True
                    print(f"✓ Found model at: {model_path}")
                    break
            elif loc.exists():
                model_path = loc
                found = True
                print(f"✓ Found model at: {model_path}")
                break

        if not found:
            print(f"⚠ Model not found locally: {model_path}")
            print(f"Note: If this is a base model name (e.g., yolo11s.pt), Ultralytics will auto-download.")
            print(f"      For custom trained models, check the path is correct.")

    else:
        print(f"✓ Using model at: {model_path}")

    # Load model (Ultralytics handles auto-download for base models)
    try:
        model = YOLO(str(model_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print(f"\nPlease check:")
        print(f"  - Model file exists at: {model_path}")
        print(f"  - Model format is valid (.pt file)")
        print(f"  - For base models (yolo11s.pt), ensure internet connection for auto-download")
        return False

    # Run validation
    print(f"Running evaluation on {args.split} set...\n")

    try:
        # Check if this is a YOLO26 model with end-to-end capability
        is_yolo26 = hasattr(model.model, 'end2end')

        # Determine end2end mode
        if args.end2end == 'both' and is_yolo26:
            print("YOLO26 model detected - running dual-head evaluation")
            print("-" * 70)

            # Evaluate one-to-one head (end2end=True)
            print("\n[1/2] Evaluating one-to-one head (end2end, NMS-free)...")
            results_e2e = model.val(
                data=str(data_path),
                split=args.split,
                batch=args.batch,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device if args.device else '',
                save_json=args.save_json,
                save_hybrid=args.save_hybrid,
                plots=args.plots,
                verbose=args.verbose,
                end2end=True,
                name=f'eval_e2e_{args.split}'
            )
            metrics_e2e = compute_industrial_metrics(results_e2e)
            metrics_e2e['eval_mode'] = 'one-to-one (end2end=True)'

            # Reload model to reset state for one-to-many head evaluation
            print("\n[2/2] Evaluating one-to-many head (traditional + NMS)...")
            print("Reloading model to reset detection head state...")
            model = YOLO(str(model_path))

            results_o2m = model.val(
                data=str(data_path),
                split=args.split,
                batch=args.batch,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device if args.device else '',
                save_json=False,  # Only save once
                save_hybrid=False,
                plots=False,  # Avoid overwriting
                verbose=args.verbose,
                end2end=False,
                name=f'eval_o2m_{args.split}'
            )
            metrics_o2m = compute_industrial_metrics(results_o2m)
            metrics_o2m['eval_mode'] = 'one-to-many (end2end=False)'

            # Compute differences
            metrics_diff = {
                'mAP50': metrics_o2m['mAP50'] - metrics_e2e['mAP50'],
                'mAP50_95': metrics_o2m['mAP50_95'] - metrics_e2e['mAP50_95'],
                'precision': metrics_o2m['precision'] - metrics_e2e['precision'],
                'recall': metrics_o2m['recall'] - metrics_e2e['recall'],
            }

            # Print comparison
            print("\n" + "=" * 70)
            print("YOLO26 Dual-Head Comparison")
            print("=" * 70)
            print(f"\n{'Metric':<20} {'One-to-One':<15} {'One-to-Many':<15} {'Difference':<15}")
            print("-" * 70)
            print(f"{'mAP@0.5':<20} {metrics_e2e['mAP50']:<15.4f} {metrics_o2m['mAP50']:<15.4f} {metrics_diff['mAP50']:+.4f}")
            print(f"{'mAP@0.5:0.95':<20} {metrics_e2e['mAP50_95']:<15.4f} {metrics_o2m['mAP50_95']:<15.4f} {metrics_diff['mAP50_95']:+.4f}")
            print(f"{'Precision':<20} {metrics_e2e['precision']:<15.4f} {metrics_o2m['precision']:<15.4f} {metrics_diff['precision']:+.4f}")
            print(f"{'Recall':<20} {metrics_e2e['recall']:<15.4f} {metrics_o2m['recall']:<15.4f} {metrics_diff['recall']:+.4f}")
            print("=" * 70)

            # Save combined results
            combined_metrics = {
                'one_to_one': metrics_e2e,
                'one_to_many': metrics_o2m,
                'difference': metrics_diff,
                'metadata': {
                    'model': str(model_path),
                    'data': str(data_path),
                    'split': args.split,
                    'timestamp': datetime.now().isoformat(),
                    'conf_threshold': args.conf,
                    'iou_threshold': args.iou,
                    'imgsz': args.imgsz,
                    'batch': args.batch,
                    'end2end_mode': 'both',
                    'save_dir_e2e': str(results_e2e.save_dir),
                    'save_dir_o2m': str(results_o2m.save_dir),
                }
            }

            if args.output:
                save_metrics_json(combined_metrics, args.output)
            else:
                output_dir = model_path.parent.parent / 'evaluation'
                output_file = output_dir / f"dual_head_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_metrics_json(combined_metrics, output_file)

            if args.save_txt:
                print(f"\n✓ One-to-one predictions saved to: {results_e2e.save_dir}")
                print(f"✓ One-to-many predictions saved to: {results_o2m.save_dir}")

            return True

        else:
            # Single evaluation mode
            end2end_arg = {}
            if args.end2end != 'auto' and is_yolo26:
                end2end_arg['end2end'] = (args.end2end == 'true')
                print(f"YOLO26 model - using end2end={end2end_arg['end2end']}")

            results = model.val(
                data=str(data_path),
                split=args.split,
                batch=args.batch,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device if args.device else '',
                save_json=args.save_json,
                save_hybrid=args.save_hybrid,
                save_txt=args.save_txt,
                plots=args.plots,
                verbose=args.verbose,
                **end2end_arg
            )

            # Compute industrial metrics
            metrics = compute_industrial_metrics(results)

            # Add metadata
            metrics['metadata'] = {
                'model': str(model_path),
                'data': str(data_path),
                'split': args.split,
                'timestamp': datetime.now().isoformat(),
                'conf_threshold': args.conf,
                'iou_threshold': args.iou,
                'imgsz': args.imgsz,
                'batch': args.batch,
                'end2end_mode': args.end2end if is_yolo26 else 'N/A',
                'save_dir': str(results.save_dir),
            }

            # Print results
            print_evaluation_results(metrics, model_path)

            # Save metrics to JSON
            if args.output:
                save_metrics_json(metrics, args.output)
            else:
                # Auto-generate output path
                output_dir = model_path.parent.parent / 'evaluation'
                output_file = output_dir / f"metrics_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_metrics_json(metrics, output_file)

            # Save predictions to txt files
            if args.save_txt:
                print(f"\n✓ Predictions saved to: {results.save_dir}")

            return True

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO models on NEU-DET with industrial metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python evaluate_yolo.py --model outputs/train_xxx/weights/best.pt

    # Evaluate on validation set with custom confidence threshold
    python evaluate_yolo.py --model best.pt --split val --conf 0.25

    # Save predictions and plots
    python evaluate_yolo.py --model best.pt --save-txt --plots

    # Custom output path
    python evaluate_yolo.py --model best.pt --output results/metrics.json

Industrial KPIs:
    FNR (False Negative Rate) - Critical metric for defect detection
        < 1%:  Excellent - meets industrial quality standards
        < 5%:  Acceptable - suitable for assisted inspection
        ≥ 5%:  Poor - missing too many defects

    FPR (False Positive Rate) - Secondary metric
        < 5%:  Good - minimal false alarms
        < 10%: Moderate - acceptable with manual filtering
        ≥ 10%: High - too many false alarms
        """
    )

    # Core arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='configs/neudet.yaml',
        help='Path to dataset YAML config (default: configs/neudet.yaml)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate (default: val)'
    )

    # Evaluation parameters
    parser.add_argument(
        '--imgsz',
        type=int,
        default=224,
        help='Image size for evaluation (default: 224)'
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='IoU threshold for NMS (default: 0.6)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device: "", "cpu", "0", "0,1", etc. (default: auto-detect)'
    )

    parser.add_argument(
        '--end2end',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false', 'both'],
        help='YOLO26 end-to-end mode: auto (framework default), true (one-to-one), false (one-to-many+NMS), both (compare both heads)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for metrics JSON (default: auto-generated)'
    )

    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save predictions to txt files'
    )

    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save predictions to COCO JSON format'
    )

    parser.add_argument(
        '--save-hybrid',
        action='store_true',
        help='Save hybrid labels (for auto-labeling)'
    )

    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate evaluation plots (confusion matrix, PR curves)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed evaluation logs'
    )

    args = parser.parse_args()

    # Execute evaluation
    success = evaluate(args)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
