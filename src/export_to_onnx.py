#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export YOLO models to ONNX format for deployment.

Author: MaxML154
Created: 2026-08-05
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages: pip install ultralytics torch")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export YOLO models to ONNX format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to YOLO model weights (.pt file)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: same as model with .onnx extension)'
    )

    # YOLO26 specific
    parser.add_argument(
        '--end2end',
        type=str,
        default='false',
        choices=['true', 'false', 'auto'],
        help='YOLO26 detection head: true (one-to-one), false (one-to-many+NMS), auto (framework default)'
    )

    # Export parameters
    parser.add_argument(
        '--imgsz',
        type=int,
        nargs='+',
        default=[640],
        help='Input image size (height width) or single size for square images'
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Batch size for export (use -1 for dynamic batch)'
    )

    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic axes for batch size and image dimensions'
    )

    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX model using onnxslim/onnx-simplifier'
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version'
    )

    parser.add_argument(
        '--half',
        action='store_true',
        help='Export in FP16 (half precision) mode'
    )

    parser.add_argument(
        '--int8',
        action='store_true',
        help='Export with INT8 quantization (requires calibration data)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for export (cpu, cuda, cuda:0, etc.)'
    )

    # Metadata
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Include training metadata in exported model'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed export information'
    )

    return parser.parse_args()


def validate_model_path(model_path):
    """Validate model path exists and is a .pt file."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.suffix != '.pt':
        raise ValueError(f"Model must be a .pt file, got: {model_path.suffix}")

    return model_path


def determine_output_path(model_path, output_arg, end2end_mode):
    """Determine the output ONNX file path."""
    if output_arg:
        output_path = Path(output_arg)
    else:
        # Auto-generate output path
        model_path = Path(model_path)
        parent_dir = model_path.parent

        # Add end2end mode suffix for YOLO26
        if end2end_mode == 'true':
            suffix = '_one_to_one'
        elif end2end_mode == 'false':
            suffix = '_one_to_many'
        else:
            suffix = ''

        output_path = parent_dir / f"{model_path.stem}{suffix}.onnx"

    return output_path


def export_model(model, args, output_path):
    """Export YOLO model to ONNX format."""

    print("\n" + "=" * 70)
    print("YOLO Model Export to ONNX")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Output:       {output_path}")
    print(f"Image size:   {args.imgsz}")
    print(f"Batch size:   {'Dynamic' if args.dynamic or args.batch == -1 else args.batch}")
    print(f"Device:       {args.device}")
    print(f"ONNX opset:   {args.opset}")
    print(f"Simplify:     {args.simplify}")
    print(f"Half (FP16):  {args.half}")
    print(f"INT8:         {args.int8}")

    # YOLO26 specific
    if args.end2end != 'auto':
        print(f"End2end mode: {args.end2end} ({'one-to-one' if args.end2end == 'true' else 'one-to-many+NMS'})")

    print("=" * 70)

    # Prepare export parameters
    export_kwargs = {
        'format': 'onnx',
        'imgsz': args.imgsz if len(args.imgsz) > 1 else args.imgsz[0],
        'batch': args.batch,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'half': args.half,
        'int8': args.int8,
        'device': args.device,
    }

    # Add end2end parameter for YOLO26
    if args.end2end == 'true':
        export_kwargs['end2end'] = True
    elif args.end2end == 'false':
        export_kwargs['end2end'] = False
    # 'auto' means don't set it, let framework decide

    print("\nStarting export...")
    print("-" * 70)

    try:
        # Export model
        export_path = model.export(**export_kwargs)

        print("-" * 70)
        print(f"✓ Export successful!")
        print(f"✓ Saved to: {export_path}")

        # Rename if custom output path specified
        export_path_obj = Path(export_path)
        if export_path_obj != output_path:
            if output_path.exists():
                print(f"Warning: Overwriting existing file: {output_path}")
            export_path_obj.rename(output_path)
            print(f"✓ Renamed to: {output_path}")

        # Get model info
        model_info = {
            'export_time': datetime.now().isoformat(),
            'input_shape': f"batch={args.batch if not args.dynamic else 'dynamic'}, channels=3, height={args.imgsz[0] if len(args.imgsz) > 1 else args.imgsz}, width={args.imgsz[-1]}",
            'opset_version': args.opset,
            'precision': 'FP16' if args.half else ('INT8' if args.int8 else 'FP32'),
            'simplified': args.simplify,
        }

        if args.end2end != 'auto':
            model_info['detection_head'] = 'one-to-one (NMS-free)' if args.end2end == 'true' else 'one-to-many (with NMS)'

        print("\n" + "=" * 70)
        print("Export Summary")
        print("=" * 70)
        for key, value in model_info.items():
            print(f"{key.replace('_', ' ').title():<20}: {value}")
        print("=" * 70)

        # Save export info
        info_path = output_path.with_suffix('.export_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("YOLO ONNX Export Information\n")
            f.write("=" * 70 + "\n\n")
            for key, value in model_info.items():
                f.write(f"{key.replace('_', ' ').title():<20}: {value}\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")

        print(f"\n✓ Export information saved to: {info_path}")

        return True

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        # Validate model path
        model_path = validate_model_path(args.model)

        # Determine output path
        output_path = determine_output_path(model_path, args.output, args.end2end)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load model
        print(f"\nLoading model from: {model_path}")
        model = YOLO(str(model_path))

        # Check if YOLO26
        is_yolo26 = hasattr(model.model, 'end2end')
        if is_yolo26:
            print("✓ YOLO26 model detected")
            if args.end2end == 'auto':
                print("  Warning: Using 'auto' mode for YOLO26. Specify --end2end true/false for explicit head selection.")
        else:
            print("✓ Standard YOLO model detected")
            if args.end2end != 'auto':
                print(f"  Warning: --end2end {args.end2end} has no effect on non-YOLO26 models")

        # Export model
        success = export_model(model, args, output_path)

        if success:
            print("\n" + "=" * 70)
            print("ONNX export completed successfully!")
            print("=" * 70)
            print("\nNext steps:")
            print("1. Test the exported ONNX model with ONNXRuntime")
            print("2. Optimize for target deployment platform (TensorRT, OpenVINO, etc.)")
            print("3. Benchmark inference speed and accuracy")
            print("=" * 70)
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
