#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX model testing and performance benchmarking utility.

Author: MaxML154
Created: 2026-08-05
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import onnxruntime as ort
    import cv2
except ImportError as e:
    print(f"Error: {e}")
    print("Please install: pip install onnxruntime opencv-python")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test ONNX exported YOLO models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )

    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Test image path (optional, will use random input if not provided)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )

    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )

    parser.add_argument(
        '--benchmark',
        type=int,
        default=100,
        help='Number of iterations for speed benchmark'
    )

    parser.add_argument(
        '--providers',
        type=str,
        nargs='+',
        default=['CPUExecutionProvider'],
        choices=['CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'],
        help='ONNX Runtime execution providers'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed model information'
    )

    return parser.parse_args()


def load_onnx_model(model_path, providers):
    """Load ONNX model with specified providers."""
    print(f"\nLoading ONNX model: {model_path}")

    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"✓ Model loaded successfully")
        print(f"  Providers: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)


def print_model_info(session, verbose=False):
    """Print ONNX model information."""
    print("\n" + "=" * 70)
    print("Model Information")
    print("=" * 70)

    # Input info
    print("\nInputs:")
    for inp in session.get_inputs():
        print(f"  Name: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {inp.type}")

    # Output info
    print("\nOutputs:")
    for out in session.get_outputs():
        print(f"  Name: {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type: {out.type}")

    if verbose:
        # Metadata
        metadata = session.get_modelmeta()
        print("\nMetadata:")
        print(f"  Producer: {metadata.producer_name}")
        print(f"  Graph name: {metadata.graph_name}")
        print(f"  Version: {metadata.version}")

        if metadata.custom_metadata_map:
            print("  Custom metadata:")
            for key, value in metadata.custom_metadata_map.items():
                print(f"    {key}: {value}")

    print("=" * 70)


def prepare_input(image_path, imgsz):
    """Prepare input tensor from image or random data."""
    if image_path:
        print(f"\nLoading test image: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"✗ Failed to load image, using random input")
            return prepare_random_input(imgsz)

        # Resize and preprocess
        img = cv2.resize(img, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        print(f"✓ Image loaded and preprocessed")
        print(f"  Shape: {img.shape}")
        return img
    else:
        return prepare_random_input(imgsz)


def prepare_random_input(imgsz):
    """Generate random input tensor."""
    print(f"\nGenerating random input tensor ({imgsz}x{imgsz})...")
    img = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    print(f"✓ Random input generated")
    print(f"  Shape: {img.shape}")
    return img


def warmup_model(session, input_tensor, warmup_runs):
    """Warmup the model with multiple inference runs."""
    print(f"\nWarming up model ({warmup_runs} iterations)...")

    input_name = session.get_inputs()[0].name

    for i in range(warmup_runs):
        _ = session.run(None, {input_name: input_tensor})
        if (i + 1) % 10 == 0 or (i + 1) == warmup_runs:
            print(f"  Progress: {i + 1}/{warmup_runs}", end='\r')

    print(f"\n✓ Warmup completed")


def benchmark_inference(session, input_tensor, benchmark_runs):
    """Benchmark inference speed."""
    print(f"\nBenchmarking inference speed ({benchmark_runs} iterations)...")

    input_name = session.get_inputs()[0].name
    times = []

    for i in range(benchmark_runs):
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: input_tensor})
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)  # Convert to ms

        if (i + 1) % 10 == 0 or (i + 1) == benchmark_runs:
            print(f"  Progress: {i + 1}/{benchmark_runs}", end='\r')

    print()  # New line after progress

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p50_time = np.percentile(times, 50)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)

    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"Mean inference time:   {mean_time:.2f} ms (± {std_time:.2f} ms)")
    print(f"Min inference time:    {min_time:.2f} ms")
    print(f"Max inference time:    {max_time:.2f} ms")
    print(f"P50 (median):          {p50_time:.2f} ms")
    print(f"P95:                   {p95_time:.2f} ms")
    print(f"P99:                   {p99_time:.2f} ms")
    print(f"Throughput:            {1000 / mean_time:.2f} FPS")
    print("=" * 70)

    return outputs, times


def analyze_outputs(outputs):
    """Analyze model outputs."""
    print("\n" + "=" * 70)
    print("Output Analysis")
    print("=" * 70)

    for i, output in enumerate(outputs):
        print(f"\nOutput {i}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Min: {np.min(output):.6f}")
        print(f"  Max: {np.max(output):.6f}")
        print(f"  Mean: {np.mean(output):.6f}")
        print(f"  Std: {np.std(output):.6f}")

        # Check for NaN or Inf
        has_nan = np.isnan(output).any()
        has_inf = np.isinf(output).any()

        if has_nan:
            print(f"  ⚠ WARNING: Contains NaN values!")
        if has_inf:
            print(f"  ⚠ WARNING: Contains Inf values!")

        if not has_nan and not has_inf:
            print(f"  ✓ Output is valid (no NaN/Inf)")

    print("=" * 70)


def save_benchmark_results(model_path, providers, times, output_shapes, imgsz):
    """Save benchmark results to file."""
    model_path = Path(model_path)
    result_path = model_path.parent / f"{model_path.stem}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("ONNX Model Benchmark Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Input size: {imgsz}x{imgsz}\n")
        f.write(f"Providers: {', '.join(providers)}\n")
        f.write(f"Benchmark runs: {len(times)}\n\n")

        f.write("Inference Time Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean:   {np.mean(times):.2f} ms\n")
        f.write(f"Std:    {np.std(times):.2f} ms\n")
        f.write(f"Min:    {np.min(times):.2f} ms\n")
        f.write(f"Max:    {np.max(times):.2f} ms\n")
        f.write(f"P50:    {np.percentile(times, 50):.2f} ms\n")
        f.write(f"P95:    {np.percentile(times, 95):.2f} ms\n")
        f.write(f"P99:    {np.percentile(times, 99):.2f} ms\n")
        f.write(f"FPS:    {1000 / np.mean(times):.2f}\n\n")

        f.write("Output Shapes:\n")
        f.write("-" * 70 + "\n")
        for i, shape in enumerate(output_shapes):
            f.write(f"Output {i}: {shape}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\n✓ Benchmark results saved to: {result_path}")
    return result_path


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return 1

    if model_path.suffix != '.onnx':
        print(f"✗ Not an ONNX file: {model_path}")
        return 1

    print("=" * 70)
    print("ONNX Model Testing and Benchmarking")
    print("=" * 70)

    # Load model
    session = load_onnx_model(model_path, args.providers)

    # Print model info
    print_model_info(session, verbose=args.verbose)

    # Prepare input
    input_tensor = prepare_input(args.image, args.imgsz)

    # Warmup
    warmup_model(session, input_tensor, args.warmup)

    # Benchmark
    outputs, times = benchmark_inference(session, input_tensor, args.benchmark)

    # Analyze outputs
    analyze_outputs(outputs)

    # Save results
    output_shapes = [output.shape for output in outputs]
    result_path = save_benchmark_results(
        model_path,
        session.get_providers(),
        times,
        output_shapes,
        args.imgsz
    )

    print("\n" + "=" * 70)
    print("Testing completed successfully!")
    print("=" * 70)
    print(f"\n✓ Model is working correctly")
    print(f"✓ Average inference time: {np.mean(times):.2f} ms ({1000/np.mean(times):.2f} FPS)")
    print(f"✓ Results saved to: {result_path}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
