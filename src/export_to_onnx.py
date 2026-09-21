"""
Export a trained YOLO checkpoint to ONNX (raw boxes, NMS left to post-process).

Author: MaxML154
Created: 2026-08-05
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Import error: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from model_io import MODELS_DIR, resolve_weight


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO weights to ONNX. NMS is not baked into the graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Checkpoint path or models/ filename")
    parser.add_argument(
        "--output",
        default=None,
        help="ONNX output path (default: models/<stem>_<head>.onnx)",
    )
    parser.add_argument(
        "--end2end",
        choices=["true", "false"],
        default="false",
        help="YOLO26 head: false=one-to-many (recommended), true=one-to-one",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export input size")
    parser.add_argument("--batch", type=int, default=1, help="Fixed batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--device", default="cpu", help="Export device")
    parser.add_argument("--half", action="store_true", help="FP16 weights (GPU deployment)")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch/spatial axes")
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX graph simplification",
    )
    return parser.parse_args()


def default_output(model_path: Path, end2end: bool) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_one_to_one" if end2end else "_one_to_many"
    return MODELS_DIR / f"{model_path.stem}{suffix}.onnx"


def main() -> int:
    args = parse_args()
    end2end = args.end2end == "true"

    try:
        model_path = resolve_weight(args.model, download=False)
    except FileNotFoundError as e:
        print(e)
        return 1

    output_path = Path(args.output) if args.output else default_output(model_path, end2end)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YOLO ONNX export")
    print("=" * 70)
    print(f"Weights:     {model_path}")
    print(f"Output:      {output_path}")
    print(f"imgsz:       {args.imgsz}")
    print(f"batch:       {args.batch}{' (dynamic)' if args.dynamic else ''}")
    print(f"opset:       {args.opset}")
    print(f"precision:   {'FP16' if args.half else 'FP32'}")
    print(f"simplify:    {not args.no_simplify}")
    print(f"head:        {'one-to-one' if end2end else 'one-to-many'}")
    print("NMS:         post-process (not in graph)")
    print("=" * 70)

    model = YOLO(str(model_path))
    is_yolo26 = hasattr(model.model, "end2end")
    if is_yolo26:
        print(f"YOLO26 detected; exporting end2end={end2end}")
    elif end2end:
        print("Warning: --end2end true is ignored on non-YOLO26 models")

    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "simplify": not args.no_simplify,
        "opset": args.opset,
        "half": args.half,
        "device": args.device,
        "nms": False,
    }
    if is_yolo26:
        export_kwargs["end2end"] = end2end

    try:
        exported = Path(model.export(**export_kwargs))
    except Exception as e:
        print(f"Export failed: {e}")
        return 1

    if exported.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported), str(output_path))

    info_path = output_path.with_suffix(".export_info.txt")
    lines = [
        "YOLO ONNX Export Information",
        "=" * 70,
        f"Export time     : {datetime.now().isoformat()}",
        f"Source weights  : {model_path}",
        f"ONNX            : {output_path}",
        f"Input           : batch={args.batch}, 3x{args.imgsz}x{args.imgsz}",
        f"Opset           : {args.opset}",
        f"Precision       : {'FP16' if args.half else 'FP32'}",
        f"Simplified      : {not args.no_simplify}",
        f"Detection head  : {'one-to-one' if end2end else 'one-to-many'}",
        "NMS             : not in graph (apply after inference)",
        "=" * 70,
        f"Command: {' '.join(sys.argv)}",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved ONNX: {output_path}")
    print(f"Saved info: {info_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
