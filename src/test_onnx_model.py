"""
Load an exported ONNX detector, run NMS in post-process, and optionally compare to PyTorch.

Author: MaxML154
Created: 2026-08-05
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    import onnxruntime as ort
except ImportError as e:
    print(f"Import error: {e}")
    print("Install: pip install onnxruntime opencv-python")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from model_io import resolve_weight

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX inference and decode detections with NMS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="ONNX path or models/ filename")
    parser.add_argument("--image", default=None, help="Test image (random tensor if omitted)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for NMS")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--benchmark", type=int, default=100)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime providers",
    )
    parser.add_argument(
        "--compare-pt",
        default=None,
        help="Optional .pt checkpoint to compare decoded boxes on --image",
    )
    parser.add_argument("--save", default=None, help="Optional path to save a visualization")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(image_path: str | None, imgsz: int):
    if not image_path:
        tensor = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        return tensor, None, 1.0, (0.0, 0.0), (imgsz, imgsz)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_hw = img.shape[:2]
    padded, ratio, pad = letterbox(img, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(np.ascontiguousarray(tensor), 0)
    return tensor, img, ratio, pad, orig_hw


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def nms_detections(raw: np.ndarray, conf_thres: float, iou_thres: float, max_det: int = 300):
    """Decode Ultralytics detect output (1, 4+nc, n) and apply OpenCV NMS."""
    if raw.ndim == 3:
        raw = raw[0]
    pred = raw.T
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    cls_ids = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    keep = conf >= conf_thres
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh[keep])
    conf = conf[keep]
    cls_ids = cls_ids[keep]
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=conf.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )
    if len(idxs) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    idxs = np.array(idxs).reshape(-1)[:max_det]
    dets = np.concatenate(
        [boxes_xyxy[idxs], conf[idxs, None], cls_ids[idxs, None].astype(np.float32)],
        axis=1,
    )
    return dets


def scale_boxes(dets: np.ndarray, ratio: float, pad, orig_hw):
    if dets.size == 0:
        return dets
    pad_x, pad_y = pad
    h, w = orig_hw
    dets = dets.copy()
    dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / ratio
    dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / ratio
    dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, w)
    dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, h)
    return dets


def draw_dets(img: np.ndarray, dets: np.ndarray) -> np.ndarray:
    vis = img.copy()
    for x1, y1, x2, y2, conf, cls_id in dets:
        cls_id = int(cls_id)
        name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, (0, 180, 0), 2)
        cv2.putText(
            vis,
            f"{name} {conf:.2f}",
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def print_session_info(session, verbose: bool):
    print("\nInputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} {inp.type}")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} {out.type}")
    if verbose:
        meta = session.get_modelmeta()
        print(f"Producer: {meta.producer_name}")
        for k, v in (meta.custom_metadata_map or {}).items():
            print(f"  {k}: {v}")


def compare_with_pytorch(pt_path: Path, image_path: str, conf: float, iou: float, imgsz: int, end2end: bool):
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    kwargs = dict(imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    if hasattr(model.model, "end2end"):
        kwargs["end2end"] = end2end
    result = model.predict(image_path, **kwargs)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()[:, None]
    clss = boxes.cls.cpu().numpy()[:, None]
    return np.concatenate([xyxy, confs, clss], axis=1)


def main() -> int:
    args = parse_args()
    try:
        model_path = resolve_weight(args.model, download=False)
    except FileNotFoundError as e:
        print(e)
        return 1
    if model_path.suffix.lower() != ".onnx":
        print(f"Not an ONNX file: {model_path}")
        return 1

    print("=" * 70)
    print("ONNX inference test")
    print("=" * 70)
    print(f"Model: {model_path}")

    available = ort.get_available_providers()
    providers = [p for p in args.providers if p in available] or ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"Providers: {session.get_providers()}")
    print_session_info(session, args.verbose)

    tensor, orig_bgr, ratio, pad, orig_hw = preprocess(args.image, args.imgsz)
    input_name = session.get_inputs()[0].name

    for _ in range(args.warmup):
        session.run(None, {input_name: tensor})

    times = []
    outputs = None
    for _ in range(args.benchmark):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    mean = float(times.mean())

    print("\nBenchmark")
    print("-" * 70)
    print(f"Mean: {mean:.2f} ms   P50: {np.percentile(times, 50):.2f} ms   "
          f"P95: {np.percentile(times, 95):.2f} ms   FPS: {1000 / mean:.2f}")

    raw = outputs[0]
    print(f"\nRaw output shape: {raw.shape}  min={raw.min():.4f}  max={raw.max():.4f}")
    if np.isnan(raw).any() or np.isinf(raw).any():
        print("WARNING: output contains NaN/Inf")
        return 1

    dets = nms_detections(raw, args.conf, args.iou)
    if orig_bgr is not None:
        dets = scale_boxes(dets, ratio, pad, orig_hw)

    print(f"Detections after NMS (conf={args.conf}, iou={args.iou}): {len(dets)}")
    for x1, y1, x2, y2, conf, cls_id in dets:
        cls_id = int(cls_id)
        name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        print(f"  {name:16s}  {conf:.3f}  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    if args.compare_pt and args.image:
        try:
            pt_path = resolve_weight(args.compare_pt, download=False)
        except FileNotFoundError as e:
            print(e)
            return 1
        end2end = "one_to_one" in model_path.stem
        pt_dets = compare_with_pytorch(pt_path, args.image, args.conf, args.iou, args.imgsz, end2end)
        print(f"\nPyTorch detections: {len(pt_dets)}   ONNX detections: {len(dets)}")
        print("Counts should be close; small coordinate diffs are expected after ORT.")

    if args.save and orig_bgr is not None:
        vis = draw_dets(orig_bgr, dets)
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)
        print(f"Wrote {save_path}")

    result_path = model_path.parent / f"{model_path.stem}_benchmark.txt"
    result_path.write_text(
        "\n".join(
            [
                "ONNX benchmark",
                f"Model: {model_path}",
                f"Providers: {session.get_providers()}",
                f"Mean ms: {mean:.2f}",
                f"FPS: {1000 / mean:.2f}",
                f"Raw shape: {raw.shape}",
                f"NMS detections: {len(dets)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
