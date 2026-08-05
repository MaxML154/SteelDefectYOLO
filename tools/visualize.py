#!/usr/bin/env python3
"""
Visualization Tools for NEU-DET Industrial Defect Detection

Features:
  - Prediction visualization with bounding boxes
  - Ground truth vs prediction comparison
  - Defect heatmaps and distribution analysis
  - Batch visualization for quick inspection
  - Export to images/videos

Usage:
    python visualize.py --model outputs/train_xxx/weights/best.pt --source ../NEU-DET/images
    python visualize.py --model best.pt --source path/to/images --conf 0.5 --save

Author: MaxML154
Date: 2026-07-27
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)


CLASS_NAMES = [
    'crazing', 'inclusion', 'patches',
    'pitted', 'rolled-in', 'scratches'
]

# Color palette for each defect class (BGR format)
CLASS_COLORS = [
    (0, 255, 255),    # crazing - cyan
    (255, 0, 255),    # inclusion - magenta
    (0, 255, 0),      # patches - green
    (255, 255, 0),    # pitted - yellow
    (255, 0, 0),      # rolled-in - blue
    (0, 0, 255),      # scratches - red
]


def load_ground_truth(label_path: Path, img_shape: Tuple[int, int]) -> List[dict]:
    """
    Load YOLO format ground truth labels.

    Args:
        label_path: Path to .txt label file
        img_shape: (height, width) of the image

    Returns:
        List of ground truth boxes with format:
        [{'class': int, 'bbox': [x1, y1, x2, y2], 'conf': 1.0}]
    """
    if not label_path.exists():
        return []

    h, w = img_shape
    gt_boxes = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # Convert normalized YOLO format to pixel coordinates
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            gt_boxes.append({
                'class': cls_id,
                'bbox': [x1, y1, x2, y2],
                'conf': 1.0,  # Ground truth has confidence 1.0
                'label': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
            })

    return gt_boxes


def draw_boxes(image: np.ndarray, boxes: List[dict], color_map=None, thickness: int = 2,
               show_conf: bool = True, show_label: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR)
        boxes: List of boxes with 'class', 'bbox', 'conf', 'label'
        color_map: Optional color mapping for classes
        thickness: Line thickness
        show_conf: Whether to show confidence scores
        show_label: Whether to show class labels

    Returns:
        Annotated image
    """
    img = image.copy()

    for box in boxes:
        cls_id = box['class']
        x1, y1, x2, y2 = box['bbox']
        conf = box.get('conf', 1.0)
        label = box.get('label', CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}')

        # Get color
        if color_map and cls_id in color_map:
            color = color_map[cls_id]
        elif cls_id < len(CLASS_COLORS):
            color = CLASS_COLORS[cls_id]
        else:
            color = (128, 128, 128)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        if show_label or show_conf:
            if show_conf and conf < 1.0:
                text = f"{label} {conf:.2f}"
            else:
                text = label

            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def visualize_prediction(image_path: Path, model: YOLO, conf_threshold: float = 0.25,
                        show_gt: bool = False, labels_dir: Path = None) -> np.ndarray:
    """
    Visualize model prediction on a single image.

    Args:
        image_path: Path to input image
        model: Trained YOLO model
        conf_threshold: Confidence threshold for predictions
        show_gt: Whether to overlay ground truth boxes
        labels_dir: Directory containing ground truth labels

    Returns:
        Annotated image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Run prediction
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)

    # Extract predictions
    pred_boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            pred_boxes.append({
                'class': cls_id,
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'label': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
            })

    # Draw predictions
    img_pred = draw_boxes(img, pred_boxes, show_conf=True, show_label=True, thickness=2)

    # Optionally draw ground truth
    if show_gt and labels_dir:
        label_path = labels_dir / (image_path.stem + '.txt')
        gt_boxes = load_ground_truth(label_path, img.shape[:2])

        # Draw GT with dashed style (approximate with thin lines)
        for box in gt_boxes:
            x1, y1, x2, y2 = box['bbox']
            color = (0, 255, 0)  # Green for GT
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 1)

            # Add "GT" label
            label = f"GT: {box['label']}"
            cv2.putText(img_pred, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img_pred


def visualize_comparison(image_path: Path, model: YOLO, labels_dir: Path,
                        conf_threshold: float = 0.25) -> np.ndarray:
    """
    Create side-by-side comparison of ground truth and prediction.

    Args:
        image_path: Path to input image
        model: Trained YOLO model
        labels_dir: Directory containing ground truth labels
        conf_threshold: Confidence threshold

    Returns:
        Side-by-side comparison image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]

    # Get ground truth
    label_path = labels_dir / (image_path.stem + '.txt')
    gt_boxes = load_ground_truth(label_path, (h, w))

    # Get predictions
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
    pred_boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            pred_boxes.append({
                'class': cls_id,
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'label': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
            })

    # Draw GT and prediction separately
    img_gt = draw_boxes(img.copy(), gt_boxes, show_conf=False, show_label=True, thickness=2)
    img_pred = draw_boxes(img.copy(), pred_boxes, show_conf=True, show_label=True, thickness=2)

    # Add titles
    cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_pred, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Concatenate horizontally
    comparison = np.hstack([img_gt, img_pred])

    return comparison


def create_grid_visualization(images: List[np.ndarray], grid_size: Tuple[int, int] = None,
                              titles: List[str] = None) -> np.ndarray:
    """
    Create a grid of images for batch visualization.

    Args:
        images: List of images to display
        grid_size: (rows, cols) for grid layout. Auto-computed if None
        titles: Optional titles for each image

    Returns:
        Grid visualization image
    """
    n = len(images)
    if n == 0:
        return None

    # Auto-compute grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid_size = (rows, cols)

    rows, cols = grid_size

    # Resize all images to same size
    target_h, target_w = images[0].shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))
        resized.append(img)

    # Create grid
    grid_rows = []
    for i in range(rows):
        row_imgs = []
        for j in range(cols):
            idx = i * cols + j
            if idx < len(resized):
                img = resized[idx].copy()

                # Add title if provided
                if titles and idx < len(titles):
                    cv2.putText(img, titles[idx], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                row_imgs.append(img)
            else:
                # Fill empty slots with black
                row_imgs.append(np.zeros_like(resized[0]))

        grid_rows.append(np.hstack(row_imgs))

    grid = np.vstack(grid_rows)

    return grid


def visualize_batch(source_dir: Path, model: YOLO, output_dir: Path = None,
                   conf_threshold: float = 0.25, max_images: int = 16,
                   save_individual: bool = True, show_gt: bool = False,
                   labels_dir: Path = None):
    """
    Visualize predictions on a batch of images.

    Args:
        source_dir: Directory containing input images
        model: Trained YOLO model
        output_dir: Output directory for visualizations
        conf_threshold: Confidence threshold
        max_images: Maximum number of images to process
        save_individual: Whether to save individual annotated images
        show_gt: Whether to overlay ground truth
        labels_dir: Directory containing ground truth labels
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")

    # Get image files
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_exts:
        image_files.extend(source_dir.glob(ext))

    image_files = sorted(image_files)[:max_images]

    if len(image_files) == 0:
        print(f"No images found in {source_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('outputs/visualize')
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    annotated_images = []
    titles = []

    for img_path in tqdm(image_files, desc="Visualizing"):
        try:
            img_annotated = visualize_prediction(
                img_path, model, conf_threshold,
                show_gt=show_gt, labels_dir=labels_dir
            )

            annotated_images.append(img_annotated)
            titles.append(img_path.stem)

            # Save individual image
            if save_individual:
                output_path = output_dir / f"{img_path.stem}_annotated.jpg"
                cv2.imwrite(str(output_path), img_annotated)

        except Exception as e:
            print(f"✗ Failed to process {img_path.name}: {e}")

    # Create grid visualization
    if len(annotated_images) > 1:
        grid = create_grid_visualization(annotated_images, titles=titles)
        grid_path = output_dir / 'grid_visualization.jpg'
        cv2.imwrite(str(grid_path), grid)
        print(f"\n✓ Grid visualization saved to: {grid_path}")

    print(f"✓ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualization tools for NEU-DET industrial defect detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize predictions on a directory
    python visualize.py --model outputs/train_xxx/weights/best.pt --source ../NEU-DET/images

    # Show ground truth overlays
    python visualize.py --model best.pt --source images/ --show-gt --labels ../NEU-DET/labels

    # Custom confidence threshold
    python visualize.py --model best.pt --source images/ --conf 0.5

    # Process limited number of images
    python visualize.py --model best.pt --source images/ --max-images 50

    # Save to custom output directory
    python visualize.py --model best.pt --source images/ --output results/viz
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing images'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: outputs/visualize)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--max-images',
        type=int,
        default=16,
        help='Maximum number of images to process (default: 16)'
    )

    parser.add_argument(
        '--show-gt',
        action='store_true',
        help='Overlay ground truth boxes (requires --labels)'
    )

    parser.add_argument(
        '--labels',
        type=str,
        help='Directory containing ground truth labels (YOLO format)'
    )

    parser.add_argument(
        '--no-save-individual',
        action='store_true',
        help='Do not save individual annotated images'
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Validate labels directory if show-gt is enabled
    labels_dir = None
    if args.show_gt:
        if not args.labels:
            print("✗ Error: --labels required when using --show-gt")
            sys.exit(1)
        labels_dir = Path(args.labels)
        if not labels_dir.exists():
            print(f"✗ Error: Labels directory not found: {labels_dir}")
            sys.exit(1)

    # Run visualization
    try:
        visualize_batch(
            source_dir=Path(args.source),
            model=model,
            output_dir=Path(args.output) if args.output else None,
            conf_threshold=args.conf,
            max_images=args.max_images,
            save_individual=not args.no_save_individual,
            show_gt=args.show_gt,
            labels_dir=labels_dir
        )

        print("\n✓ Visualization completed successfully!")

    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
