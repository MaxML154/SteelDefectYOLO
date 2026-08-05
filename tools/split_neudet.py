#!/usr/bin/env python3
"""
NEU-DET Dataset Splitter (stratified, no file copying)

Reads neudet_yolo.txt and writes:
  neudet_yolo_train.txt  — 80% image paths
  neudet_yolo_val.txt    — 20% image paths

Stratification uses the class prefix in filenames (e.g. crazing_1.jpg → 'crazing').

Usage:
    python split_neudet.py --input ../datasets/neudet_yolo.txt
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val split for NEU-DET")
    parser.add_argument('--input', default='../datasets/neudet_yolo.txt', help='Image list from converter_neudet.py')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input)
    paths = src.read_text().splitlines()

    # Extract class label from filename prefix (e.g. "crazing_1.jpg" → "crazing")
    labels = [Path(p).stem.rsplit('_', 1)[0] for p in paths]

    train_paths, val_paths = train_test_split(
        paths,
        test_size=1 - args.train_ratio,
        stratify=labels,
        random_state=args.seed,
    )

    train_file = src.with_name(src.stem + '_train.txt')
    val_file   = src.with_name(src.stem + '_val.txt')

    train_file.write_text('\n'.join(train_paths))
    val_file.write_text('\n'.join(val_paths))

    print(f"✓ Train: {len(train_paths)} samples → {train_file}")
    print(f"✓ Val:   {len(val_paths)} samples → {val_file}")


if __name__ == '__main__':
    main()
