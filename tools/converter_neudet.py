#!/usr/bin/env python3
"""
NEU-DET XML to YOLO Format Converter

Creates:
  NEU-DET/labels/*.txt       — individual YOLO label files (no image copying)
  datasets/neudet_yolo.txt   — list of absolute image paths for train/val split

Usage:
    python converter_neudet.py --neudet ../../NEU-DET --output ../datasets/neudet_yolo.txt

Author: MaxML154
Date: 2026-07-27
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from tqdm import tqdm

CLASS_TO_ID = {
    'crazing': 0,
    'inclusion': 1,
    'patches': 2,
    'pitted_surface': 3,      # NEU-DET uses 'pitted_surface' not 'pitted'
    'pitted': 3,              # Fallback for compatibility
    'rolled-in_scale': 4,     # NEU-DET uses 'rolled-in_scale' not 'rolled-in'
    'rolled-in': 4,           # Fallback for compatibility
    'scratches': 5,
}


def _convert_xml(xml_path: Path, labels_dir: Path) -> bool:
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_TO_ID:
            continue
        bb = obj.find('bndbox')
        xmin, ymin = int(bb.find('xmin').text), int(bb.find('ymin').text)
        xmax, ymax = int(bb.find('xmax').text), int(bb.find('ymax').text)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    (labels_dir / (xml_path.stem + '.txt')).write_text('\n'.join(lines))
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert NEU-DET XML annotations to YOLO format")
    parser.add_argument('--neudet', default='../../NEU-DET', help='NEU-DET root directory')
    parser.add_argument('--output', default='../datasets/neudet_yolo_cleaned.txt', help='Output image list path')
    parser.add_argument('--annot-folder', default='annotations_cleaned', help='Annotation folder name')
    parser.add_argument('--label-folder', default='labels_clean', help='Label output folder name')
    args = parser.parse_args()

    neudet = Path(args.neudet).resolve()
    labels_dir = neudet / args.label_folder
    labels_dir.mkdir(exist_ok=True)

    xml_files = sorted((neudet / args.annot_folder).glob('*.xml'))
    image_paths = []

    for xml in tqdm(xml_files, desc="Converting"):
        img = neudet / 'images' / (xml.stem + '.jpg')
        if img.exists() and _convert_xml(xml, labels_dir):
            image_paths.append(str(img))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(image_paths))

    print(f"✓ Annotations from: {neudet / args.annot_folder}")
    print(f"✓ Labels written to: {labels_dir}  ({len(image_paths)} files)")
    print(f"✓ Image list:        {out}")


if __name__ == '__main__':
    main()
