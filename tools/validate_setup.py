#!/usr/bin/env python3
"""
Quick validation script for NEU-DET data preparation pipeline

Usage:
    python tools/validate_setup.py
"""

from pathlib import Path


def validate():
    print("=" * 60)
    print("NEU-DET Data Preparation Validation")
    print("=" * 60)

    # Check NEU-DET dataset
    neudet = Path('../NEU-DET')
    if not neudet.exists():
        print(f"✗ NEU-DET not found at: {neudet.resolve()}")
        return False

    images = list((neudet / 'images').glob('*.jpg'))
    xmls = list((neudet / 'annotations').glob('*.xml'))

    print(f"\n1. NEU-DET Dataset")
    print(f"   Location: {neudet.resolve()}")
    print(f"   Images:   {len(images)}")
    print(f"   XMLs:     {len(xmls)}")

    if len(images) != len(xmls):
        print(f"   ⚠️  Warning: image/xml count mismatch!")

    # Check if labels directory exists
    labels = neudet / 'labels'
    if labels.exists():
        txt_files = list(labels.glob('*.txt'))
        print(f"\n2. YOLO Labels (already converted)")
        print(f"   Location: {labels}")
        print(f"   Files:    {len(txt_files)}")
    else:
        print(f"\n2. YOLO Labels")
        print(f"   Status:   Not yet converted")
        print(f"   Run:      python tools/converter_neudet.py")

    # Check split files
    datasets = Path('datasets')
    all_list = datasets / 'neudet_yolo.txt'
    train_list = datasets / 'neudet_yolo_train.txt'
    val_list = datasets / 'neudet_yolo_val.txt'

    print(f"\n3. Dataset Split Files")
    if all_list.exists():
        count = len(all_list.read_text().splitlines())
        print(f"   All images:  {all_list} ({count} paths)")
    else:
        print(f"   All images:  Not generated yet")

    if train_list.exists() and val_list.exists():
        train_count = len(train_list.read_text().splitlines())
        val_count = len(val_list.read_text().splitlines())
        print(f"   Train set:   {train_list} ({train_count} paths)")
        print(f"   Val set:     {val_list} ({val_count} paths)")
        print(f"   Split ratio: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
    else:
        print(f"   Train/Val:   Not split yet")
        print(f"   Run:         python tools/split_neudet.py")

    # Check config
    config = Path('configs/neudet.yaml')
    print(f"\n4. YOLO Config")
    if config.exists():
        print(f"   Status: ✓ {config}")
        print(f"   Note:   Update 'path' field to match your NEU-DET location")
    else:
        print(f"   Status: ✗ Not found at {config}")

    print("\n" + "=" * 60)
    print("Next steps:")
    if not labels.exists():
        print("  1. python tools/converter_neudet.py")
        print("  2. python tools/split_neudet.py")
    elif not train_list.exists():
        print("  1. python tools/split_neudet.py")
    else:
        print("  1. Update configs/neudet.yaml 'path' field")
        print("  2. python src/train_yolo.py --data configs/neudet.yaml")
    print("=" * 60)

    return True


if __name__ == '__main__':
    validate()
