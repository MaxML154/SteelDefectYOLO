import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from tqdm import tqdm
from pathlib import Path
import os
from operator import itemgetter

"""
Adapted for NEU-DET flat directory structure.

针对NEU-DET标注质量的4个主要问题：
1. CRAZING（龟裂）- 背景误识别率41%，需要合并邻近框减少碎片化标注
2. INCLUSION（杂质）- 30.9%框小于1000像素，保持原样（小目标检测任务特性）
3. PITTED_SURFACE（点蚀）- 29.9%框超过30k像素，合并重叠框
4. SCRATCHES（划痕）- 79.8%极端长宽比，保持原样（线状缺陷特性）

[flat structure]
- NEU-DET
    - annotations
        - crazing_*.xml
        - ...
    - images
        - crazing_*.jpg
        - ...

[Generated structure]
- NEU-DET
    - annotations_clean
        - crazing_*.xml
        - ...

[Usage]
$ python clean_neudet_labels.py

[Parameters]
- DILATED_WIDTH: 膨胀宽度，用于合并邻近框（针对crazing）
  - 值越大，越容易合并距离较远的框
  - 推荐值：8-12，默认10

- OVERLAP_FRACTION: 重叠阈值，用于合并重叠框
  - 当重叠面积 > OVERLAP_FRACTION * min_box_area 时合并
  - 推荐值：0.3-0.5，默认0.4

- PROCESS_CLASSES: 需要处理的类别
  - crazing: 应用膨胀+重叠合并（解决碎片化标注）
  - pitted_surface: 仅应用重叠合并（解决多框标注同一区域）
"""

ROOT_FOLDER_PATH = r'H:\Study\YoloForSurface\NEU-DET'
ANNOT_FOLDER = 'annotations'
OUTPUT_FOLDER = 'annotations_clean'
LABELS = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# For merging near-by boxes, DILATED_WIDTH ~ 0.5 * gap_between_boxes
DILATED_WIDTH = 10

# If overlapped area > OVERLAP_FRACTION * min_box_area, merge boxes
OVERLAP_FRACTION = 0.4

# Classes that need box merging
# crazing: dilated + overlap (碎片化标注问题)
# pitted_surface: overlap only (多框标注同一区域)
DILATED_CLASSES = ['crazing']  # Apply dilation-based merging
OVERLAP_CLASSES = ['crazing', 'pitted_surface']  # Apply overlap-based merging

# Classes to keep unchanged
PASS_CLASSES = ['inclusion', 'patches', 'rolled-in_scale', 'scratches']


def read_xml(in_fp):
    """ Read content of a xml file.
    Args:
        in_fp (str): xml file path
    Returns:
        w (int): image width
        h (int): image height
        dict_group (dict): key is class name (str),
                           value is list of [xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    tree = ET.parse(in_fp)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    dict_group = {}
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in LABELS and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = [int(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')]
            if cls in dict_group:
                dict_group[cls].append(bb)
            else:
                dict_group[cls] = [bb]
    return w, h, dict_group


def dilate_rectangle(box, width, height):
    """ Dilate the rectangle by DILATED_WIDTH.
    Args:
        box (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        width (int): image width, prevent dilation from exceeding width
        height (int): image height, prevent dilation from exceeding height
    Returns:
        box (list): [xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    xmin = max(0, box[0] - DILATED_WIDTH)
    xmax = min(width, box[1] + DILATED_WIDTH)
    ymin = max(0, box[2] - DILATED_WIDTH)
    ymax = min(height, box[3] + DILATED_WIDTH)

    return [xmin, xmax, ymin, ymax]


def compute_overlapped_area(boxa, boxb, dilated=False, width=0, height=0):
    """
    Args:
        boxa (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        boxb (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        dilated (bool): whether to dilate the boxes
        width (int): image width, prevent dilation from exceeding width
        height (int): image height, prevent dilation from exceeding height
    Returns:
        area (int): overlapped area

    """
    # Scan along x-axis
    list_sort_xmin = sorted([boxa, boxb], key=itemgetter(0))

    # Copy
    box1 = list_sort_xmin[0][:]
    box2 = list_sort_xmin[1][:]

    # Get left-side x1 and right-side x2
    x1 = box2[0]
    x2 = box1[1]
    if x2 < x1:  # No intersection
        return 0

    if dilated:
        box1 = dilate_rectangle(box1, width, height)
        box2 = dilate_rectangle(box2, width, height)

    # Get upper-side y1 and bottom-side y2
    if box2[2] < box1[3]:
        y1, y2 = box2[2], box1[3]
    else:
        y1, y2 = box1[2], box2[3]

    # Calculate intersection area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


def area(box):
    """ Compute the area of the box. """
    width = box[1] - box[0]
    height = box[3] - box[2]
    return width * height


def merge(box1, box2):
    """ Union two rectangles.
    Args:
        box1 (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        box2 (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
    Returns:
        box (list): [xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    x1 = min(box1[0], box2[0])
    x2 = max(box1[1], box2[1])
    y1 = min(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, x2, y1, y2]


def union_boxes(list_boxes, dilated, width, height):
    """ Iteratively union rectangles.

    Args:
        list_boxes (list): list of [xmin (int), xmax(int), ymin (int), ymax (int)]
        dilated (bool): whether to dilate the boxes
        width (int): image width, prevent dilation from exceeding width
        height (int): image height, prevent dilation from exceeding height
    Returns:
        list_boxes (list): list of [xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    # Scan along x-axis
    list_sort_xmin = sorted(list_boxes, key=itemgetter(0))

    n_box = len(list_boxes)
    for i in range(n_box-1):
        if list_sort_xmin[i] is None:
            continue

        # Shallow copy
        box = list_sort_xmin[i][:]

        # The enlarged box may intersect with previous boxes
        list_index_review = []

        # Examine the rest boxes
        for j in range(n_box):
            if list_sort_xmin[j] is None or j == i:
                continue

            overlapped_area = compute_overlapped_area(box,
                list_sort_xmin[j], dilated, width, height)

            min_area = min(area(box), area(list_sort_xmin[j]))
            thresh = 1 if dilated else (OVERLAP_FRACTION * min_area)

            if overlapped_area > thresh:
                # Enlarge box
                box = merge(box, list_sort_xmin[j])

                # Erase box_j
                list_sort_xmin[j] = None

                # Check previous boxes
                for k in list_index_review[::-1]:
                    if list_sort_xmin[k] is None:
                        continue

                    overlapped_area = compute_overlapped_area(box,
                        list_sort_xmin[k], dilated, width, height)
                    min_area = min(area(box), area(list_sort_xmin[k]))
                    thresh = 1 if dilated else (OVERLAP_FRACTION * min_area)

                    if overlapped_area > thresh:
                        # Enlarge box
                        box = merge(box, list_sort_xmin[k])

                        # Erase box_k
                        list_sort_xmin[k] = None
            else:
                list_index_review.append(j)

        # Update
        list_sort_xmin[i] = box[:]

    # Remove non-exist boxes
    list_boxes = [box for box in list_sort_xmin if box is not None]

    return list_boxes


def indent(elem, level=0):
    """ Pretty print with new line and indentation. """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_xml(output_path, filename, width, height, list_bbox):
    """ Write cleaned xml file.

    Args:
        output_path (str): output folder of xml file
        filename (str): file name without extension
        width (int): image width
        height (int): image height
        list_bbox (list): list of [class_name, xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    # Extract folder name from file_name
    folder_name = ''.join([i for i in filename if not i.isdigit()])
    folder_name = folder_name.strip('_')

    root = Element('annotation')
    SubElement(root, 'folder').text = folder_name
    SubElement(root, 'filename').text = filename + '.jpg'
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'NEU-DET'

    # Size
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '1'
    SubElement(root, 'segmented').text = '0'

    # Boxes
    for (class_name, xmin, xmax, ymin, ymax) in list_bbox:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)

    tree = ElementTree(root)
    indent(root)
    xml_filename = os.path.join(output_path, filename+'.xml')
    tree.write(xml_filename)


def clean(xml_files, annot_folder, output_folder):
    stats = {
        'total_files': len(xml_files),
        'boxes_before': 0,
        'boxes_after': 0,
        'boxes_before_by_class': {cls: 0 for cls in LABELS},
        'merged_by_class': {cls: 0 for cls in LABELS}
    }

    for xml_fn in tqdm(xml_files, desc='Cleaning labels'):
        in_fp = os.path.join(annot_folder, xml_fn)
        w, h, dict_group = read_xml(in_fp)

        for class_name in dict_group:
            count = len(dict_group[class_name])
            stats['boxes_before'] += count
            stats['boxes_before_by_class'][class_name] += count

        for class_name in dict_group:
            if class_name in PASS_CLASSES or len(dict_group[class_name]) < 2:
                # Keep unchanged
                continue

            # Determine merge strategy
            if class_name in DILATED_CLASSES:
                dilated = True  # Apply dilation for nearby boxes
            else:
                dilated = False  # Only merge overlapping boxes

            original_count = len(dict_group[class_name])
            list_boxes = union_boxes(dict_group[class_name], dilated, w, h)
            merged_count = original_count - len(list_boxes)

            dict_group[class_name] = list_boxes
            stats['merged_by_class'][class_name] += merged_count

        # Convert dict to list
        list_bbox = []
        for class_name in dict_group:
            for box in dict_group[class_name]:
                list_bbox.append([class_name])
                list_bbox[-1].extend(box)

        # Count cleaned boxes
        stats['boxes_after'] += len(list_bbox)

        # Write xml in pretty format
        write_xml(output_folder, xml_fn.split('.')[0], w, h, list_bbox)

    return stats


def main():
    print("="*70)
    print("NEU-DET Label Cleaning Tool")
    print("="*70)
    print(f"\nInput:  {os.path.join(ROOT_FOLDER_PATH, ANNOT_FOLDER)}")
    print(f"Output: {os.path.join(ROOT_FOLDER_PATH, OUTPUT_FOLDER)}")
    print(f"\nParameters:")
    print(f"  DILATED_WIDTH = {DILATED_WIDTH}")
    print(f"  OVERLAP_FRACTION = {OVERLAP_FRACTION}")
    print(f"  Dilated classes: {DILATED_CLASSES}")
    print(f"  Overlap classes: {OVERLAP_CLASSES}")
    print(f"  Unchanged: {PASS_CLASSES}")
    print("\n" + "="*70)

    annot_folder = os.path.join(ROOT_FOLDER_PATH, ANNOT_FOLDER)
    xml_files = [f for f in os.listdir(annot_folder) if f.endswith('.xml')]

    output_folder = os.path.join(ROOT_FOLDER_PATH, OUTPUT_FOLDER)
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    stats = clean(xml_files, annot_folder, output_folder)

    print("\n" + "="*70)
    print("Cleaning Results")
    print("="*70)
    print(f"Files processed: {stats['total_files']}")
    print(f"Boxes before: {stats['boxes_before']}")
    print(f"Boxes after:  {stats['boxes_after']}")
    print(f"Merged: {stats['boxes_before'] - stats['boxes_after']} ({(stats['boxes_before'] - stats['boxes_after'])/stats['boxes_before']*100:.1f}%)")

    print(f"\nPer-class merge:")
    for cls in LABELS:
        orig = stats['boxes_before_by_class'].get(cls, 0)
        merged = stats['merged_by_class'][cls]
        if merged > 0:
            print(f"  {cls:<20} {merged:>4} merged ({merged/orig*100:.1f}%)" if orig > 0 else f"  {cls:<20} {merged:>4} merged")

    print("\n" + "="*70)
    print("Done. Next steps:")
    print("  1. Convert to YOLO: converter_neudet.py --annot-folder annotations_clean --label-folder labels_clean")
    print("  2. Create symlink: NEU-DET/images_clean -> images")
    print("  3. Update split txt: replace /images/ with /images_clean/")
    print("="*70)


if __name__ == '__main__':
    main()
