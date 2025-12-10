#!/usr/bin/env python3

import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

CLASS_MAPPING = {
    'pitcher': 0,
    'hitter': 1,
    'catcher': 2,
    'the_chief_umpire': 3,
    'normal_umpire': 4,
    'center_fielder': 5,
    'runner': 6,
    'second_baseman': 7,
    'short_stop': 8,
    'right_fielder': 9,
    'left_fielder': 10,
    'first_baseman': 11,
    'third_baseman': 12,
    'others': 13,
    'fielder': 14,
}

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def process_single_json(args):
    json_path, output_dir = args
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        resolution = data.get('image', {}).get('resolution', [DEFAULT_WIDTH, DEFAULT_HEIGHT])
        img_width, img_height = resolution[0], resolution[1]
        filename = data.get('image', {}).get('filename', '')
        if not filename:
            filename = Path(json_path).stem + '.jpg'
        txt_filename = Path(filename).stem + '.txt'
        output_path = Path(output_dir) / txt_filename
        yolo_lines = []
        annotations = data.get('annotations', [])
        for ann in annotations:
            ann_class = ann.get('class', '')
            if ann_class not in ['player', 'umpire']:
                continue
            attributes = ann.get('attribute', [])
            position = None
            for attr in attributes:
                if 'position' in attr:
                    position = attr['position']
                    break
            if position is None or position not in CLASS_MAPPING:
                continue
            class_id = CLASS_MAPPING[position]
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                continue
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
            if width <= 0 or height <= 0:
                continue
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        return (json_path, True, len(yolo_lines), None)
    except Exception as e:
        return (json_path, False, 0, str(e))


def get_all_json_files(input_dir):
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def main():
    parser = argparse.ArgumentParser(description='Convert JSON annotations to YOLO format')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for YOLO txt files')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Scanning for JSON files in {args.input_dir}...")
    json_files = get_all_json_files(args.input_dir)
    print(f"Found {len(json_files)} JSON files")
    if not json_files:
        print("No JSON files found!")
        return
    process_args = [(json_path, args.output_dir) for json_path in json_files]
    num_workers = args.workers if args.workers else cpu_count()
    print(f"Using {num_workers} worker processes")
    success_count = 0
    fail_count = 0
    total_annotations = 0
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_json, process_args),
            total=len(json_files),
            desc="Converting annotations"
        ))
    for json_path, success, num_ann, error_msg in results:
        if success:
            success_count += 1
            total_annotations += num_ann
        else:
            fail_count += 1
            if fail_count <= 10:
                print(f"Error processing {json_path}: {error_msg}")
    print(f"\nConversion complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total annotations: {total_annotations}")


if __name__ == '__main__':
    main()
