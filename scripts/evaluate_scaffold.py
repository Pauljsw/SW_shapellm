#!/usr/bin/env python3
"""비계 모델 평가 스크립트"""

import argparse
import json
import numpy as np
import torch
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

def parse_bbox_from_text(text: str) -> np.ndarray:
    """텍스트에서 bbox 8개 코너 추출"""
    # [[x,y,z],[x,y,z],...] 형식 찾기
    pattern = r'\[\[([^\]]+)\]\]'
    match = re.search(pattern, text)

    if not match:
        return None

    try:
        # Parse nested list
        bbox_str = '[[' + match.group(1) + ']]'
        # Remove extra spaces and parse
        bbox_str = bbox_str.replace(' ', '')
        coords = eval(bbox_str)  # Safe since we control input

        if len(coords) == 8 and all(len(c) == 3 for c in coords):
            return np.array(coords, dtype=np.float32)
    except:
        pass

    return None

def calculate_iou_3d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """3D IoU 계산 (GAPartNet 방식)"""
    # bbox: (8, 3) - 8 corners

    # Min/max 좌표 추출
    min1 = bbox1.min(axis=0)
    max1 = bbox1.max(axis=0)
    min2 = bbox2.min(axis=0)
    max2 = bbox2.max(axis=0)

    # Intersection 계산
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    inter_vol = np.prod(np.maximum(0, inter_max - inter_min))

    # Union 계산
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union_vol = vol1 + vol2 - inter_vol

    if union_vol < 1e-6:
        return 0.0

    return inter_vol / union_vol

def evaluate_referring_segmentation(predictions: List[Dict], labels_dir: Path, meta_dir: Path):
    """Referring segmentation 평가 (IoU)"""
    print("\n📊 Referring Segmentation 평가")
    print("=" * 60)

    ious = []
    correct_025 = 0  # IoU > 0.25
    correct_050 = 0  # IoU > 0.50

    for pred in tqdm(predictions, desc="Evaluating"):
        if pred.get('task_type') != 'referring_segmentation':
            continue

        scene_id = pred['point'].replace('.npy', '')

        # Ground truth bbox 로드
        label_path = labels_dir / f"{scene_id}_label.json"
        if not label_path.exists():
            continue

        labels = json.load(open(label_path))
        target_instance = pred.get('target_instance_id')

        # 해당 instance 찾기
        gt_bbox = None
        for label in labels:
            if label['instance_id'] == target_instance:
                gt_bbox = np.array(label['bbox_norm'], dtype=np.float32)
                break

        if gt_bbox is None:
            continue

        # Prediction에서 bbox 추출
        response = pred['conversations'][1]['value']  # GPT response
        pred_bbox = parse_bbox_from_text(response)

        if pred_bbox is None:
            ious.append(0.0)
            continue

        # IoU 계산
        iou = calculate_iou_3d(pred_bbox, gt_bbox)
        ious.append(iou)

        if iou > 0.25:
            correct_025 += 1
        if iou > 0.50:
            correct_050 += 1

    if ious:
        print(f"Total samples: {len(ious)}")
        print(f"Mean IoU: {np.mean(ious):.4f}")
        print(f"Median IoU: {np.median(ious):.4f}")
        print(f"Acc@0.25: {correct_025/len(ious)*100:.2f}%")
        print(f"Acc@0.50: {correct_050/len(ious)*100:.2f}%")

        return {
            'mean_iou': float(np.mean(ious)),
            'median_iou': float(np.median(ious)),
            'acc_025': correct_025/len(ious),
            'acc_050': correct_050/len(ious),
        }
    else:
        print("No valid samples found")
        return {}

def evaluate_missing_detection(predictions: List[Dict], labels_dir: Path):
    """누락 감지 평가 (Precision, Recall)"""
    print("\n📊 Missing Detection 평가")
    print("=" * 60)

    tp, fp, fn = 0, 0, 0

    for pred in tqdm(predictions, desc="Evaluating"):
        if 'missing' not in pred.get('task_type', '').lower():
            continue

        scene_id = pred['point'].replace('.npy', '')

        # Ground truth 로드
        label_path = labels_dir / f"{scene_id}_label.json"
        if not label_path.exists():
            continue

        labels = json.load(open(label_path))

        # GT에서 missing 개수 세기
        gt_missing = [l for l in labels if l.get('metadata', {}).get('defect') == 'missing']

        # Prediction 파싱 (단순화: "Yes"/"No" 체크)
        response = pred['conversations'][1]['value'].lower()

        if gt_missing:
            # GT에 누락이 있음
            if 'yes' in response or 'missing' in response:
                tp += 1  # Correct detection
            else:
                fn += 1  # Missed
        else:
            # GT에 누락 없음
            if 'no' in response or '없' in response:
                pass  # True negative (not counted)
            else:
                fp += 1  # False alarm

    total = tp + fp + fn
    if total > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return {'precision': precision, 'recall': recall, 'f1': f1}
    else:
        print("No valid samples found")
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to model predictions JSON')
    parser.add_argument('--data_dir', type=str,
                        default='./playground/data/shapellm/scaffold_sft',
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON path')
    args = parser.parse_args()

    # 데이터 로드
    predictions = json.load(open(args.predictions))
    data_dir = Path(args.data_dir)
    labels_dir = data_dir / 'labels'
    meta_dir = data_dir / 'meta'

    print(f"Loaded {len(predictions)} predictions")

    # 평가
    results = {}

    # 1. Referring segmentation
    results['referring'] = evaluate_referring_segmentation(
        predictions, labels_dir, meta_dir
    )

    # 2. Missing detection
    results['missing'] = evaluate_missing_detection(
        predictions, labels_dir
    )

    # 결과 저장
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
