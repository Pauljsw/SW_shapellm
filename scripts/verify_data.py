#!/usr/bin/env python3
"""데이터 품질 검증 스크립트"""

import json
import numpy as np
from pathlib import Path
import sys

def verify_dataset(data_dir):
    """데이터셋 검증"""
    data_dir = Path(data_dir)

    print("🔍 데이터셋 검증 시작...")
    print("=" * 60)

    errors = []
    warnings = []

    # 1. 필수 파일 확인
    print("\n1. 필수 파일 확인:")
    required_files = ['split.json', 'metadata.json', 'instructions_train.json']
    for f in required_files:
        if (data_dir / f).exists():
            print(f"   ✅ {f}")
        else:
            errors.append(f"Missing: {f}")
            print(f"   ❌ {f}")

    # 2. split.json 검증
    print("\n2. Split 검증:")
    try:
        split = json.load(open(data_dir / 'split.json'))
        total = len(split['train']) + len(split['val']) + len(split['test'])
        print(f"   Train: {len(split['train'])} ({len(split['train'])/total*100:.1f}%)")
        print(f"   Val:   {len(split['val'])} ({len(split['val'])/total*100:.1f}%)")
        print(f"   Test:  {len(split['test'])} ({len(split['test'])/total*100:.1f}%)")

        # 중복 확인
        all_ids = split['train'] + split['val'] + split['test']
        if len(all_ids) != len(set(all_ids)):
            errors.append("Duplicate scene IDs in split")
    except Exception as e:
        errors.append(f"Split validation failed: {e}")

    # 3. Point cloud 검증
    print("\n3. Point cloud 검증:")
    pcs_dir = data_dir / 'pcs'
    if pcs_dir.exists():
        npy_files = list(pcs_dir.glob('*.npy'))
        print(f"   Total files: {len(npy_files)}")

        # 샘플 검증
        if npy_files:
            sample = np.load(npy_files[0])
            print(f"   Shape: {sample.shape}")
            print(f"   Dtype: {sample.dtype}")
            print(f"   Range: [{sample.min():.3f}, {sample.max():.3f}]")

            if sample.shape[1] != 3:
                errors.append(f"Point cloud should be (N, 3), got {sample.shape}")
            if sample.dtype != np.float32:
                warnings.append(f"Point cloud should be float32, got {sample.dtype}")
    else:
        errors.append("pcs/ directory not found")

    # 4. Annotations 검증
    print("\n4. Annotations 검증:")
    try:
        train_ann = json.load(open(data_dir / 'instructions_train.json'))
        print(f"   Train annotations: {len(train_ann)}")

        # 첫 번째 샘플 구조 확인
        if train_ann:
            sample = train_ann[0]
            required_keys = ['id', 'point', 'conversations']
            for key in required_keys:
                if key not in sample:
                    errors.append(f"Annotation missing key: {key}")
                else:
                    print(f"   ✅ {key}: {type(sample[key]).__name__}")

            # Conversation 구조 확인
            if 'conversations' in sample and sample['conversations']:
                conv = sample['conversations'][0]
                if 'from' in conv and 'value' in conv:
                    print(f"   ✅ Conversation format correct")
                else:
                    errors.append("Conversation format incorrect")
    except Exception as e:
        errors.append(f"Annotation validation failed: {e}")

    # 5. Meta/Labels 검증
    print("\n5. Meta & Labels 검증:")
    meta_dir = data_dir / 'meta'
    labels_dir = data_dir / 'labels'

    if meta_dir.exists() and labels_dir.exists():
        meta_files = list(meta_dir.glob('*.json'))
        label_files = list(labels_dir.glob('*.json'))
        print(f"   Meta files: {len(meta_files)}")
        print(f"   Label files: {len(label_files)}")

        # 샘플 검증
        if meta_files:
            meta = json.load(open(meta_files[0]))
            if 'norm_params' in meta:
                norm = meta['norm_params']
                print(f"   ✅ Norm params: centroid, scale, Rz_deg")
                print(f"      Scale: {norm.get('scale', 'N/A'):.3f}")
            else:
                warnings.append("Meta missing norm_params")

        if label_files:
            labels = json.load(open(label_files[0]))
            if labels:
                label = labels[0]
                if 'bbox_world' in label and 'bbox_norm' in label:
                    print(f"   ✅ Dual bbox format")
                else:
                    warnings.append("Label missing dual bbox")
    else:
        warnings.append("meta/ or labels/ directory not found")

    # 최종 결과
    print("\n" + "=" * 60)
    if errors:
        print("❌ 검증 실패:")
        for err in errors:
            print(f"   - {err}")
        return False
    elif warnings:
        print("⚠️  경고 사항:")
        for warn in warnings:
            print(f"   - {warn}")
        print("\n✅ 검증 통과 (경고 있음)")
        return True
    else:
        print("✅ 완벽한 데이터셋!")
        return True

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./playground/data/shapellm/scaffold_sft"
    success = verify_dataset(data_dir)
    sys.exit(0 if success else 1)
