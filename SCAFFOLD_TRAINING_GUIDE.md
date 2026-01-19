# 🏗️ ShapeLLM 비계 안전 검사 훈련 가이드

## 📋 목차
1. [환경 준비](#1-환경-준비)
2. [데이터 생성](#2-데이터-생성)
3. [훈련](#3-훈련)
4. [평가](#4-평가)
5. [추론 테스트](#5-추론-테스트)
6. [FAQ](#6-faq)

---

## 1. 환경 준비

### 1.1 필수 요구사항

- **GPU**: NVIDIA GPU 24GB+ (A100, RTX 3090, RTX 4090)
- **Python**: 3.9+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+

### 1.2 체크리스트

```bash
# 환경 확인
python --version        # Python 3.9+
nvidia-smi              # GPU 확인
pip list | grep torch   # PyTorch 확인
```

### 1.3 ReCon++ Checkpoint 다운로드

```bash
mkdir -p checkpoints/recon
cd checkpoints/recon

# Option 1: wget
wget https://huggingface.co/qizekun/ReCon/resolve/main/large.pth

# Option 2: 수동 다운로드
# https://huggingface.co/qizekun/ReCon/blob/main/large.pth

cd ../..
```

---

## 2. 데이터 생성

### 2.1 빠른 테스트 (50 scenes, ~3분)

```bash
python tools/generate_scaffold_data_improved.py \
  --num_scenes 50 \
  --output_dir ./playground/data/shapellm/scaffold_sft \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --random_seed 42
```

### 2.2 실제 훈련용 (1000 scenes, ~1시간)

```bash
python tools/generate_scaffold_data_improved.py \
  --num_scenes 1000 \
  --output_dir ./playground/data/shapellm/scaffold_sft \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --random_seed 42
```

### 2.3 데이터 검증

```bash
python scripts/verify_data.py ./playground/data/shapellm/scaffold_sft
```

**예상 출력**:
```
✅ 완벽한 데이터셋!
```

### 2.4 생성된 파일 구조

```
playground/data/shapellm/scaffold_sft/
├── pcs/                      # Point clouds (N, 3) float32
│   ├── scaffold_00000.npy
│   └── ...
├── meta/                     # Normalization params
│   ├── scaffold_00000_meta.json
│   └── ...
├── labels/                   # Bbox labels (world + norm)
│   ├── scaffold_00000_label.json
│   └── ...
├── split.json               # Train/val/test split
├── instructions_train.json  # ~2800 annotations
├── instructions_val.json    # ~350 annotations
├── instructions_test.json   # ~350 annotations
└── metadata.json            # Dataset statistics
```

---

## 3. 훈련

### 3.1 훈련 시작

```bash
# Multi-GPU (권장)
bash scripts/finetune_lora_test.sh

# 단일 GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/finetune_lora_test.sh
```

### 3.2 훈련 설정

현재 설정 (`scripts/finetune_lora_test.sh`):

```bash
# LoRA 설정
--lora_enable True
--lora_r 128              # LoRA rank
--lora_alpha 256          # Scaling factor

# 학습률
--learning_rate 2e-4      # LLM
--mm_projector_lr 2e-5    # Vision projector

# 훈련 설정
--num_train_epochs 5
--per_device_train_batch_size 4
--gradient_accumulation_steps 4

# Point cloud 설정
--sample_points_num 10000
--with_color False        # xyz only
```

### 3.3 훈련 모니터링

```bash
# 실시간 로그
tail -f checkpoints/shapellm-7bs-scaffold2-scaffold-lorascaffold-test2/training_log.txt

# Loss 확인
grep "loss" checkpoints/*/training_log.txt | tail -20

# GPU 사용률
watch -n 1 nvidia-smi
```

### 3.4 예상 훈련 시간

| GPU | Batch Size | 시간/Epoch | 총 시간 (5 epochs) |
|-----|------------|-----------|-------------------|
| A100 1개 | 16 | ~3시간 | **~15시간** |
| A100 4개 | 64 | ~50분 | **~4시간** |
| V100 1개 | 8 | ~5시간 | **~25시간** |

### 3.5 체크포인트

```bash
# 저장 위치
checkpoints/shapellm-7bs-scaffold2-scaffold-lorascaffold-test2/
├── checkpoint-5000/
├── checkpoint-10000/
└── ...

# 최종 모델 사용
--model_path ./checkpoints/.../checkpoint-XXXX
```

---

## 4. 평가

### 4.1 모델 추론 (테스트셋)

```bash
# TODO: Batch inference script
# python scripts/batch_inference.py \
#   --model_path ./checkpoints/.../checkpoint-XXXX \
#   --test_data ./playground/data/shapellm/scaffold_sft/instructions_test.json \
#   --output predictions.json
```

### 4.2 평가 실행

```bash
python scripts/evaluate_scaffold.py \
  --predictions predictions.json \
  --data_dir ./playground/data/shapellm/scaffold_sft \
  --output evaluation_results.json
```

### 4.3 평가 지표

**Referring Segmentation**:
- Mean IoU
- Median IoU
- Acc@0.25 (IoU > 0.25)
- Acc@0.50 (IoU > 0.50)

**Missing Detection**:
- Precision
- Recall
- F1 Score

---

## 5. 추론 테스트

### 5.1 단일 Scene 테스트

```bash
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/shapellm-7bs-scaffold2-scaffold-lorascaffold-test2/checkpoint-10000 \
  --point_file ./playground/data/shapellm/scaffold_sft/pcs/scaffold_00000.npy \
  --question "이 비계의 안전성을 평가해주세요."
```

### 5.2 다양한 질문 예시

```bash
# Referring
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/.../checkpoint-XXXX \
  --point_file scaffold_00000.npy \
  --question "발판의 위치를 알려주세요."

# Missing detection
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/.../checkpoint-XXXX \
  --point_file scaffold_00000.npy \
  --question "누락된 부재가 있나요?"

# Safety assessment
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/.../checkpoint-XXXX \
  --point_file scaffold_00000.npy \
  --question "구조적 안전성을 평가해주세요."
```

---

## 6. FAQ

### Q1. OOM (Out of Memory) 에러 발생

**해결**:
```bash
# scripts/finetune_lora_test.sh 수정
--per_device_train_batch_size 2  # 4 → 2
--gradient_accumulation_steps 8  # 4 → 8
--sample_points_num 8000         # 10000 → 8000
```

### Q2. Loss가 감소하지 않음

**원인**: Learning rate 부적절

**해결**:
```bash
--learning_rate 5e-4  # 2e-4 → 5e-4 (증가)
# 또는
--learning_rate 1e-4  # 2e-4 → 1e-4 (감소)
```

### Q3. 훈련 중간에 멈춤

**원인**: DeepSpeed 설정 문제

**해결**:
```bash
# 단일 GPU로 먼저 테스트
CUDA_VISIBLE_DEVICES=0 bash scripts/finetune_lora_test.sh
```

### Q4. ReCon++ checkpoint 다운로드 실패

**대안**:
1. 브라우저로 직접 다운로드: https://huggingface.co/qizekun/ReCon/tree/main
2. `git lfs` 사용:
   ```bash
   git lfs install
   git clone https://huggingface.co/qizekun/ReCon
   cp ReCon/large.pth ./checkpoints/recon/
   ```

### Q5. 예상 성능은?

| Metric | 목표 | 우수 | 최고 |
|--------|------|------|------|
| Acc@0.25 | 50% | 65% | 75% |
| Missing F1 | 60% | 70% | 80% |

---

## 7. 전체 워크플로우 (요약)

```bash
# 1. 데이터 생성 (1시간)
python tools/generate_scaffold_data_improved.py --num_scenes 1000

# 2. 검증
python scripts/verify_data.py

# 3. 훈련 (12시간)
bash scripts/finetune_lora_test.sh

# 4. 추론 테스트
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/.../checkpoint-XXXX \
  --point_file ./playground/data/shapellm/scaffold_sft/pcs/scaffold_00000.npy \
  --question "이 비계의 안전성을 평가해주세요."

# 5. 평가 (TODO)
# python scripts/evaluate_scaffold.py --predictions predictions.json
```

---

## 8. 핵심 원리

### 왜 이 학습이 잘 되는가?

**1. Transfer Learning 3단계**:
```
OpenShape → GAPartNet → Scaffold (우리)
(8M shapes) → (27K parts) → (1K scaffolds)
```

**2. Consistent 합성 데이터**:
- Normalization 일치 (centroid, scale)
- Bbox 정확 (자동 생성)
- Annotation 일관성 (수동 라벨링 없음)

**3. LoRA 효율성**:
- 학습 파라미터: ~100M (1.4%)
- 기존 지식 보존 (GAPartNet bbox 능력)
- 새 지식 추가 (비계 도메인)

**4. Multi-Task Learning**:
- 5개 태스크 상호 보완
- Curriculum learning (easy → hard)

---

## 9. 참고 자료

- ShapeLLM 논문: https://arxiv.org/abs/2402.17766
- ReCon++ 논문: https://arxiv.org/abs/2311.07688
- GAPartNet 데이터셋: https://pku-epic.github.io/GAPartNet/

---

**문제 발생 시**: GitHub Issues에 보고 또는 문서 확인
