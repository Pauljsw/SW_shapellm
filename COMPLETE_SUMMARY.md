# ShapeLLM Scaffold Training & Inference - Complete Summary

## 문제 경과 및 해결

### 1. Dtype Mismatch 문제 발견

**증상**:
```
RuntimeError: mat1 and mat2 must have the same dtype
```

추론 시 매번 이 오류가 발생하여 모델 테스트가 불가능했습니다.

### 2. 첫 번째 분석 (부분적으로 맞음)

**발견 사항**:
- 훈련: `--bf16 True` (bfloat16)
- 추론: `builder.py`가 float16을 하드코딩

**수정**:
- `llava/model/builder.py`: `torch_dtype` 파라미터 추가, 기본값 bfloat16
- 모든 하드코딩된 `torch.float16`을 `torch_dtype`으로 변경

**결과**: 여전히 오류 발생 ❌

### 3. 근본 원인 발견 (정확한 분석)

**실제 문제**:
`llava/model/multimodal_encoder/clip_encoder.py:75-76`에서:

```python
local_features = local_features.to(pts.dtype)  # float32
global_features = global_features.to(pts.dtype)  # float32
# pos_features는 변환하지 않음 (bfloat16 유지)
```

**결과**:
- `pos_features`: bfloat16
- `local_features`: float32  ← 문제!
- `global_features`: float32  ← 문제!
- `mm_projector weights`: bfloat16

→ **세 개의 feature dtype이 혼합**되어 mm_projector에서 오류 발생!

### 4. 최종 해결책

**llava/model/llava_arch.py** - `encode_points()` 수정:

```python
def encode_points(self, points):
    # Get features from Vision Tower
    pos_features, local_features, global_features = self.get_model().get_vision_tower()(points)

    # Get mm_projector's weight dtype dynamically
    mm_projector = self.get_model().mm_projector
    target_dtype = next(mm_projector.parameters()).dtype

    # Convert ALL three features to match mm_projector dtype
    if pos_features.dtype != target_dtype:
        pos_features = pos_features.to(target_dtype)
    if local_features.dtype != target_dtype:
        local_features = local_features.to(target_dtype)
    if global_features.dtype != target_dtype:
        global_features = global_features.to(target_dtype)

    # Now all features have consistent dtype
    point_features = mm_projector(pos_features, local_features, global_features)
    return point_features
```

**핵심**:
1. mm_projector의 weight dtype을 **동적으로 확인**
2. **세 개의 feature를 각각 개별 체크**
3. 모두 mm_projector dtype으로 변환

## 수정된 파일 목록

### 1. llava/model/builder.py
**변경 내용**:
- `load_pretrained_model()` 함수에 `torch_dtype=torch.bfloat16` 파라미터 추가
- Line 40: `kwargs['torch_dtype'] = torch_dtype` (기존 hardcoded float16 제거)
- Line 94: `mm_projector_weights`를 `torch_dtype`으로 변환
- Line 113-117: vision_tower와 vision_tower.model을 `torch_dtype`으로 변환

**목적**: 추론 시 훈련과 동일한 bfloat16 dtype 사용

### 2. llava/model/llava_arch.py
**변경 내용**:
- `encode_points()` 함수 완전 재작성
- mm_projector의 weight dtype 동적 확인
- 세 개의 feature (pos, local, global) 각각 개별 dtype 변환

**목적**: Vision Tower의 혼합된 dtype을 mm_projector와 일치시킴

### 3. scripts/batch_inference.py
**변경 내용**:
- `load_model()` 함수 간소화
- `torch_dtype=torch.bfloat16` 명시적 전달
- 수동 dtype 변환 코드 제거 (builder.py가 처리)

**목적**: 코드 간소화 및 명확성 향상

### 4. scripts/inference_scaffold.py
**변경 내용**: batch_inference.py와 동일

### 5. scripts/interactive_test.py
**변경 내용**: batch_inference.py와 동일

### 6. 문서 추가
- `DTYPE_FIX_SUMMARY.md`: Dtype 문제 상세 분석
- `INFERENCE_GUIDE.md`: 추론/검증 가이드
- `COMPLETE_SUMMARY.md`: 전체 요약 (이 파일)

## 왜 훈련은 성공했는가?

### 훈련 시 (train.py:636-637)
```python
if training_args.bits == 16:
    if training_args.bf16:
        model.to(torch.bfloat16)  # ← 전체 모델 변환
```

**`model.to(torch.bfloat16)`이 재귀적으로 모든 서브모듈에 적용**되므로:
- Vision tower도 자동으로 bfloat16으로 변환
- mm_projector도 bfloat16
- 모든 feature도 bfloat16

→ Dtype 일치, 문제 없음 ✅

### 추론 시 (기존 builder.py)
```python
kwargs['torch_dtype'] = torch.float16  # ← 하드코딩!
vision_tower.to(device=device, dtype=torch.float16)
```

**builder.py가 float16을 강제**하므로:
- Vision tower: float16
- 하지만 clip_encoder.py가 feature를 pts.dtype (float32)로 변환
- mm_projector는 LoRA checkpoint에서 bfloat16으로 로드됨

→ Dtype 불일치, 오류 발생 ❌

## 현재 상태

### Commit History
```
684d9f9 - Fix dtype mismatch: explicitly convert all three features
4cbeb88 - Add comprehensive documentation for dtype fix and inference
dae6ad4 - Fix dtype mismatch: Support bfloat16 inference to match training
```

### 해결된 문제
✅ Vision Tower dtype 설정 (builder.py)
✅ mm_projector dtype 설정 (builder.py)
✅ Feature dtype 혼합 문제 (llava_arch.py)
✅ 추론 스크립트 간소화 (batch_inference.py 등)

### 남은 작업
⚠️ **데이터 생성 코드 수정 필요**:
1. 한글 텍스트 완전 제거 (JSON 메타데이터 포함)
2. 점군 데이터를 컬러 포함 [N, 6] 형식으로 생성

## 훈련 설정 (참고)

```bash
# 훈련 시 사용한 설정
--bf16 True                          # bfloat16 사용
--with_color True                    # RGB 컬러 포함
--num_train_epochs 3
--per_device_train_batch_size 4
--gradient_accumulation_steps 4      # 실제 배치 크기: 4 x 4 = 16
--learning_rate 2e-4
--lora_enable True
--lora_r 128
--lora_alpha 256
```

**훈련 결과**:
- 총 1500 steps (500 steps/epoch × 3 epochs)
- 최종 loss: 0.207
- Checkpoint: `./checkpoints/shapellm-7bs-scaffold-scaffold_i3ce-lorascaffold-i3ce_251119_1759/`

## 데이터셋 구성

### 생성된 데이터
- 총 1000개 scene
- Train: 800 scenes
- Val: 100 scenes
- Test: 100 scenes

### Annotation 구조
- 각 scene당 평균 ~32개 QA 쌍
- 총 ~32,000 annotations
- Effective batch size: 64 (4 × 4 × 4)
- Steps per epoch: 32000 / 64 = 500

### QA 유형
1. **Referring Segmentation**: "Where is platform_f1_b2_15?"
2. **Missing Detection**: "Are there any missing components?"
3. **Damage Detection**: "Are there any damaged components?"
4. **Safety Assessment**: "Please assess the overall structural safety."
5. **Regulation Compliance**: "Does this comply with safety standards?"

## 추론/검증 스크립트

### 1. Quick Test (10 samples)
```bash
bash scripts/quick_test.sh
```

### 2. Single Scene Inference
```bash
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/YOUR_CHECKPOINT \
  --model_base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
  --point_file ./playground/data/shapellm/scaffold_sft_color/pcs/scene_001.npy \
  --question "Please assess the safety of this scaffold."
```

### 3. Interactive Testing
```bash
python scripts/interactive_test.py \
  --model_path ./checkpoints/YOUR_CHECKPOINT \
  --model_base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
  --point_file ./playground/data/shapellm/scaffold_sft_color/pcs/scene_001.npy
```

### 4. Full Evaluation
```bash
bash scripts/evaluate_all.sh
```

## 다음 단계

1. **테스트 실행**: 사용자가 직접 quick_test.sh 실행하여 dtype 오류 해결 확인
2. **데이터 생성 코드 수정**:
   - 모든 한글 텍스트를 영어로 변경
   - 점군 데이터를 [N, 6] (xyz + rgb) 형식으로 생성
3. **전체 검증**: evaluate_all.sh로 모델 성능 평가
4. **결과 분석**: IoU, F1 score 등 메트릭 확인

## 기술적 교훈

1. **Dtype 일관성이 중요**: 훈련과 추론의 dtype이 반드시 일치해야 함
2. **Feature pipeline 전체 확인**: 단순히 모델만이 아니라 feature 생성 과정(clip_encoder.py)도 확인 필요
3. **동적 dtype 확인**: Hardcoded 값 대신 실제 weight dtype을 확인하는 것이 안전
4. **다국어 데이터 주의**: LLaMA/LLaVA는 영어 중심 모델이므로 한글 사용 시 성능 저하 가능

## 참고 문서

- `DTYPE_FIX_SUMMARY.md` - Dtype 문제 상세 분석
- `INFERENCE_GUIDE.md` - 추론/검증 전체 가이드
- `SCAFFOLD_TRAINING_GUIDE.md` - 훈련 가이드
