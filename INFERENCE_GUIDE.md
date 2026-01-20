# Inference & Evaluation Guide

Complete guide for running inference and evaluation on your trained scaffold safety inspection model.

## Prerequisites

Your trained LoRA checkpoint should be in:
```
./checkpoints/shapellm-7bs-scaffold-scaffold_i3ce-lorascaffold-i3ce_251119_1759/
```

And should contain:
- `adapter_model.bin` - LoRA weights
- `adapter_config.json` - LoRA configuration
- `non_lora_trainables.bin` - Non-LoRA parameters (vision tower, mm_projector)

## Available Scripts

### 1. Quick Test (10 Samples)

**Purpose**: Fast sanity check to verify model is working

```bash
bash scripts/quick_test.sh
```

**What it does**:
- Runs inference on 10 validation samples
- Saves predictions to `./quick_test_results/predictions_quick.json`
- Shows sample Q&A pairs

**Output**:
```
🚀 Quick Test: 10 Validation Samples
========================================
🔄 Loading model from ./checkpoints/...
   Detected LoRA checkpoint
✅ Model loaded with bfloat16 dtype
...
✅ Quick test complete!
📁 Results: ./quick_test_results/predictions_quick.json

📊 Sample predictions:
--- Sample 1: scene_001_qa_0 ---
Q: Are there any missing components? If so, where are they?...
A: Yes, there is a missing platform_f1_b0_15 at floor 1, bay 0...
```

### 2. Single Scene Inference

**Purpose**: Test model on a single point cloud with custom questions

```bash
python scripts/inference_scaffold.py \
  --model_path ./checkpoints/YOUR_CHECKPOINT \
  --model_base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
  --point_file ./playground/data/shapellm/scaffold_sft_color/pcs/scene_001.npy \
  --question "Please assess the overall structural safety of this scaffold." \
  --sample_points_num 10000 \
  --with_color
```

**Parameters**:
- `--model_path`: Your LoRA checkpoint directory
- `--model_base`: Base ShapeLLM model (GAPartNet pretrained)
- `--point_file`: Path to .npy point cloud file
- `--question`: Question to ask the model
- `--sample_points_num`: Number of points to sample (default: 10000)
- `--with_color`: Include RGB color information

**Output**:
```
Loading model from ./checkpoints/...
   Detected LoRA checkpoint
✅ Model loaded with bfloat16 dtype

Loading point cloud: scene_001.npy
✅ Point cloud shape: torch.Size([10000, 6])

Question: Please assess the overall structural safety of this scaffold.
============================================================
Response:
Based on the 3D point cloud analysis, this scaffold has several safety concerns:
1. Missing platform at floor 1, bay 0 (platform_f1_b0_15)
2. Damaged horizontal beam showing bending deformation (damaged_beam_bent_f2_b1_23)
3. Overall structural integrity is compromised and requires immediate attention.
============================================================
```

### 3. Interactive Testing

**Purpose**: Ask multiple questions about the same scene

```bash
python scripts/interactive_test.py \
  --model_path ./checkpoints/YOUR_CHECKPOINT \
  --model_base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
  --point_file ./playground/data/shapellm/scaffold_sft_color/pcs/scene_001.npy \
  --sample_points_num 10000 \
  --with_color
```

**Features**:
- Load point cloud once, ask multiple questions
- Preset questions for common safety checks
- Custom question support
- Interactive CLI interface

**Usage**:
```
🤖 Interactive Scaffold Inspection
======================================================================
Preset questions:
  1. Are there any missing components? If so, where are they?
  2. Please assess the overall structural safety of this scaffold.
  3. Are there any damaged components?
  4. Does this scaffold comply with industrial safety and health standards?
  5. What is the overall structure size of this scaffold?

Commands:
  - Type 1-5: Ask preset question
  - Type your question: Ask custom question
  - Type 'quit' or 'exit': Exit
======================================================================

📝 Your question (or command): 1

❓ Question: Are there any missing components? If so, where are they?
🔄 Thinking...

💬 Response:
Yes, there is a missing platform at floor 1, bay 0 (platform_f1_b0_15)...

----------------------------------------------------------------------
📝 Your question (or command): What about damaged parts?

❓ Question: What about damaged parts?
🔄 Thinking...

💬 Response:
There is a damaged horizontal beam showing bending deformation...
```

### 4. Batch Inference on Full Dataset

**Purpose**: Run inference on entire validation/test set

```bash
python scripts/batch_inference.py \
  --model_path ./checkpoints/YOUR_CHECKPOINT \
  --model_base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
  --data_dir ./playground/data/shapellm/scaffold_sft_color \
  --split val \
  --output ./results/predictions_val.json \
  --sample_points_num 10000 \
  --with_color \
  --max_samples 100  # Optional: limit samples for quick test
```

**Parameters**:
- `--data_dir`: Dataset directory containing instructions_*.json and pcs/
- `--split`: Which split to evaluate (train/val/test)
- `--output`: Where to save predictions JSON
- `--max_samples`: (Optional) Limit number of samples for quick testing

**Output**:
```
📂 Loading annotations from instructions_val.json
   Total annotations: 800

🔄 Loading model from ./checkpoints/...
   Detected LoRA checkpoint
✅ Model loaded with bfloat16 dtype

🚀 Starting inference on 100 samples...
Inference: 100%|████████████| 100/100 [05:23<00:00, 3.23s/it]
✅ Completed 100 predictions

✅ Predictions saved to ./results/predictions_val.json
   Total predictions: 100
```

### 5. Full Evaluation Pipeline

**Purpose**: Complete inference + evaluation with metrics

```bash
bash scripts/evaluate_all.sh
```

**What it does**:
1. Runs batch inference on validation set
2. Runs batch inference on test set
3. Evaluates predictions with metrics (IoU, F1)
4. Generates detailed result reports

**Output structure**:
```
./results/
├── predictions_val.json      # Validation predictions
├── predictions_test.json     # Test predictions
├── eval_results_val.json     # Validation metrics
├── eval_results_test.json    # Test metrics
└── evaluation_summary.txt    # Human-readable summary
```

**Evaluation metrics**:
```json
{
  "overall": {
    "total_samples": 800,
    "avg_iou": 0.847,
    "avg_f1": 0.912
  },
  "by_category": {
    "missing_detection": {
      "count": 320,
      "avg_f1": 0.905
    },
    "damage_detection": {
      "count": 280,
      "avg_f1": 0.918
    },
    "referring_segmentation": {
      "count": 200,
      "avg_iou": 0.847
    }
  }
}
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size or point cloud size
```bash
python scripts/batch_inference.py \
  --sample_points_num 5000 \  # Reduce from 10000
  ...
```

### Issue 2: Checkpoint Not Found

**Solution**: Verify checkpoint path
```bash
ls -la ./checkpoints/YOUR_CHECKPOINT/
# Should show: adapter_model.bin, adapter_config.json, non_lora_trainables.bin
```

### Issue 3: Base Model Not Found

**Solution**: The base model will auto-download from HuggingFace:
- `qizekun/ShapeLLM_7B_gapartnet_v1.0`

Or download manually and specify local path:
```bash
python scripts/batch_inference.py \
  --model_base /path/to/local/base_model \
  ...
```

## Dataset Format

Your validation/test annotations should be in this format:

```json
[
  {
    "id": "scene_001_qa_0",
    "point": "pcs/scene_001.npy",
    "conversations": [
      {
        "from": "human",
        "value": "<point>\nAre there any missing components? If so, where are they?"
      },
      {
        "from": "gpt",
        "value": "Yes, there is a missing platform_f1_b0_15 at floor 1, bay 0."
      }
    ]
  },
  ...
]
```

## Performance Tips

1. **Use GPU**: Inference requires CUDA
2. **Batch processing**: Use batch_inference.py for large datasets
3. **Point cloud size**: 10000 points is a good balance between quality and speed
4. **Color information**: Include color (--with_color) for better results

## Expected Inference Times

On a single GPU (e.g., RTX 3090):
- Single scene: ~3-5 seconds
- 100 scenes: ~5-8 minutes
- Full validation (800 scenes): ~40-60 minutes

## Output Format

All prediction files follow this format:

```json
[
  {
    "id": "scene_001_qa_0",
    "point": "pcs/scene_001.npy",
    "conversations": [
      {
        "from": "human",
        "value": "<point>\nAre there any missing components?"
      },
      {
        "from": "gpt",
        "value": "Yes, there is a missing platform_f1_b0_15..."  // ← Model prediction
      }
    ]
  }
]
```

The model's prediction replaces the ground truth in the "gpt" response, preserving the same JSON structure as the training data.

## Next Steps

1. **Quick validation**: Run `bash scripts/quick_test.sh`
2. **Test single scene**: Try `scripts/inference_scaffold.py` with custom questions
3. **Full evaluation**: Run `bash scripts/evaluate_all.sh` to get complete metrics
4. **Interactive testing**: Use `scripts/interactive_test.py` to explore model capabilities

All scripts now properly use **bfloat16** to match your training configuration (--bf16 True).
