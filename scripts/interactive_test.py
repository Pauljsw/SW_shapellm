#!/usr/bin/env python3
"""대화형 추론 테스트 - 하나의 scene에 여러 질문"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import load_pts, process_pts
from llava.conversation import conv_templates


class DataArgs:
    def __init__(self, sample_points_num=10000, with_color=True, occlusion=False):
        self.sample_points_num = sample_points_num
        self.with_color = with_color
        self.occlusion = occlusion


def load_model(model_path, model_base=None):
    print(f"🔄 Loading model from {model_path}...")
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name="shapellm",
        load_8bit=False,
        load_4bit=False,
        device_map="auto"
    )
    print(f"✅ Model loaded")
    return tokenizer, model, processor, context_len


def prepare_point_cloud(point_path, data_args):
    points = load_pts(str(point_path))
    points = process_pts(points, data_args)
    return points


def inference_single(model, tokenizer, point_cloud, question, conv_mode="v1"):
    conv = conv_templates[conv_mode].copy()
    question_full = f"<point>\n{question}"
    conv.append_message(conv.roles[0], question_full)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer(prompt).input_ids
    input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)
    points = point_cloud.cuda().unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            points=points,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    return output


def main():
    parser = argparse.ArgumentParser(description='Interactive testing')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_base', type=str, default='qizekun/ShapeLLM_7B_gapartnet_v1.0')
    parser.add_argument('--point_file', type=str, required=True)
    parser.add_argument('--sample_points_num', type=int, default=10000)
    parser.add_argument('--with_color', action='store_true')
    args = parser.parse_args()

    # Load model
    tokenizer, model, processor, context_len = load_model(
        args.model_path,
        args.model_base
    )

    # Load point cloud
    data_args = DataArgs(
        sample_points_num=args.sample_points_num,
        with_color=args.with_color,
        occlusion=False
    )

    print(f"\n📂 Loading point cloud: {args.point_file}")
    point_cloud = prepare_point_cloud(args.point_file, data_args)
    print(f"✅ Point cloud shape: {point_cloud.shape}")

    # Preset questions
    preset_questions = [
        "Are there any missing components? If so, where are they?",
        "Please assess the overall structural safety of this scaffold.",
        "Are there any damaged components?",
        "Does this scaffold comply with industrial safety and health standards?",
        "What is the overall structure size of this scaffold?",
    ]

    print("\n" + "=" * 70)
    print("🤖 Interactive Scaffold Inspection")
    print("=" * 70)
    print("\nPreset questions:")
    for i, q in enumerate(preset_questions, 1):
        print(f"  {i}. {q}")
    print("\nCommands:")
    print("  - Type 1-5: Ask preset question")
    print("  - Type your question: Ask custom question")
    print("  - Type 'quit' or 'exit': Exit")
    print("=" * 70)

    while True:
        print("\n" + "-" * 70)
        user_input = input("📝 Your question (or command): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        # Check if preset question
        if user_input.isdigit():
            idx = int(user_input)
            if 1 <= idx <= len(preset_questions):
                question = preset_questions[idx - 1]
            else:
                print(f"❌ Invalid preset number. Please choose 1-{len(preset_questions)}")
                continue
        else:
            question = user_input

        if not question:
            continue

        # Inference
        print(f"\n❓ Question: {question}")
        print("🔄 Thinking...")

        try:
            response = inference_single(model, tokenizer, point_cloud, question)
            print(f"\n💬 Response:\n{response}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
