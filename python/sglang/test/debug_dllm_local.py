"""
Debug script for DLLM empty answer investigation using local Engine.

Usage:
python -m sglang.test.debug_dllm_local \
    --model-path <your-model-path> \
    --dllm-algorithm LowConfidenceFDFO \
    --num-questions 5
"""

import argparse
import ast
import json
import re
import time

import numpy as np

import sglang as sgl
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_debug(args):
    print("=" * 70)
    print("DLLM Empty Answer Debug - Local Engine Mode")
    print("=" * 70)

    # Initialize engine with DLLM algorithm
    print(f"\n[INFO] Initializing Engine with model: {args.model_path}")
    print(f"[INFO] DLLM algorithm: {args.dllm_algorithm}")

    engine_kwargs = {
        "model_path": args.model_path,
        "dllm_algorithm": args.dllm_algorithm,
    }

    if args.tp_size:
        engine_kwargs["tp_size"] = args.tp_size
    if args.dllm_algorithm_config:
        engine_kwargs["dllm_algorithm_config"] = args.dllm_algorithm_config
    if args.trust_remote_code:
        engine_kwargs["trust_remote_code"] = True

    print(f"[INFO] Engine kwargs: {engine_kwargs}")

    llm = sgl.Engine(**engine_kwargs)

    # Load data
    if args.data_path is None:
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.data_path

    lines = list(read_jsonl(filename))

    # Prepare prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    prompts = []

    for i in range(min(num_questions, len(lines))):
        q = get_one_example(lines, i, False)
        questions.append(q)
        labels.append(get_answer_value(lines[i]["answer"]))
        prompts.append(few_shot_examples + q)

    print(f"\n[INFO] Prepared {len(prompts)} prompts")
    print(f"[INFO] First prompt length: {len(prompts[0])} chars")

    # Run generation
    print("\n" + "=" * 70)
    print("Running Generation...")
    print("=" * 70)

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "stop": ["Question", "Assistant:", "<|separator|>"],
    }

    print(f"[INFO] Sampling params: {sampling_params}")

    results = []
    empty_count = 0

    for i, prompt in enumerate(prompts):
        print(f"\n--- Question {i+1}/{len(prompts)} ---")

        tic = time.perf_counter()
        output = llm.generate(prompt, sampling_params)
        latency = time.perf_counter() - tic

        # Extract the generated text
        if isinstance(output, dict):
            text = output.get("text", "")
            meta = output.get("meta_info", {})
        elif hasattr(output, "text"):
            text = output.text
            meta = getattr(output, "meta_info", {})
        else:
            text = str(output)
            meta = {}

        # Debug output
        print(f"[DEBUG] Output type: {type(output)}")
        print(f"[DEBUG] Generated text length: {len(text)}")
        print(f"[DEBUG] Generated text repr: {repr(text[:200]) if text else 'EMPTY'}")

        if hasattr(output, "__dict__"):
            print(f"[DEBUG] Output attrs: {list(output.__dict__.keys())}")

        if meta:
            print(f"[DEBUG] Meta info: {json.dumps(meta, indent=2, default=str)[:500]}")

        pred = get_answer_value(text)
        is_correct = pred == labels[i]
        is_empty = text == "" or text is None
        is_invalid = pred == INVALID

        if is_empty:
            empty_count += 1
            print(f"[ALERT] EMPTY ANSWER!")
            print(f"  Question: {questions[i][:100]}...")
            print(f"  Expected: {labels[i]}")
        elif is_invalid:
            print(f"[WARN] Invalid answer (no numbers found)")
            print(f"  Answer: {text[:200]}")
        else:
            status = "✓" if is_correct else "✗"
            print(f"[{status}] Pred={pred}, Label={labels[i]}")

        print(f"[INFO] Latency: {latency:.2f}s")

        results.append(
            {
                "question": questions[i],
                "label": labels[i],
                "answer": text,
                "pred": pred,
                "is_empty": is_empty,
                "is_correct": is_correct,
                "latency": latency,
                "meta": meta,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    preds = [r["pred"] for r in results]

    accuracy = np.mean([r["is_correct"] for r in results])
    invalid_rate = np.mean([r["pred"] == INVALID for r in results])
    empty_rate = np.mean([r["is_empty"] for r in results])
    avg_latency = np.mean([r["latency"] for r in results])

    print(f"Total questions: {len(results)}")
    print(f"Empty answers: {empty_count} ({100*empty_rate:.1f}%)")
    print(
        f"Invalid answers: {sum(1 for r in results if r['pred'] == INVALID)} ({100*invalid_rate:.1f}%)"
    )
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Avg latency: {avg_latency:.2f}s")

    # Analyze empty answers
    if empty_count > 0:
        print("\n" + "=" * 70)
        print("EMPTY ANSWER ANALYSIS")
        print("=" * 70)

        for i, r in enumerate(results):
            if r["is_empty"]:
                print(f"\n[Empty #{i+1}]")
                print(f"  Question: {r['question'][:150]}...")
                print(f"  Meta: {r['meta']}")

    # Cleanup
    llm.shutdown()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--dllm-algorithm",
        type=str,
        default="LowConfidenceFDFO",
        help="DLLM algorithm name",
    )
    parser.add_argument(
        "--dllm-algorithm-config",
        type=str,
        default=None,
        help="Path to DLLM algorithm config YAML",
    )
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    run_debug(args)
