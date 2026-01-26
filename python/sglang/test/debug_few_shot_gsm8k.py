"""
Debug version of few-shot GSM-8K evaluation to investigate empty answers.

This script adds detailed diagnostics to understand why some answers are empty.

Usage:
python3 -m sglang.test.debug_few_shot_gsm8k --num-questions 10
"""

import argparse
import ast
import json
import re
import time

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

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


def analyze_empty_answer(state, question, label, idx):
    """Detailed analysis of an empty/invalid answer."""
    print("\n" + "=" * 80)
    print(f"[ANALYSIS] Empty Answer at Index {idx}")
    print("=" * 80)

    answer = state.get("answer", None)

    print(f"\n1. ANSWER CONTENT:")
    print(f"   - Raw answer: {repr(answer)}")
    print(f"   - Answer length: {len(answer) if answer else 0}")
    print(f"   - Answer type: {type(answer)}")

    if answer:
        print(f"   - First 50 chars: {repr(answer[:50])}")
        print(f"   - Last 50 chars: {repr(answer[-50:])}")
        # Check for special characters
        special_chars = [c for c in answer if ord(c) < 32 or ord(c) > 126]
        if special_chars:
            print(
                f"   - Special chars found: {[hex(ord(c)) for c in special_chars[:10]]}"
            )

    print(f"\n2. QUESTION (truncated):")
    print(f"   {question[:200]}...")

    print(f"\n3. EXPECTED LABEL: {label}")

    print(f"\n4. META INFO:")
    try:
        meta = state.get_meta_info("answer")
        print(f"   - completion_tokens: {meta.get('completion_tokens', 'N/A')}")
        print(f"   - prompt_tokens: {meta.get('prompt_tokens', 'N/A')}")
        print(f"   - finish_reason: {meta.get('finish_reason', 'N/A')}")
        print(f"   - Full meta: {json.dumps(meta, indent=4, default=str)}")
    except Exception as e:
        print(f"   - Error getting meta info: {e}")

    print(f"\n5. STATE INTERNALS:")
    try:
        # Try to access internal state
        if hasattr(state, "text"):
            print(
                f"   - state.text(): {repr(state.text()[:500]) if state.text() else 'None'}..."
            )
        if hasattr(state, "messages"):
            print(f"   - state.messages: {state.messages}")
        if hasattr(state, "_variables"):
            print(
                f"   - state._variables keys: {list(state._variables.keys()) if state._variables else 'None'}"
            )
    except Exception as e:
        print(f"   - Error accessing state internals: {e}")

    print("=" * 80 + "\n")


def run_eval(args):
    # Select backend
    backend = RuntimeEndpoint(f"{args.host}:{args.port}")
    set_default_backend(backend)

    print("\n" + "=" * 80)
    print("DEBUG MODE: Investigating Empty Answers in DLLM")
    print("=" * 80)

    # Try to get server info
    print("\n[INFO] Checking server status...")
    try:
        import requests

        resp = requests.get(f"{args.host}:{args.port}/get_model_info", timeout=5)
        if resp.status_code == 200:
            model_info = resp.json()
            print(f"[INFO] Model: {model_info.get('model_path', 'Unknown')}")
            print(f"[INFO] Server info: {json.dumps(model_info, indent=2)}")
    except Exception as e:
        print(f"[WARN] Could not get server info: {e}")

    if args.data_path is None:
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.data_path

    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer",
            max_tokens=args.max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    print(
        f"\n[INFO] Running {num_questions} questions with {args.parallel} parallel threads..."
    )
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature if hasattr(args, "temperature") else 0,
        num_threads=args.parallel,
        progress_bar=True,
        return_logprob=getattr(args, "return_logprob", None),
        logprob_start_len=getattr(args, "logprob_start_len", None),
    )
    latency = time.perf_counter() - tic

    # Analyze results
    preds = []
    empty_count = 0
    invalid_count = 0
    valid_count = 0

    print("\n" + "=" * 80)
    print("RESULT ANALYSIS")
    print("=" * 80)

    for i in range(len(states)):
        answer = states[i]["answer"]
        pred = get_answer_value(answer)
        preds.append(pred)

        # Categorize the result
        if answer == "" or answer is None:
            empty_count += 1
            if args.verbose or empty_count <= 5:  # Show first 5 empty answers
                analyze_empty_answer(states[i], questions[i], labels[i], i)
        elif pred == INVALID:
            invalid_count += 1
            if args.verbose:
                print(f"\n[INVALID #{invalid_count}] Index {i}:")
                print(f"  Answer: {repr(answer[:200])}")
                print(f"  No numbers found in answer")
        else:
            valid_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal questions: {num_questions}")
    print(f"Empty answers: {empty_count} ({100*empty_count/num_questions:.1f}%)")
    print(
        f"Invalid (no numbers): {invalid_count} ({100*invalid_count/num_questions:.1f}%)"
    )
    print(f"Valid answers: {valid_count} ({100*valid_count/num_questions:.1f}%)")

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"\nAccuracy: {acc:.3f}")
    print(f"Invalid rate: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Total output tokens: {num_output_tokens}")
    print(f"Avg tokens per answer: {num_output_tokens/num_questions:.1f}")

    # Dump results
    dump_state_text("tmp_output_gsm8k_debug.txt", states)
    print(f"\nFull results saved to: tmp_output_gsm8k_debug.txt")

    # Additional analysis: token distribution
    print("\n" + "=" * 80)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("=" * 80)

    token_counts = []
    for s in states:
        try:
            tc = s.get_meta_info("answer")["completion_tokens"]
            token_counts.append(tc)
        except:
            token_counts.append(0)

    token_counts = np.array(token_counts)
    print(f"Min tokens: {token_counts.min()}")
    print(f"Max tokens: {token_counts.max()}")
    print(f"Mean tokens: {token_counts.mean():.1f}")
    print(f"Median tokens: {np.median(token_counts):.1f}")
    print(f"Std tokens: {token_counts.std():.1f}")
    print(f"Zero token answers: {(token_counts == 0).sum()}")

    # Histogram
    print("\nToken count distribution:")
    bins = [0, 1, 10, 50, 100, 200, 500, 1000]
    for i in range(len(bins) - 1):
        count = ((token_counts >= bins[i]) & (token_counts < bins[i + 1])).sum()
        print(
            f"  [{bins[i]:4d}, {bins[i+1]:4d}): {count:4d} ({100*count/len(token_counts):.1f}%)"
        )
    count = (token_counts >= bins[-1]).sum()
    print(f"  [{bins[-1]:4d},  inf): {count:4d} ({100*count/len(token_counts):.1f}%)")

    return {
        "accuracy": acc,
        "invalid": invalid,
        "empty_count": empty_count,
        "latency": latency,
        "output_throughput": output_throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=128)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--verbose", action="store_true", help="Show all invalid answers"
    )
    args = parser.parse_args()
    run_eval(args)
