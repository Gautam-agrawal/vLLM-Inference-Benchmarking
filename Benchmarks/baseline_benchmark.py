"""
Baseline Sequential Hugging Face Inference Benchmark

Measures throughput and latency of sequential (non-batched) text generation
using the Hugging Face `transformers` library. Used as the baseline for
comparison against vLLM's continuous-batching throughput.
"""

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/Phi-3-mini-4k-instruct"
NUM_PROMPTS = 20
MAX_NEW_TOKENS = 64

PROMPTS_BASE = [
    "Explain gradient descent in simple terms.",
    "Summarize the French Revolution in two sentences.",
    "What's the difference between TCP and UDP?",
    "Describe how a binary search tree works.",
    "Write a haiku about autumn.",
]


def build_prompts(num_prompts: int) -> list[str]:
    return [PROMPTS_BASE[i % len(PROMPTS_BASE)] for i in range(num_prompts)]


def run_baseline_benchmark(
    model_name: str = MODEL,
    num_prompts: int = NUM_PROMPTS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    prompts = build_prompts(num_prompts)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    latencies = []
    total_tokens = 0
    start_all = time.perf_counter()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        total_tokens += out.shape[1] - inputs["input_ids"].shape[1]
        print(f"[{i + 1}/{num_prompts}] {t1 - t0:.2f}s")

    total_time = time.perf_counter() - start_all
    latencies.sort()

    results = {
        "throughput_req_per_sec": num_prompts / total_time,
        "throughput_tok_per_sec": total_tokens / total_time,
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": latencies[int(len(latencies) * 0.5)],
        "p99_latency": latencies[-1],
    }

    # Free GPU memory before starting vLLM
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    results = run_baseline_benchmark()
    print(json.dumps(results, indent=2))

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
