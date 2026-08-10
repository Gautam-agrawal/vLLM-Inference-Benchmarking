"""
vLLM Throughput and Latency Benchmark

Starts a local vLLM OpenAI-compatible API server, sends concurrent requests
via aiohttp, and measures throughput/latency. Compares results against a
baseline (see baseline_benchmark.py) to compute the speedup from continuous
batching.
"""

import asyncio
import json
import subprocess
import time

import aiohttp
import requests

MODEL = "microsoft/Phi-3-mini-4k-instruct"
NUM_PROMPTS = 20
MAX_NEW_TOKENS = 64
CONCURRENCY = 8
VLLM_PORT = 8000
STARTUP_TIMEOUT_S = 90
STARTUP_POLL_INTERVAL_S = 5

PROMPTS_BASE = [
    "Explain gradient descent in simple terms.",
    "Summarize the French Revolution in two sentences.",
    "What's the difference between TCP and UDP?",
    "Describe how a binary search tree works.",
    "Write a haiku about autumn.",
]


def build_prompts(num_prompts: int) -> list[str]:
    return [PROMPTS_BASE[i % len(PROMPTS_BASE)] for i in range(num_prompts)]


def start_vllm_server(model_name: str = MODEL, port: int = VLLM_PORT) -> subprocess.Popen:
    """Launch the vLLM OpenAI-compatible API server as a subprocess."""
    proc = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--gpu-memory-utilization", "0.7",
            "--dtype", "half",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    for _ in range(STARTUP_TIMEOUT_S // STARTUP_POLL_INTERVAL_S):
        try:
            r = requests.get(f"http://localhost:{port}/health")
            if r.status_code == 200:
                print("vLLM server ready.")
                return proc
        except Exception:
            pass
        time.sleep(STARTUP_POLL_INTERVAL_S)

    print("Not ready yet — checking logs:")
    print(proc.stdout.read())
    return proc


async def _send(session, sem, prompt, model_name, port, max_new_tokens):
    async with sem:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        async with session.post(
            f"http://localhost:{port}/v1/chat/completions", json=payload
        ) as resp:
            data = await resp.json()
        t1 = time.perf_counter()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return t1 - t0, tokens


async def _run_concurrent_benchmark(
    prompts, model_name, port, concurrency, max_new_tokens
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        results = await asyncio.gather(
            *[_send(session, sem, p, model_name, port, max_new_tokens) for p in prompts]
        )
        total_time = time.perf_counter() - start

    latencies = sorted(r[0] for r in results)
    total_tokens = sum(r[1] for r in results)

    return {
        "throughput_req_per_sec": len(prompts) / total_time,
        "throughput_tok_per_sec": total_tokens / total_time,
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": latencies[int(len(latencies) * 0.5)],
        "p99_latency": latencies[-1],
        "concurrency": concurrency,
    }


def run_vllm_benchmark(
    model_name: str = MODEL,
    num_prompts: int = NUM_PROMPTS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    concurrency: int = CONCURRENCY,
    port: int = VLLM_PORT,
) -> dict:
    prompts = build_prompts(num_prompts)
    return asyncio.run(
        _run_concurrent_benchmark(prompts, model_name, port, concurrency, max_new_tokens)
    )


def compute_speedup(baseline_results: dict, vllm_results: dict) -> float:
    return vllm_results["throughput_req_per_sec"] / baseline_results["throughput_req_per_sec"]


if __name__ == "__main__":
    vllm_proc = start_vllm_server()

    try:
        vllm_results = run_vllm_benchmark()
        print(json.dumps(vllm_results, indent=2))

        with open("vllm_results.json", "w") as f:
            json.dump(vllm_results, f, indent=2)

        try:
            with open("baseline_results.json") as f:
                baseline_results = json.load(f)
            speedup = compute_speedup(baseline_results, vllm_results)
            print(f"Baseline: {baseline_results['throughput_req_per_sec']:.2f} req/sec")
            print(f"vLLM:     {vllm_results['throughput_req_per_sec']:.2f} req/sec")
            print(f"Speedup:  {speedup:.2f}x")
        except FileNotFoundError:
            print("baseline_results.json not found — run baseline_benchmark.py first "
                  "to compute the speedup comparison.")
    finally:
        vllm_proc.terminate()
