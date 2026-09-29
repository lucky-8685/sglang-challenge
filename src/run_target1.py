import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:30000"
MAX_WORKERS = 8
MAX_NEW_TOKENS = 16
PARAMS = {
    "temperature": 0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "ignore_eos": True,
    "sampling_seed": 2026,
}

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def flush_cache():
    response = requests.post(f"{BASE_URL}/flush_cache", timeout=60)
    response.raise_for_status()
    if "Cache flushed." not in response.text:
        raise RuntimeError(f"缓存清理没有确认成功：{response.text}")
    print("缓存已清理。")

def warm_shared_prefix():
    with open("target1/shared_prefix/warmup.json", encoding="utf-8") as f:
        warmup = json.load(f)

    payload = {
        "input_ids": warmup["input_ids"],
        "sampling_params": {
            **PARAMS,
            "max_new_tokens": 1,
        },
        "stream": False,
    }
    response = requests.post(
        f"{BASE_URL}/generate", json=payload, timeout=(30, 600)
    )
    response.raise_for_status()
    print("共享前缀预热完成，不计入测量。")

def measure_one(request):
    input_ids = request["input_ids"]
    started = time.perf_counter()
    first_token_time = None
    final_meta = {}
    output_count = 0
    status_code = 0
    received_done = False

    try:
        payload = {
            "input_ids": input_ids,
            "sampling_params": PARAMS,
            "stream": True,
        }
        with requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            stream=True,
            timeout=(30, 600),
        ) as response:
            status_code = response.status_code
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue

                data = line[5:].strip()
                if data == b"[DONE]":
                    received_done = True
                    break

                part = json.loads(data)
                if "error" in part:
                    raise RuntimeError(part["error"])

                output_ids = part.get("output_ids") or []
                if output_ids and first_token_time is None:
                    first_token_time = time.perf_counter()
                output_count = max(output_count, len(output_ids))

                meta = part.get("meta_info") or {}
                final_meta.update(meta)

        ended = time.perf_counter()
        elapsed = ended - started
        ttft = (
            first_token_time - started
            if first_token_time is not None else None
        )
        prompt_tokens = int(final_meta.get("prompt_tokens", len(input_ids)))
        completion_tokens = int(
            final_meta.get("completion_tokens") or output_count
        )
        cached_tokens = int(final_meta.get("cached_tokens", 0))
        success = (
            status_code == 200
            and received_done
            and first_token_time is not None
            and completion_tokens == MAX_NEW_TOKENS
        )
        tpot = (
            (elapsed - ttft) / (completion_tokens - 1)
            if ttft is not None and completion_tokens > 1 else None
        )

        return {
            "request_id": request["request_id"],
            "success": success,
            "status_code": status_code,
            "input_tokens": len(input_ids),
            "actual_input_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "prefill_tokens": max(0, prompt_tokens - cached_tokens),
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "e2e_seconds": elapsed,
            "error": "",
        }

    except Exception as error:
        ended = time.perf_counter()
        return {
            "request_id": request["request_id"],
            "success": False,
            "status_code": status_code,
            "input_tokens": len(input_ids),
            "actual_input_tokens": len(input_ids),
            "completion_tokens": output_count,
            "cached_tokens": 0,
            "prefill_tokens": len(input_ids),
            "ttft_seconds": None,
            "tpot_seconds": None,
            "e2e_seconds": ended - started,
            "error": str(error),
        }

def percentile(values, percent):
    values = sorted(values)
    if not values:
        return None
    index = math.ceil(percent * len(values)) - 1
    return values[max(0, min(index, len(values) - 1))]

def run_group(name):
    requests_to_send = read_jsonl(
        f"target1/{name}/requests.jsonl"
    )
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(measure_one, item)
            for item in requests_to_send
        ]
        results = [future.result() for future in as_completed(futures)]

    elapsed = time.perf_counter() - started
    results.sort(key=lambda row: row["request_id"])

    output_dir = Path("results/target1")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    fields = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    successful = [row for row in results if row["success"]]
    prompt_total = sum(row["actual_input_tokens"] for row in successful)
    cached_total = sum(row["cached_tokens"] for row in successful)

    summary = {
        "group": name,
        "request_count": len(results),
        "success_count": len(successful),
        "success_rate": len(successful) / len(results),
        "throughput_requests_per_second": len(successful) / elapsed,
        "cache_hit_rate": cached_total / prompt_total if prompt_total else 0,
        "cached_tokens_total": cached_total,
        "actual_prefill_tokens_total": sum(
            row["prefill_tokens"] for row in successful
        ),
        "wall_time_seconds": elapsed,
    }

    for metric in ("ttft_seconds", "tpot_seconds", "e2e_seconds"):
        values = [
            row[metric] for row in successful
            if row[metric] is not None
        ]
        summary[f"{metric}_p50"] = percentile(values, 0.50)
        summary[f"{metric}_p95"] = percentile(values, 0.95)

    with open(output_dir / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"{name} 完成：成功 {len(successful)}/{len(results)}")
    print(f"请求结果保存到：{csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    requests.get(f"{BASE_URL}/v1/models", timeout=10).raise_for_status()

    shared = read_jsonl("target1/shared_prefix/requests.jsonl")
    dispersed = read_jsonl("target1/dispersed_prefix/requests.jsonl")
    if len(shared) != 32 or len(dispersed) != 32:
        raise SystemExit("两组都必须正好有 32 条请求。")

    print("先测共享前缀组。")
    flush_cache()
    warm_shared_prefix()
    run_group("shared_prefix")

    print("再测分散前缀组。")
    flush_cache()
    run_group("dispersed_prefix")

if __name__ == "__main__":
    main()
