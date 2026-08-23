"""Run a resumable LongBench-E accuracy evaluation with vLLM Ascend."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from hypic_longbench import load_json, load_jsonl, load_metrics, score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATASETS = ("qasper", "gov_report", "hotpotqa", "multi_news")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("full_recompute", "hypic"), required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=45056)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seam-sink-tokens", type=int, default=8)
    parser.add_argument("--max-cache-segments", type=int, default=128)
    args = parser.parse_args()
    if args.mode == "hypic" and args.chunk_size <= 0:
        parser.error("--chunk-size must be positive in hypic mode")
    if args.mode == "full_recompute" and args.chunk_size != 0:
        parser.error("--chunk-size must be 0 in full_recompute mode")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.mode == "hypic" and args.batch_size != 1:
        parser.error("HYPIC currently requires --batch-size 1")
    return args


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            completed[int(item["index"])] = item
    return completed


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
    for item in items:
        length = int(item["length"])
        bucket = "0-4k" if length < 4000 else "4-8k" if length < 8000 else "8k+"
        buckets[bucket].append(float(item["score"]))
    return {
        "score": round(100 * sum(item["score"] for item in items) / len(items), 2),
        "samples": len(items),
        "mean_latency": sum(item["latency"] for item in items) / len(items),
        "mean_input_tokens": sum(item["input_tokens"] for item in items) / len(items),
        "mean_cached_tokens": 0.0,
        "length_buckets": {bucket: round(100 * sum(values) / len(values), 2) for bucket, values in buckets.items()},
    }


def write_summary(
    output_dir: Path,
    run_name: str,
    dataset_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "run_name": run_name,
        "dataset_count": len(dataset_summaries),
        "macro_average": round(
            sum(item["score"] for item in dataset_summaries.values()) / len(dataset_summaries),
            2,
        ),
        "datasets": dataset_summaries,
    }
    if all("length_buckets" in item for item in dataset_summaries.values()):
        summary["length_bucket_macro_average"] = {
            bucket: round(
                sum(item["length_buckets"][bucket] for item in dataset_summaries.values()) / len(dataset_summaries),
                2,
            )
            for bucket in ("0-4k", "4-8k", "8k+")
        }
    target = output_dir / f"{run_name}.summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_json(args.config_dir / "dataset2prompt.json")
    max_lens = load_json(args.config_dir / "dataset2maxlen.json")
    metrics = load_metrics(args.config_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    run_name = args.run_name or ("full_recompute" if args.mode == "full_recompute" else f"hypic_{args.chunk_size}")

    engine_args: dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.mode == "hypic":
        engine_args["additional_config"] = {
            "hypic_config": {
                "enabled": True,
                "chunk_size": args.chunk_size,
                "seam_sink_tokens": args.seam_sink_tokens,
                "max_cache_segments": args.max_cache_segments,
            }
        }
    else:
        engine_args["enable_prefix_caching"] = False
    engine = LLM(**engine_args)

    dataset_summaries: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        rows = load_jsonl(args.data_dir / f"{dataset}.jsonl")
        if args.limit > 0:
            rows = rows[: args.limit]
        output_path = args.output_dir / f"{run_name}.{dataset}.jsonl"
        completed = load_completed(output_path)
        print(
            f"DATASET_START run={run_name} dataset={dataset} rows={len(rows)} resumed={len(completed)}",
            flush=True,
        )
        with output_path.open("a", encoding="utf-8") as output_file:
            pending = [(index, row) for index, row in enumerate(rows) if index not in completed]
            for batch_start in range(0, len(pending), args.batch_size):
                batch = pending[batch_start : batch_start + args.batch_size]
                prepared = []
                for index, row in batch:
                    formatted = prompts[dataset].format(
                        context=row["context"],
                        input=row.get("input", ""),
                    )
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": formatted}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    prepared.append(
                        (
                            index,
                            row,
                            prompt,
                            len(tokenizer.encode(prompt, add_special_tokens=False)),
                        )
                    )
                started = time.perf_counter()
                request_outputs = engine.generate(
                    [item[2] for item in prepared],
                    SamplingParams(temperature=0.0, max_tokens=int(max_lens[dataset])),
                    use_tqdm=False,
                )
                batch_latency = time.perf_counter() - started
                for (index, row, _, input_tokens), request_output in zip(
                    prepared,
                    request_outputs,
                    strict=True,
                ):
                    generation = request_output.outputs[0]
                    prediction = generation.text.strip()
                    item = {
                        "dataset": dataset,
                        "index": index,
                        "pred": prediction,
                        "answers": row["answers"],
                        "all_classes": row.get("all_classes", []),
                        "length": row.get("length"),
                        "score": score(metrics[dataset], prediction, row),
                        "input_tokens": input_tokens,
                        "segments": (
                            (input_tokens + args.chunk_size - 1) // args.chunk_size if args.mode == "hypic" else 1
                        ),
                        "output_ids": list(generation.token_ids),
                        "cached_tokens": 0,
                        "prompt_tokens": input_tokens,
                        "completion_tokens": len(generation.token_ids),
                        "latency": batch_latency,
                        "finish_reason": generation.finish_reason,
                    }
                    output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    completed[index] = item
                    print(
                        f"SAMPLE run={run_name} dataset={dataset} "
                        f"{index + 1}/{len(rows)} score={100 * item['score']:.2f} "
                        f"tokens={input_tokens} batch_latency={batch_latency:.2f}s",
                        flush=True,
                    )

        ordered = [completed[index] for index in range(len(rows))]
        dataset_summaries[dataset] = summarize(ordered)
        summary = write_summary(args.output_dir, run_name, dataset_summaries)
        print(
            f"DATASET_DONE run={run_name} dataset={dataset} "
            f"score={dataset_summaries[dataset]['score']:.2f} "
            f"macro_so_far={summary['macro_average']:.2f}",
            flush=True,
        )

    summary = write_summary(args.output_dir, run_name, dataset_summaries)
    print("RUN_DONE " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
