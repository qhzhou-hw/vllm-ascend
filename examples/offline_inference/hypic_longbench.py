"""Compare vLLM Ascend HYPIC with existing SGLang LongBench results.

The script samples one manageable example per dataset, evaluates a cold and a
warm HYPIC request, and reports both exact-text agreement and the official
LongBench per-example metric.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_DATASETS = ("qasper", "hotpotqa", "gov_report", "multi_news")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-prefix", default="hypic_512")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--min-input-tokens", type=int, default=1000)
    parser.add_argument("--max-input-tokens", type=int, default=5000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seam-sink-tokens", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_metrics(config_dir: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("longbench_metrics", config_dir / "metrics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load LongBench metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return {
        "qasper": metrics.qa_f1_score,
        "hotpotqa": metrics.qa_f1_score,
        "gov_report": metrics.rouge_score,
        "multi_news": metrics.rouge_score,
    }


def score(metric: Any, prediction: str, row: dict[str, Any]) -> float:
    return max(
        metric(
            prediction,
            answer,
            all_classes=row.get("all_classes", []),
        )
        for answer in row["answers"]
    )


def choose_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    selected = []
    for dataset in args.datasets:
        references = load_jsonl(args.reference_dir / f"{args.reference_prefix}.{dataset}.jsonl")
        eligible = [
            item
            for item in references
            if args.min_input_tokens <= int(item.get("input_tokens") or 0) <= args.max_input_tokens
        ]
        if not eligible:
            raise ValueError(f"no eligible reference rows for {dataset}")
        reference = rng.choice(eligible)
        rows = load_jsonl(args.data_dir / f"{dataset}.jsonl")
        selected.append(
            {
                "dataset": dataset,
                "index": int(reference["index"]),
                "row": rows[int(reference["index"])],
                "reference": reference,
            }
        )
    return selected


def main() -> None:
    args = parse_args()
    prompts = load_json(args.config_dir / "dataset2prompt.json")
    max_lens = load_json(args.config_dir / "dataset2maxlen.json")
    metrics = load_metrics(args.config_dir)
    selected = choose_samples(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.90,
        additional_config={
            "hypic_config": {
                "enabled": True,
                "chunk_size": args.chunk_size,
                "seam_sink_tokens": args.seam_sink_tokens,
            }
        },
    )

    results = []
    for item in selected:
        dataset = item["dataset"]
        row = item["row"]
        formatted = prompts[dataset].format(context=row["context"], input=row.get("input", ""))
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=int(max_lens[dataset]),
        )
        generated = []
        for _ in range(2):
            output = engine.generate([prompt], params, use_tqdm=False)[0].outputs[0]
            generated.append({"pred": output.text.strip(), "output_ids": list(output.token_ids)})

        reference = item["reference"]
        cold_score = score(metrics[dataset], generated[0]["pred"], row)
        warm_score = score(metrics[dataset], generated[1]["pred"], row)
        result = {
            "dataset": dataset,
            "index": item["index"],
            "reference_input_tokens": reference.get("input_tokens"),
            "sglang_pred": reference["pred"],
            "sglang_score": reference["score"],
            "vllm_cold_pred": generated[0]["pred"],
            "vllm_cold_score": cold_score,
            "vllm_warm_pred": generated[1]["pred"],
            "vllm_warm_score": warm_score,
            "cold_warm_token_match": (generated[0]["output_ids"] == generated[1]["output_ids"]),
            "sglang_vllm_text_match": reference["pred"] == generated[0]["pred"],
            "score_delta": cold_score - float(reference["score"]),
            "cold_output_ids": generated[0]["output_ids"],
            "warm_output_ids": generated[1]["output_ids"],
        }
        results.append(result)
        print(
            f"RESULT dataset={dataset} index={item['index']} "
            f"tokens={reference.get('input_tokens')} "
            f"sglang_score={100 * float(reference['score']):.2f} "
            f"vllm_score={100 * cold_score:.2f} "
            f"score_delta={100 * result['score_delta']:+.2f} "
            f"text_match={result['sglang_vllm_text_match']} "
            f"cold_warm_tokens={result['cold_warm_token_match']}",
            flush=True,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
