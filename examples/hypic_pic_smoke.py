"""Server-side PIC/APC isolation and small-pool smoke test.

Token chunks are constructed explicitly for this synthetic test. This is not
the MCPAgentBench prompt formatter or an accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seam", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--cache-segments", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len * args.batch_size,
        additional_config={
            "hypic_config": {
                "enabled": True,
                "chunk_size": args.chunk_size,
                "seam_sink_tokens": args.seam,
                "max_cache_segments": args.cache_segments,
            }
        },
    )
    tokenizer = llm.get_tokenizer()
    descriptions = [
        "Tool search_docs: Search technical documents by a text query.\n",
        "Tool get_weather: Look up the weather of a named city.\n",
        "Tool add_numbers: Add a list of numbers.\n",
    ]

    def prompt(order):
        # These token sequences are passed unchanged to the engine. Boundaries
        # therefore refer to actual tokens, not source JSON character offsets.
        parts = ["Choose a tool for this question. Reply with its name only.\n"]
        parts += [descriptions[i] for i in order]
        parts += ["Question: What is the weather in Beijing?\nAnswer:"]
        token_ids, boundaries = [], [0]
        for part in parts:
            token_ids.extend(tokenizer.encode(part, add_special_tokens=False))
            boundaries.append(len(token_ids))
        return {"prompt_token_ids": token_ids}, boundaries

    original, boundaries = prompt([0, 1, 2])
    reordered, reordered_boundaries = prompt([2, 0, 1])
    records = {}

    def run(name, policy, item, cuts):
        params = SamplingParams(
            temperature=0,
            max_tokens=16,
            seed=20260916,
            extra_args={
                "hypic_cache_policy": policy,
                "hypic_segment_boundaries": cuts,
            },
        )
        outputs = llm.generate([item] * args.batch_size, params, use_tqdm=False)
        records[name] = [{"tokens": list(out.outputs[0].token_ids), "text": out.outputs[0].text} for out in outputs]
        print(name, json.dumps(records[name], ensure_ascii=False), flush=True)
        if any(not row["tokens"] for row in records[name]):
            raise AssertionError(f"Empty generation in {name}")

    run("full_before", "full_recompute", original, boundaries)
    run("pic_cold", "pic", original, boundaries)
    run("pic_warm", "pic", original, boundaries)
    run("pic_reordered", "pic", reordered, reordered_boundaries)
    run("prefix_cold", "prefix_only", original, boundaries)
    run("prefix_warm", "prefix_only", original, boundaries)
    run("full_after", "full_recompute", original, boundaries)
    comparisons = {
        name: records[name] == records["full_before"]
        for name in ("pic_cold", "pic_warm", "prefix_cold", "prefix_warm", "full_after")
    }
    # Preserve evidence even on an isolation failure. PIC/full differences are
    # reported, not treated as exactness failures of the approximate algorithm.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {"config": vars(args) | {"output": str(args.output)}, "runs": records, "exact_vs_full_before": comparisons},
            stream,
            ensure_ascii=False,
            indent=2,
        )
    for name in ("prefix_cold", "prefix_warm", "full_after"):
        if not comparisons[name]:
            raise AssertionError(f"Native/APC regression after PIC: {name}; inspect {args.output}")


if __name__ == "__main__":
    main()
