"""Run a cold/warm HYPIC smoke test with deterministic decoding."""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--repeat", type=int, default=140)
    parser.add_argument("--disable-hypic", action="store_true")
    parser.add_argument("--seam", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_args = dict(
        model=args.model,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.90,
    )
    if not args.disable_hypic:
        engine_args["additional_config"] = {
            "hypic_config": {
                "enabled": True,
                "chunk_size": 512,
                "seam_sink_tokens": args.seam,
            }
        }
    llm = LLM(**engine_args)
    prompt = "Summarize this repeated technical note. " + (
        "Ascend inference reuses stable context segments while recomputing "
        "boundary tokens. "
        * args.repeat
    )
    params = SamplingParams(temperature=0.0, max_tokens=4)
    token_runs: list[list[int]] = []
    num_runs = 1 if args.disable_hypic else 2
    for run in range(num_runs):
        output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        token_ids = list(output.token_ids)
        token_runs.append(token_ids)
        print(f"RUN {run} TOKENS {token_ids} TEXT {output.text!r}")
    if len(token_runs) == 2:
        print(f"COLD_WARM_EXACT_MATCH {token_runs[0] == token_runs[1]}")


if __name__ == "__main__":
    main()
