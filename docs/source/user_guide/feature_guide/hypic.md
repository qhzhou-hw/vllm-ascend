# HYPIC

HYPIC accelerates repeated long-context prefill by caching fixed-size context
segments. On a cache hit, vLLM Ascend restores the segment's attention KV data
and recurrent Gated DeltaNet state, while recomputing a small seam around the
segment boundary. The feature is opt-in and currently targets Qwen3.5 hybrid
attention models on Ascend 910B.

For implementation details and reproducible accuracy evaluation, see the
[Chinese porting guide](../../developer_guide/hypic_ascend_porting_zh.md) and
[LongBench-E guide](../../developer_guide/evaluation/hypic_longbench_zh.md).

## Prerequisites

HYPIC's Gated DeltaNet path uses the Ascend chunk kernel distributed by
`sgl-kernel-npu`. Install a build compatible with the Python, CANN, and hardware
versions in the vLLM Ascend environment. The implementation was validated with
`sgl-kernel-npu==2026.5.1` and CANN 9.0.0 on Ascend 910B.

## Offline inference

Enable HYPIC through `additional_config`. Prefix caching remains enabled
internally because vLLM uses its block manager for decode KV allocation; HYPIC
handles prefill reuse independently.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3.5-35B-A3B",
    tensor_parallel_size=2,
    enforce_eager=True,
    max_num_seqs=4,
    max_num_batched_tokens=45056,
    additional_config={
        "hypic_config": {
            "enabled": True,
            "chunk_size": 512,
            "seam_sink_tokens": 8,
            "max_cache_segments": 96,
        }
    },
)

outputs = llm.generate(
    ["A long prompt whose stable segments will be reused."],
    SamplingParams(temperature=0, max_tokens=128),
)
```

`chunk_size` controls the fixed segment size in tokens. A final partial segment
is always recomputed. `seam_sink_tokens` controls how many tokens at a reused
segment boundary are recomputed to reduce boundary error.

`max_cache_segments` sizes fixed model-owned attention and GDN state pools.
vLLM accounts for these buffers before sizing its ordinary KV cache. It must be
at least `ceil(max_num_batched_tokens / chunk_size) - 1`, so every cacheable
segment in one packed prefill keeps a stable slot across all model layers.

The repository includes two runnable examples:

```bash
python examples/hypic_smoke.py \
    --model /path/to/Qwen3.5-35B-A3B

python examples/offline_inference/hypic_longbench.py \
    --model /path/to/Qwen3.5-35B-A3B \
    --data-dir /path/to/LongBench/data \
    --config-dir /path/to/LongBench/config \
    --reference-dir /path/to/sglang/results \
    --chunk-size 512 \
    --output /tmp/hypic-longbench.json
```

## Current limitations

- Only Qwen3.5 dense and MoE hybrid-attention architectures are supported.
- Tensor parallelism is supported. Pipeline, data, context, decode-context, and
  prefill-context parallelism are not supported.
- HYPIC requires eager execution and disables vLLM chunked prefill. Batched
  prefill and decode are supported; the scheduler keeps a HYPIC prefill group
  separate from already-running decode requests. `max_num_batched_tokens`
  limits the total tokens admitted per step, so the effective concurrency may
  be lower than `max_num_seqs` for very long prompts. Speculative decoding and
  KV-transfer connectors are not supported.
- Prompt logprobs are not supported. Requests must contain prompt token IDs by
  the time they reach the model runner; normal vLLM text inputs satisfy this
  requirement through tokenizer preprocessing.
- The cache is process-local and is cleared when the engine exits.
