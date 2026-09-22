# HYPIC

HYPIC accelerates repeated long-context prefill by caching fixed-size context
segments. On a cache hit, vLLM Ascend restores the segment's attention KV data
and recurrent Gated DeltaNet state, while recomputing a small seam around the
segment boundary. The feature is opt-in and currently targets Qwen3.5 hybrid
attention models on Ascend 910B.

The PIC scheduler adapter described below is newly implemented and awaits
Ascend integration/accuracy validation. Published results for the previous
HYPIC implementation do not validate this adapter.

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

`seam_sink_tokens` may be set to `0` for complete segment reuse:

```python
additional_config={
    "hypic_config": {
        "enabled": True,
        "chunk_size": 512,
        "seam_sink_tokens": 0,
        "max_cache_segments": 96,
    }
}
```

With zero seam, a non-final cache-hit segment contributes no query tokens.
HYPIC restores its attention KV data and composes its cached GDN transition
state directly. The final segment is still fully recomputed to produce logits,
and a cache miss is always computed regardless of this setting. Zero seam is
supported by the execution path, but the published LongBench-E and
MCPAgentBench accuracy results used the default value of `8`. Use `0` only
after checking accuracy for the target workload; it may amplify approximation
error around segment boundaries.

`max_cache_segments` sizes fixed model-owned attention and GDN state pools.
vLLM accounts for these buffers before sizing its ordinary KV cache. Admitted
hits keep their slots for the whole forward. If the pool cannot store all new
segments, the remaining misses compute normally without a persistent cache
write; the old minimum-slot formula no longer applies.

`max_prefill_units` (default `256`) independently bounds GDN S/T workspace
units per batch. Many short semantic segments can consume substantial FP32
workspace despite a small token count. Requests exceeding this limit on their
own are rejected with an explicit error; increase it only after sizing the
workspace on the target model/device.

## PIC matching and request policies

PIC keys use segment token content, request `cache_salt`, and the effective
full/interior GDN representation. Entries belong to one engine/model lifetime.
Moving an interior tool segment to another interior position can hit even when
its prefix changes. Moving between the first and an interior segment with a
nonzero seam may miss, because their cached transitions cover different ranges.
Short segments entirely consumed by the seam are computed without caching.

RoPE relocation does not remove dependence on the original context. PIC is
approximate reuse and must be evaluated against full recomputation.

```python
params = SamplingParams(
    temperature=0,
    max_tokens=128,
    extra_args={
        "hypic_cache_policy": "pic",
        # Token offsets in the actual rendered/tokenized input; optional.
        "hypic_segment_boundaries": [0, 128, 512, 900],
    },
)
```

The boundary offsets above are illustrative and must match the actual input.
Each semantic region is split further when longer than `chunk_size`.

| Policy | Behavior with HYPIC enabled |
| --- | --- |
| `pic` (default) | Segment reuse; ordinary APC lookup and publication disabled |
| `prefix_only` | Native inference and ordinary exact prefix caching |
| `full_recompute` | Native inference with neither PIC nor APC reuse/publication |

The scheduler separates PIC and native prefill batches. PIC results never
enter the ordinary APC namespace, including subsequent decode blocks. Native
decode consumes the materialized request KV and composed request GDN state.
Combining APC and PIC within one request is not enabled in this adapter.

The first adapter conservatively budgets the full logical prompt length during
admission. Only the worker executes the smaller query set; logical progress and
block allocation retain their native meaning. Restore volume is bounded by this
same logical-token budget. Consequently a warm cache does not increase the
number of logical prompt tokens admitted per batch yet.

An in-flight cache failure stops the engine rather than publishing uncertain
slots or retrying against a partially overwritten cache. Restarting creates a
new cache epoch and requires warming the cache again.

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
- The PIC adapter rejects LoRA and quantized-model configurations, as well as
  routed-expert output. These require additional payload and metadata support.
- The cache is process-local and is cleared when the engine exits.
- `seam_sink_tokens=0` is allowed, but is a more aggressive reuse mode than the
  validated default of `8` and has not yet passed the documented accuracy
  suites.

Server validation commands and implementation boundaries are recorded in the
[PIC implementation and validation notes](../../developer_guide/hypic_pic_implementation_zh.md).
