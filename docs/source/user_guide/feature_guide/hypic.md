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

### Segment convolution history in legacy HYPIC

`reset_conv_history` is a boolean under `hypic_config`, defaulting to `False`.
The default preserves HYPIC's existing behavior: the preceding segment's last
`K-1` raw, pre-convolution QKV vectors initialize the next segment's convolution
(including a cache-hit seam, a miss, or the final Query). Cached tails remain
available even when their source segment is skipped on a cache hit.

Set it to `True` to start convolution with zero history at each planned segment
boundary. For a workload that does not recompute hit seams, configure both
options explicitly:

```python
additional_config={
    "hypic_config": {
        "enabled": True,
        "chunk_size": 512,
        "seam_sink_tokens": 0,
        "reset_conv_history": True,
        "max_cache_segments": 96,
    }
}
```

These options are independent: `seam_sink_tokens=0` alone does not reset history;
`reset_conv_history=True` alone does not disable seam recomputation. With a
nonzero seam, reset happens once at the segment start, not between its seam and
interior. Every segment produced by the planner resets, including subdivisions
of a semantic region longer than `chunk_size` and the final Query.
This switch applies to the HYPIC segmented execution path. In legacy mode,
`hypic_cache_policy="full_recompute"` still uses native causal inference and
does not apply the segment convolution reset.

Tail caching is retained in both modes. Reset uses a separate zero buffer and
does not erase another segment's cached tail. The final Query's resulting tail
still initializes native autoregressive decode; this is not a per-token reset.
The reset mode is carried in scheduler plans and isolated in segment cache keys.
Treat it as an engine-startup setting and restart the engine to change modes.

This flag changes **convolution only**. GDN S/T composition, prefix-seeded replay
for computed document tokens, and the existing attention visibility are
unchanged. It is not a complete block-native PIC training/inference alignment
switch and does not by itself make document KV prefix-independent. Use the
`block_native_pic` mode below for training-aligned graph semantics. Existing
accuracy results do not validate reset mode; Ascend end-to-end validation is
still required.

### MindSpeed-MM block-native PIC mode

The default `mode="transition_rope_recompute"` retains the original HYPIC
algorithm. To match MindSpeed-MM's block-native PIC training graph, select:

```python
additional_config={
    "hypic_config": {
        "enabled": True,
        "mode": "block_native_pic",
        "chunk_size": 512,
        "seam_sink_tokens": 0,
        "max_cache_segments": 96,
    }
}
```

In this mode, convolution resets at every segment, including Query, regardless
of `reset_conv_history`. Document GDN outputs start from zero state; their S/T
summaries are composed in request order only to initialize Query. Document
attention is causal within each document; Query attention sees all preceding
documents plus its own causal prefix. Native decode continues Query's conv/GDN
state and reads the materialized document and Query KV cache.

Every request must supply `hypic_segment_boundaries` in `SamplingParams.extra_args`:
integer token offsets `[0, doc1_end, doc2_end, ..., prompt_length]`. The last
region is Query (including the assistant generation prefix); at least one
Document and a nonempty Query are required. Match the training token stream,
chat template and boundaries exactly. Remove structural PIC separator token
IDs just as the training preprocessing does, then compute offsets on the
remaining IDs. The engine does not strip separators or infer training regions.

Nonzero seam is rejected. Unlike legacy HYPIC, documents are never automatically
subdivided by `chunk_size`: an oversized document fails with an instruction to
increase the slot size. Query is not cached as a document and may be longer
than `chunk_size`. This prevents storage sizing from changing training semantics.

`hypic_cache_policy="full_recompute"` in this mode executes the **same block-native
graph**, but does not read or populate PIC entries. Use it as the cold baseline
against `hypic_cache_policy="pic"`; disabling HYPIC entirely would instead run
ordinary causal inference and is not an equivalent baseline. `prefix_only`
and preemption replay are rejected rather than silently changing the graph.

Cache keys include mode and effective convolution-reset behavior. Protocol v3
rejects older control plans; restart all engine workers on upgrade and warm
fresh caches. This implements graph alignment, not a claim of bitwise numerical
parity across kernels/devices. Ascend random-weight operator tests passed for
64/129/512-token documents and GDN batches 1/4, including cold/warm reuse.
Full-engine and model-level accuracy validation remain pending.
See the [training-alignment notes](../../developer_guide/hypic_block_native_alignment_zh.md).

### Experimental ordered state-composition backend

`state_compose_backend="torch"` remains the default. The opt-in `"triton"`
backend fuses ordered FP32 GDN state composition across heads and value rows;
`state_compose_batch_size` (default `1`) bounds the number of requests
whose live workspaces are grouped into one composition launch. Larger groups
can increase peak memory. Neither option changes segment order, seam handling,
or history reset.

When every Document is cached and no seam needs recomputation, both backends
compose only the cached Document summaries and run Query once, using its actual
final state for native decode. They do not compute Query S/T. Query-only legacy
plans use the same fast path. Requests with a miss or a fresh seam retain the
original packed three-pass execution, including legacy prefix-seeded replay.
Each layer retains at most eight immutable, device-qualified sequence-layout
tensors to reuse GDN metadata; this is not an additional segment/state cache.

On Ascend 910B2, the optimized vector implementation shares transition tiles
across eight value rows and packs metadata into one upload. It is **2.03–5.13×
faster than the first experimental kernel** on the tested 128×128 workloads.
Larger composition groups/sequences also outperform the torch reference, but
four-segment single-request workloads remain slower. Group size four requires
`state_compose_batch_size=4`, not merely scheduler batch size four. These are
composition-call timings, not model-level TTFT speedups. This backend is not enabled
automatically and does not silently fall back on compiler errors. See the
[design, measured results and mock benchmark instructions](../../developer_guide/hypic_state_compose_optimization_zh.md).

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
In the default legacy mode, each semantic region is split further when longer
than `chunk_size`. Block-native mode uses the stricter boundary contract above.

| Policy | Legacy HYPIC | Block-native PIC |
| --- | --- | --- |
| `pic` (default) | Segment reuse; ordinary APC disabled | Independent document reuse; ordinary APC disabled |
| `prefix_only` | Native inference and ordinary exact prefix caching | Rejected |
| `full_recompute` | Native inference without PIC/APC reuse | Block-native graph without PIC/APC reuse/publication |

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
