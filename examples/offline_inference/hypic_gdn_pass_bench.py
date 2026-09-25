"""Real Ascend GDN before/after differential benchmark with mock weights only.

--before-root must point at the preserved pre-pass-reduction checkout.
This times one GDN layer including convolution, gating, S/T, composition,
replay and native-state writes, not a whole model or generation request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from hypic_mock_validate import bootstrap, load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--documents", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--doc-len", type=int, default=512)
    parser.add_argument("--query-len", type=int, default=9)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mode", choices=["block-native", "legacy", "legacy-reset"], default="block-native")
    parser.add_argument("--seam", type=int, default=8)
    parser.add_argument("--heterogeneous", action="store_true", help="mix cache-only and fresh requests in one group")
    args = parser.parse_args()
    if min(*args.batches, args.doc_len, args.query_len, args.heads, args.repeats, args.iterations) < 1:
        parser.error("sizes/repetitions must be positive")
    if min(args.documents) < 0 or not 0 <= args.seam < args.doc_len:
        parser.error("documents must be nonnegative and seam must be shorter than a document")
    if args.mode == "block-native" and 0 in args.documents:
        parser.error("block-native plans require at least one document")
    root = Path(__file__).resolve().parents[2]
    bootstrap(root)
    import torch_npu  # noqa: F401

    from vllm_ascend.hypic import gdn
    from vllm_ascend.hypic.config import HypicConfig
    from vllm_ascend.hypic.planner import build_plan
    from vllm_ascend.hypic.runtime import HypicBatchContext

    before = load_module("hypic_gdn_before", args.before_root / "vllm_ascend/hypic/gdn.py")
    torch.npu.set_device(0)
    torch.manual_seed(20260924)
    device, dtype = "npu:0", torch.bfloat16
    heads, dim, width = args.heads, 128, 4
    channels = 3 * heads * dim
    native = args.mode == "block-native"
    config = HypicConfig(
        mode="block_native_pic" if native else "transition_rope_recompute",
        chunk_size=args.doc_len,
        seam_sink_tokens=0 if native else args.seam,
        reset_conv_history=native or args.mode == "legacy-reset",
    )
    results = []
    for batch in args.batches:
        for documents in args.documents:
            total = documents * args.doc_len + args.query_len
            raw = (torch.randn((total + args.doc_len, channels)) * 0.15).to(device, dtype)
            a = (torch.randn((len(raw), heads)) * 0.1).to(device, dtype)
            b = (torch.randn((len(raw), heads)) * 0.1).to(device, dtype)
            weight = (torch.randn((channels, 1, width)) * 0.1).to(device, dtype)

            def make_layer(batch=batch, documents=documents, weight=weight):
                return SimpleNamespace(
                    prefix="mock.gdn",
                    hypic_state_compose_backend=args.backend,
                    hypic_state_compose_batch_size=batch,
                    conv1d=SimpleNamespace(weight=weight, bias=None),
                    activation="silu",
                    hypic_conv_pool=torch.zeros((documents, width - 1, channels), device=device, dtype=dtype),
                    hypic_zero_state_pool=torch.zeros((documents, heads, dim, dim), device=device),
                    hypic_transition_pool=torch.zeros((documents, heads, dim, dim), device=device),
                    kv_cache=(
                        torch.zeros((batch, width - 1, channels), device=device, dtype=dtype),
                        torch.zeros((batch, heads, dim, dim), device=device),
                    ),
                    # Slower decay than the smoke test exercises accumulated history.
                    A_log=torch.full((heads,), -5.0, device=device),
                    dt_bias=torch.zeros(heads, device=device),
                    rearrange_mixed_qkv=lambda x: tuple(t.reshape(1, -1, heads, dim) for t in x.chunk(3, -1)),
                )

            layers = [make_layer(), make_layer()]
            modules = [before, gdn]
            ready, slots = {}, {}
            cases = ["cold", "warm", "reordered", "mixed"] + (["heterogeneous"] if args.heterogeneous else [])
            for case in cases:
                doc_tokens = list(range(documents * args.doc_len))
                if case == "reordered":
                    doc_tokens = [
                        t
                        for doc in reversed(range(documents))
                        for t in range(doc * args.doc_len, (doc + 1) * args.doc_len)
                    ]
                if case == "mixed" and documents:
                    doc_tokens[: args.doc_len] = range(total, total + args.doc_len)
                tokens = [
                    doc_tokens + list(range(documents * args.doc_len, total - min(row, args.query_len - 1)))
                    for row in range(batch)
                ]
                plans = {}
                for row, row_tokens in enumerate(tokens):
                    boundaries = [i * args.doc_len for i in range(documents + 1)] + [len(row_tokens)]
                    row_ready = {} if case == "heterogeneous" and row % 2 else ready
                    plan = build_plan(row_tokens, row_ready, config, segment_boundaries=boundaries)
                    for s in plan["segments"][:-1]:
                        if case == "cold":
                            slots.setdefault(s["hash"], len(slots))
                        s["store"] = row == 0 and not s["hit"] and s["hash"] in slots
                    plans[str(row)] = plan
                context = HypicBatchContext(plans, tuple(plans), SimpleNamespace(lookup=slots.get))
                packed = [
                    torch.cat([table[ts][plans[str(row)]["query_positions"]] for row, ts in enumerate(tokens)])
                    for table in (raw, b, a)
                ]
                outputs = [torch.empty((len(packed[0]), heads, dim), device=device, dtype=dtype) for _ in layers]
                metadata = SimpleNamespace(
                    num_actual_tokens=len(packed[0]),
                    non_spec_state_indices_tensor=torch.arange(batch - 1, -1, -1, device=device),
                )
                call_counts = []
                for module, layer, output in zip(modules, layers, outputs):
                    original = module._run_gdn
                    counts = []

                    def tracked(*inputs, counts=counts, original=original):
                        counts.append(inputs[1].shape[1])
                        return original(*inputs)

                    module._run_gdn = tracked
                    try:
                        module.forward_hypic_gdn(layer, *packed, output, context, metadata)
                    finally:
                        module._run_gdn = original
                    call_counts.append(counts)
                torch.npu.synchronize()
                torch.testing.assert_close(outputs[1], outputs[0], atol=1e-5, rtol=0.02)
                for actual, expected in zip(layers[1].kv_cache, layers[0].kv_cache):
                    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.02)
                for name in ("hypic_conv_pool", "hypic_zero_state_pool", "hypic_transition_pool"):
                    torch.testing.assert_close(
                        getattr(layers[1], name), getattr(layers[0], name), atol=2e-5, rtol=0.002
                    )
                peaks = []
                for module, layer, output in zip(modules, layers, outputs):
                    torch.npu.synchronize()
                    baseline = torch.npu.memory_allocated()
                    torch.npu.reset_peak_memory_stats()
                    module.forward_hypic_gdn(layer, *packed, output, context, metadata)
                    torch.npu.synchronize()
                    peaks.append(torch.npu.max_memory_allocated() - baseline)
                # Warmup/JIT is excluded. Rotate timing order to limit order bias.
                timings = [[], []]
                for repeat in range(args.repeats + 1):
                    for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            modules[index].forward_hypic_gdn(layers[index], *packed, outputs[index], context, metadata)
                        torch.npu.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000 / args.iterations
                        if repeat:
                            timings[index].append(elapsed)
                before_ms, after_ms = map(statistics.median, timings)
                result = dict(
                    mode=args.mode,
                    backend=args.backend,
                    batch=batch,
                    documents=documents,
                    case=case,
                    doc_len=args.doc_len,
                    query_len=args.query_len,
                    heads=heads,
                    before_ms=before_ms,
                    after_ms=after_ms,
                    speedup=before_ms / after_ms,
                    output_max_abs=(outputs[1].float() - outputs[0].float()).abs().max().item(),
                    state_max_abs=(layers[1].kv_cache[1] - layers[0].kv_cache[1]).abs().max().item(),
                    gdn_calls_before=len(call_counts[0]),
                    gdn_calls_after=len(call_counts[1]),
                    gdn_tokens_before=sum(call_counts[0]),
                    gdn_tokens_after=sum(call_counts[1]),
                    extra_peak_bytes_before=peaks[0],
                    extra_peak_bytes_after=peaks[1],
                )
                results.append(result)
                print(json.dumps(result), flush=True)
                if case == "cold":
                    ready = {s["hash"]: tuple(s["token_ids"]) for s in plans["0"]["segments"][:-1]}
    print(json.dumps(dict(status="PASS", cases=len(results), mock_weights_only=True)), flush=True)


if __name__ == "__main__":
    main()
