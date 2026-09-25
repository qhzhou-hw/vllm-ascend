"""Mock-only Ascend composition A/B benchmark; no model or vLLM engine needed.

Run on the NPU server:
  python examples/offline_inference/hypic_compose_bench.py --batch-sizes 1 4 --segments 4 16 64

Timing includes metadata upload and allocation, but excludes JIT compilation.
This measures composition only, not full prefill/TTFT or model accuracy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType

import torch


def bootstrap():
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "vllm_ascend")]
    sys.modules["vllm_ascend"] = package


def timed(fn, iterations, repeats):
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return statistics.median(samples)


def timed_variants(functions, iterations, repeats):
    """Rotate measurement order to reduce systematic warmup/clock drift bias."""
    samples = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            samples[name].append(timed(functions[name], iterations, 1))
    return {name: statistics.median(values) for name, values in samples.items()}


def peak_workspace(fn):
    """Allocator peak above live inputs; excludes permanent pools and JIT."""
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    result = fn()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - baseline
    del result
    return peak


def check_identity(args):
    """Catch compiler regressions even on exact, non-random affine updates."""
    from vllm_ascend.hypic.compose import ComposeStep, compose_fused_batch

    dim, value_dim = args.dim, args.value_dim or args.dim
    zeros = torch.ones((1, 1, value_dim, dim), device="npu:0")
    identity = torch.eye(dim, device="npu:0")[None, None]
    # Repeated slots, different sequence lengths and incoming replay states.
    sequences = [[ComposeStep(-1)] * count + [ComposeStep(0, 0)] for count in (0, 1, 3)]
    replays, finals = compose_fused_batch([zeros] * 3, [identity] * 3, zeros, identity, sequences)
    for row, count in enumerate((0, 1, 3)):
        torch.testing.assert_close(replays[row], torch.full_like(zeros, count), rtol=0, atol=0)
        torch.testing.assert_close(finals[row], torch.full_like(zeros[0], count + 1), rtol=0, atol=0)
    print(json.dumps({"status": "PASS", "check": "identity", "dim": dim, "value_dim": value_dim}), flush=True)


def run_case(args, batch_size, segments, case):
    from vllm_ascend.hypic.compose import ComposeStep, compose_fused_batch, compose_reference

    device = "npu:0"
    heads, dim = args.heads, args.dim
    value_dim = args.value_dim or dim
    torch.manual_seed(args.seed)
    zero_pool = torch.randn((segments, heads, value_dim, dim), device=device) * 0.02
    identity = torch.eye(dim, device=device).reshape(1, 1, dim, dim)
    transition_pool = (
        torch.randn((segments, heads, dim, dim), device=device) * (0.03 / dim**0.5) + identity * 0.9
    ).contiguous()
    fresh_s, fresh_t, steps = [], [], []
    # Deliberately use noncommuting matrices, different orders and ragged
    # segment counts. All requests still have a final fresh Query unit.
    for row in range(batch_size):
        count = max(1, segments - row)
        sequence, unit = [], 0
        for index in range(count):
            hit = case == "warm" or (case == "mixed" and index % 2 == 0)
            if hit:
                slot = (index * 3 + row) % segments
                sequence.append(ComposeStep(-slot - 1))
            else:
                sequence.append(ComposeStep(unit, unit if args.mode == "legacy" else -1))
                unit += 1
        fresh_count = unit
        if not args.cache_only:
            sequence.append(ComposeStep(unit, unit))
            fresh_count += 1
        zeros = torch.randn((fresh_count, heads, value_dim, dim), device=device) * 0.02
        transforms = (
            torch.randn((fresh_count, heads, dim, dim), device=device) * (0.03 / dim**0.5) + identity * 0.9
        ).contiguous()
        fresh_s.append(zeros)
        fresh_t.append(transforms)
        steps.append(sequence)

    def reference():
        return [
            compose_reference(s, t, zero_pool, transition_pool, p, zero_replay=args.mode != "legacy")
            for s, t, p in zip(fresh_s, fresh_t, steps)
        ]

    def fused():
        return compose_fused_batch(fresh_s, fresh_t, zero_pool, transition_pool, steps)

    def before():
        # Benchmark only: route through the same validation wrapper so the
        # previous implementation includes its original host/API overhead.
        from vllm_ascend.hypic import compose_triton

        current = compose_triton.launch_compose
        try:
            compose_triton.launch_compose = args.before_launch
            return fused()
        finally:
            compose_triton.launch_compose = current

    def packed_baseline():
        from vllm_ascend.hypic import compose_triton

        current = compose_triton.launch_compose
        try:
            compose_triton.launch_compose = args.packed_launch
            return fused()
        finally:
            compose_triton.launch_compose = current

    saved_pools = zero_pool.clone(), transition_pool.clone()
    expected = reference()

    def native_baseline(launch=None, api=None):
        from vllm_ascend.hypic import compose_triton

        current = compose_triton.launch_compose
        try:
            compose_triton.launch_compose = args.native_launch if launch is None else launch
            return (api or compose_fused_batch)(fresh_s, fresh_t, zero_pool, transition_pool, steps)
        finally:
            compose_triton.launch_compose = current

    def indexed_baseline():
        return native_baseline(args.indexed_launch)

    def output_baseline():
        return native_baseline(args.output_launch, getattr(args, "output_api", None))

    def output_current_api():
        return native_baseline(args.output_launch)

    if getattr(args, "output_launch", None):
        output_replay, output_final = output_baseline()
        for row, (replay, final) in enumerate(expected):
            torch.testing.assert_close(output_replay[row], replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(output_final[row], final, atol=2e-5, rtol=2e-4)
        del output_replay, output_final
        if getattr(args, "output_api", None):
            current_replay, current_final = output_current_api()
            for row, (replay, final) in enumerate(expected):
                torch.testing.assert_close(current_replay[row], replay, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(current_final[row], final, atol=2e-5, rtol=2e-4)
            del current_replay, current_final

    if getattr(args, "indexed_launch", None):
        indexed_replay, indexed_final = indexed_baseline()
        for row, (replay, final) in enumerate(expected):
            torch.testing.assert_close(indexed_replay[row], replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(indexed_final[row], final, atol=2e-5, rtol=2e-4)
        del indexed_replay, indexed_final

    if getattr(args, "native_launch", None):
        native_replay, native_final = native_baseline()
        for row, (replay, final) in enumerate(expected):
            torch.testing.assert_close(native_replay[row], replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(native_final[row], final, atol=2e-5, rtol=2e-4)
        del native_replay, native_final
    if getattr(args, "packed_launch", None):
        packed_replay, packed_final = packed_baseline()
        for row, (replay, final) in enumerate(expected):
            torch.testing.assert_close(packed_replay[row], replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(packed_final[row], final, atol=2e-5, rtol=2e-4)
        del packed_replay, packed_final
    if args.before_root:
        old_replay, old_final = before()
        for row, (replay, final) in enumerate(expected):
            torch.testing.assert_close(old_replay[row], replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(old_final[row], final, atol=2e-5, rtol=2e-4)
        del old_replay, old_final
    actual_replay, actual_final = fused()  # compile before timing; fail loudly
    torch.npu.synchronize()
    max_abs = 0.0
    for row, (replay, final) in enumerate(expected):
        torch.testing.assert_close(actual_replay[row], replay, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual_final[row], final, atol=2e-5, rtol=2e-4)
        max_abs = max(max_abs, (actual_final[row] - final).abs().max().item())
        if replay.numel():
            max_abs = max(max_abs, (actual_replay[row] - replay).abs().max().item())
    functions = {"torch": reference, "after": fused}
    if args.before_root:
        functions["before"] = before
    if getattr(args, "packed_launch", None):
        functions["packed"] = packed_baseline
    if getattr(args, "native_launch", None):
        functions["native"] = native_baseline
    if getattr(args, "indexed_launch", None):
        functions["indexed"] = indexed_baseline
    if getattr(args, "output_launch", None):
        functions["output"] = output_baseline
        if getattr(args, "output_api", None):
            functions["output_current_api"] = output_current_api
    for _ in range(3):
        for fn in functions.values():
            fn()
    timings = timed_variants(functions, args.iterations, args.repeats)
    reference_ms, fused_ms = timings["torch"], timings["after"]
    reference_peak = peak_workspace(reference)
    fused_peak = peak_workspace(fused)
    before_peak = peak_workspace(before) if args.before_root else None
    packed_peak = peak_workspace(packed_baseline) if getattr(args, "packed_launch", None) else None
    native_peak = peak_workspace(native_baseline) if getattr(args, "native_launch", None) else None
    indexed_peak = peak_workspace(indexed_baseline) if getattr(args, "indexed_launch", None) else None
    output_peak = peak_workspace(output_baseline) if getattr(args, "output_launch", None) else None
    torch.testing.assert_close(zero_pool, saved_pools[0], rtol=0, atol=0)
    torch.testing.assert_close(transition_pool, saved_pools[1], rtol=0, atol=0)
    # Recheck after repeated launches; metadata owners must remain stream-safe.
    replay, final = fused()
    for row, (expected_replay, expected_final) in enumerate(expected):
        torch.testing.assert_close(replay[row], expected_replay, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(final[row], expected_final, atol=2e-5, rtol=2e-4)
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": args.mode,
                "case": case,
                "cache_only": args.cache_only,
                "batch_size": batch_size,
                "segments": segments,
                "heads": heads,
                "dim": dim,
                "value_dim": value_dim,
                "max_abs": max_abs,
                "torch_ms": reference_ms,
                "triton_ms": fused_ms,
                "speedup": reference_ms / fused_ms,
                "metadata_included": True,
                "before_ms": timings.get("before"),
                "packed_ms": timings.get("packed"),
                "native_ms": timings.get("native"),
                "indexed_ms": timings.get("indexed"),
                "output_ms": timings.get("output"),
                "output_current_api_ms": timings.get("output_current_api"),
                "output_workspace_bytes": output_peak,
                "indexed_workspace_bytes": indexed_peak,
                "native_workspace_bytes": native_peak,
                "packed_workspace_bytes": packed_peak,
                "before_workspace_bytes": before_peak,
                "before_speedup": timings["before"] / fused_ms if args.before_root else None,
                "torch_workspace_bytes": reference_peak,
                "triton_workspace_bytes": fused_peak,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--segments", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--cases", choices=["cold", "mixed", "warm"], nargs="+", default=["cold", "mixed", "warm"])
    parser.add_argument("--mode", choices=["legacy", "block-native"], default="block-native")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cache-only", action="store_true", help="All S/T from cache; no fresh Query summary/replay")
    parser.add_argument("--before-root", type=Path, help="Previous checkout for same-input before/after timing")
    args = parser.parse_args()
    if args.cache_only and args.cases != ["warm"]:
        parser.error("--cache-only requires --cases warm")
    if min(*args.batch_sizes, *args.segments, args.heads, args.dim, args.iterations, args.repeats) <= 0:
        parser.error("sizes, iterations and repeats must be positive")
    if args.dim > 128:
        parser.error("the experimental kernel supports key_dim <= 128")
    if args.value_dim is not None and args.value_dim <= 0:
        parser.error("--value-dim must be positive")
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    bootstrap()
    if args.before_root:
        source = args.before_root / "vllm_ascend/hypic/compose_triton.py"
        spec = importlib.util.spec_from_file_location("hypic_compose_before", source)
        previous = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = previous
        spec.loader.exec_module(previous)
        args.before_launch = previous.launch_compose
    check_identity(args)
    for batch_size in args.batch_sizes:
        for segments in args.segments:
            for case in args.cases:
                run_case(args, batch_size, segments, case)


if __name__ == "__main__":
    main()
