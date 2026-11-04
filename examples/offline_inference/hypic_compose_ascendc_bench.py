# SPDX-License-Identifier: Apache-2.0
"""Mock-only Ascend C validation/benchmark; no vLLM, model, or GDN required."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch


def load_source(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(s, t, slots):
    result = torch.zeros((len(slots), *s.shape[1:]), device=s.device, dtype=s.dtype)
    for row, sequence in enumerate(slots):
        for slot in sequence:
            result[row] = torch.bmm(result[row], t[slot]) + s[slot]
    return result


def inputs(batch, segments, heads, seed):
    torch.manual_seed(seed)
    pool_size = max(1, segments)
    s = torch.randn((pool_size, heads, 128, 128), device="npu:0") * 0.02
    identity = torch.eye(128, device=s.device)[None, None]
    t = identity * 0.9 + torch.randn_like(s) * 0.003
    # Non-monotone, repeated slots and unequal request lengths.
    slots = [[(i * 3 + row) % pool_size for i in range(max(0, segments - row))] for row in range(batch)]
    return s, t, slots


def benchmark(calls, iterations, repeats):
    values = {name: [] for name in calls}
    names = list(calls)
    for call in calls.values():
        for _ in range(3):
            call()
    torch.npu.synchronize()
    for repeat in range(repeats):
        for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                calls[name]()
            torch.npu.synchronize()
            values[name].append((time.perf_counter() - start) * 1000 / iterations)
    return {name: statistics.median(samples) for name, samples in values.items()}


def final_output(call):
    return call()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--namespace", default="hypic_ascendc")
    parser.add_argument("--tile-rows", type=int, nargs="+", choices=[0, 32, 64, 128], default=[0])
    parser.add_argument("--baseline-root", type=Path, help="Frozen v1 library and Python wrapper directory")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--segments", type=int, nargs="+", default=[0, 1, 2, 4, 16, 64])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--triton", action="store_true", help="Also compare existing validated Vector/Cube launchers")
    parser.add_argument("--tests", action="store_true", help="Run NPU regression suite before benchmarks")
    parser.add_argument("--trace", action="store_true", help="Print validation/measurement checkpoints")
    parser.add_argument("--tree", action="store_true", help="Compare ordered Ascend C tree against sequential compose")
    parser.add_argument("--single-tree", action="store_true", help="Compare one-launch tree/hybrid at --merge-levels")
    parser.add_argument(
        "--merge-levels", type=int, nargs="+", default=[0], help="0: full tree; positive: tree then sequential tail"
    )
    args = parser.parse_args()
    if min(*args.batch_sizes, args.heads, args.iterations, args.repeats) < 1 or min(args.segments) < 0:
        parser.error("batch/head/iteration/repeat counts must be positive; segment counts nonnegative")
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.ops.load_library(str(args.library.resolve()))
    root = Path(__file__).resolve().parents[2]
    impl = load_source("cached_ascendc", root / "vllm_ascend/hypic/compose_ascendc.py")
    baseline = None
    if args.baseline_root:
        if args.namespace == "hypic_ascendc":
            parser.error("a same-process v1 comparison requires a distinct candidate namespace")
        if len(args.batch_sizes) * len(args.segments) != 1 or args.iterations * args.repeats > 100:
            parser.error("v1 retains workspace: compare one shape per process with <=100 timed calls per variant")
        torch.ops.load_library(str((args.baseline_root / "libhypic_compose_ascendc.so").resolve()))
        baseline = load_source("cached_ascendc_v1", args.baseline_root / "compose_ascendc.py")
    if args.tests:
        if args.namespace != "hypic_ascendc" or baseline:
            parser.error("regression tests require the default operator namespace without v1 loaded")
        import pytest

        test = root / "tests/ut/hypic/test_compose_ascendc_npu.py"
        code = pytest.main(["-q", f"--confcutdir={test.parent}", str(test)])
        if code:
            raise SystemExit(code)
    if args.triton:
        import hypic_compose_bench as old_bench
        import hypic_cube_batch as cube

        old_bench.bootstrap()
        from vllm_ascend.hypic.compose import ComposeStep
        from vllm_ascend.hypic.compose_triton import launch_compose

    for batch in args.batch_sizes:
        for count in args.segments:
            s, t, slots = inputs(batch, count, args.heads, args.seed)
            saved_s, saved_t = s.clone(), t.clone()
            expected = reference(s, t, slots)
            calls = {}
            if baseline:
                old_plan = baseline.CachedComposeAscendC(s, t, slots)
                calls["v1_full_ms"] = partial(baseline.compose_cached_ascendc, s, t, slots)
                calls["v1_prepared_ms"] = partial(old_plan, s, t)
            for rows in args.tile_rows:
                options = {"tile_rows": rows, "ops_namespace": args.namespace}
                plan = impl.CachedComposeAscendC(s, t, slots, **options)
                label = "auto" if rows == 0 else f"r{rows}"
                calls[f"ascendc_{label}_full_ms"] = partial(impl.compose_cached_ascendc, s, t, slots, **options)
                calls[f"ascendc_{label}_prepared_ms"] = partial(plan, s, t)
                if args.tree:
                    for depth in args.merge_levels:
                        tree_options = {**options, "merge_levels": depth}
                        tree_plan = impl.CachedComposeTreeAscendC(s, t, slots, **tree_options)
                        tree_label = "tree" if depth == 0 else f"hybrid{depth}"
                        calls[f"{tree_label}_{label}_full_ms"] = partial(
                            impl.compose_cached_tree_ascendc, s, t, slots, **tree_options
                        )
                        calls[f"{tree_label}_{label}_prepared_ms"] = partial(tree_plan, s, t)
                if args.single_tree:
                    for depth in args.merge_levels:
                        single_options = {**options, "single_kernel": True, "merge_levels": depth}
                        single_plan = impl.CachedComposeTreeAscendC(s, t, slots, **single_options)
                        single_label = "single_tree" if depth == 0 else f"single_hybrid{depth}"
                        calls[f"{single_label}_{label}_full_ms"] = partial(
                            impl.compose_cached_tree_ascendc, s, t, slots, **single_options
                        )
                        calls[f"{single_label}_{label}_prepared_ms"] = partial(single_plan, s, t)
            if args.triton:
                fresh = [s.new_empty((0, args.heads, 128, 128)) for _ in slots]
                steps = [[ComposeStep(-slot - 1) for slot in seq] for seq in slots]
                calls["triton_vector_ms"] = partial(final_output, partial(launch_compose, fresh, fresh, s, t, steps))
                calls["triton_cube_cached_ms"] = partial(
                    final_output, partial(cube.batch_compose, fresh, fresh, s, t, steps, pack_mode="cached")
                )
            errors = {}
            relative_l2 = {}
            peak_extra_mib = {}
            for name, call in calls.items():
                if args.trace:
                    print(f"VALIDATE batch={batch} segments={count} call={name}", flush=True)
                torch.npu.synchronize()
                allocated_before = torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                actual = call()
                torch.npu.synchronize()
                peak_extra_mib[name] = (torch.npu.max_memory_allocated() - allocated_before) / 2**20
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                errors[name] = (actual - expected).abs().max().item()
                relative_l2[name] = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
            if args.trace:
                print(f"MEASURE batch={batch} segments={count}", flush=True)
            timings = benchmark(calls, args.iterations, args.repeats)
            torch.testing.assert_close(s, saved_s, atol=0, rtol=0)
            torch.testing.assert_close(t, saved_t, atol=0, rtol=0)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "batch": batch,
                        "segments": count,
                        "heads": args.heads,
                        "allocated_mib": torch.npu.memory_allocated() / 2**20,
                        "max_abs": errors,
                        "relative_l2": relative_l2,
                        "peak_extra_mib": peak_extra_mib,
                        **timings,
                        "timing": "round-average wall time; prepared excludes plan setup, not pure kernel time",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
