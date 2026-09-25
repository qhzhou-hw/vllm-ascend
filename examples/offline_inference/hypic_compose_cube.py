"""Mock-only Cube candidates, deliberately not registered as production backends.

Reuse hypic_compose_bench's precision gates and host-inclusive timings. All
operands and accumulators remain FP32; no TF32/BF16 approximation is requested.
"""

import argparse
import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _cube_kernel(
    pointers,
    metadata,
    lengths,
    zero_pool,
    transition_pool,
    final_states,
    HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    KEY_DIM: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    keys = tl.arange(0, BLOCK_K)
    offsets = head * VALUE_DIM * KEY_DIM + rows[:, None] * KEY_DIM + keys[None, :]
    mask = (rows[:, None] < VALUE_DIM) & (keys[None, :] < KEY_DIM)
    t_offsets = head * KEY_DIM * KEY_DIM + keys[:, None] * KEY_DIM + keys[None, :]
    t_mask = (keys[:, None] < KEY_DIM) & (keys[None, :] < KEY_DIM)
    fresh_s = tl.load(pointers + request * 3).to(tl.pointer_type(tl.float32))
    fresh_t = tl.load(pointers + request * 3 + 1).to(tl.pointer_type(tl.float32))
    replay = tl.load(pointers + request * 3 + 2).to(tl.pointer_type(tl.float32))
    count = tl.load(lengths + request).to(tl.int32)
    state = tl.full((BLOCK_V, BLOCK_K), 0, tl.float32)
    if count > 0:
        source = tl.load(metadata + request * MAX_STEPS * 2).to(tl.int32)
        target = tl.load(metadata + request * MAX_STEPS * 2 + 1).to(tl.int32)
        if target >= 0:
            tl.store(replay + target * HEADS * VALUE_DIM * KEY_DIM + offsets, state, mask)
        if source < 0:
            state = tl.load(zero_pool + (-source - 1) * HEADS * VALUE_DIM * KEY_DIM + offsets, mask, other=0)
        else:
            state = tl.load(fresh_s + source * HEADS * VALUE_DIM * KEY_DIM + offsets, mask, other=0)
    for index in range(1, count):
        source = tl.load(metadata + (request * MAX_STEPS + index) * 2).to(tl.int32)
        target = tl.load(metadata + (request * MAX_STEPS + index) * 2 + 1).to(tl.int32)
        if target >= 0:
            tl.store(replay + target * HEADS * VALUE_DIM * KEY_DIM + offsets, state, mask)
        if source < 0:
            zero = tl.load(zero_pool + (-source - 1) * HEADS * VALUE_DIM * KEY_DIM + offsets, mask, other=0)
            transition = tl.load(
                transition_pool + (-source - 1) * HEADS * KEY_DIM * KEY_DIM + t_offsets, t_mask, other=0
            )
        else:
            zero = tl.load(fresh_s + source * HEADS * VALUE_DIM * KEY_DIM + offsets, mask, other=0)
            transition = tl.load(fresh_t + source * HEADS * KEY_DIM * KEY_DIM + t_offsets, t_mask, other=0)
        state = tl.dot(state, transition, input_precision="ieee") + zero
    tl.store(final_states + request * HEADS * VALUE_DIM * KEY_DIM + offsets, state, mask)


@triton.jit
def _cube_step_kernel(
    pointers,
    metadata,
    lengths,
    zero_pool,
    transition_pool,
    final_states,
    previous,
    HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    KEY_DIM: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    STEP: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    keys = tl.arange(0, BLOCK_K)
    offsets = head * VALUE_DIM * KEY_DIM + rows[:, None] * KEY_DIM + keys[None, :]
    mask = (rows[:, None] < VALUE_DIM) & (keys[None, :] < KEY_DIM)
    final_offsets = request * HEADS * VALUE_DIM * KEY_DIM + offsets
    count = tl.load(lengths + request).to(tl.int32)
    if STEP == 0:
        state = tl.full((BLOCK_V, BLOCK_K), 0, tl.float32)
    else:
        state = tl.load(previous + final_offsets, mask, other=0)
    if count > STEP:
        source = tl.load(metadata + (request * MAX_STEPS + STEP) * 2).to(tl.int32)
        target = tl.load(metadata + (request * MAX_STEPS + STEP) * 2 + 1).to(tl.int32)
        fresh_s = tl.load(pointers + request * 3).to(tl.pointer_type(tl.float32))
        fresh_t = tl.load(pointers + request * 3 + 1).to(tl.pointer_type(tl.float32))
        replay = tl.load(pointers + request * 3 + 2).to(tl.pointer_type(tl.float32))
        if target >= 0:
            tl.store(replay + target * HEADS * VALUE_DIM * KEY_DIM + offsets, state, mask)
        if source < 0:
            zero = tl.load(zero_pool + (-source - 1) * HEADS * VALUE_DIM * KEY_DIM + offsets, mask, other=0)
        else:
            zero = tl.load(fresh_s + source * HEADS * VALUE_DIM * KEY_DIM + offsets, mask, other=0)
        if STEP == 0:
            state = zero
        else:
            t_offsets = head * KEY_DIM * KEY_DIM + keys[:, None] * KEY_DIM + keys[None, :]
            t_mask = (keys[:, None] < KEY_DIM) & (keys[None, :] < KEY_DIM)
            if source < 0:
                transition = tl.load(
                    transition_pool + (-source - 1) * HEADS * KEY_DIM * KEY_DIM + t_offsets, t_mask, other=0
                )
            else:
                transition = tl.load(fresh_t + source * HEADS * KEY_DIM * KEY_DIM + t_offsets, t_mask, other=0)
            state = tl.dot(state, transition, input_precision="ieee") + zero
    tl.store(final_states + final_offsets, state, mask)


class StepLaunch:
    """Benchmark adapter; reuse the production metadata/validation unmodified."""

    def __getitem__(self, grid):
        def run(pointers, metadata, lengths, zero_pool, transition_pool, final_states, **kwargs):
            scratch = torch.empty_like(final_states)
            count = kwargs["MAX_STEPS"]
            # Arrange parity so the last step always writes the caller's output.
            previous, destination = (scratch, final_states) if count % 2 else (final_states, scratch)
            for step in range(count):
                _cube_step_kernel[grid](
                    pointers,
                    metadata,
                    lengths,
                    zero_pool,
                    transition_pool,
                    destination,
                    previous,
                    **kwargs,
                    STEP=step,
                )
                previous, destination = destination, previous

        return run


def direct_compose(zero_states, transitions, zero_pool, transition_pool, steps, rows):
    """Control: direct tensor arguments, no in-kernel pointer-table branches."""
    from hypic_cube_probe import _step

    first = zero_states[0]
    _, heads, value_dim, key_dim = first.shape
    if key_dim != 128:
        raise ValueError("the direct diagnostic currently requires key_dim=128")
    replays, finals = [], []
    for zeros, transforms, sequence in zip(zero_states, transitions, steps):
        replay = torch.zeros_like(zeros)
        state = torch.zeros((heads, value_dim, key_dim), dtype=first.dtype, device=first.device)
        for index, step in enumerate(sequence):
            if step.replay >= 0:
                replay[step.replay].copy_(state)
            zero = zero_pool[-step.source - 1] if step.source < 0 else zeros[step.source]
            transition = transition_pool[-step.source - 1] if step.source < 0 else transforms[step.source]
            if index == 0:
                state = zero.clone()
            else:
                output = torch.empty_like(state)
                _step[(heads, triton.cdiv(value_dim, rows))](
                    state,
                    zero,
                    transition,
                    output,
                    value_dim,
                    key_dim,
                    rows,
                    False,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                state = output
        replays.append(replay)
        finals.append(state)
    return replays, torch.stack(finals)


@triton.jit
def _packed_chain(
    s,
    t,
    history,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    HISTORY: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.program_id(1) * M + tl.arange(0, M)
    keys = tl.arange(0, K)
    offsets = head * V * K + rows[:, None] * K + keys[None, :]
    mask = rows[:, None] < V
    transition_offsets = head * K * K + keys[:, None] * K + keys[None, :]
    tl.store(history + offsets, tl.full((M, K), 0, tl.float32), mask)
    state = tl.load(s + offsets, mask, other=0)
    tl.store(history + H * V * K + offsets, state, mask)
    for index in range(1, N):
        zero = tl.load(s + index * H * V * K + offsets, mask, other=0)
        transition = tl.load(t + index * H * K * K + transition_offsets)
        state = tl.dot(state, transition, input_precision="ieee") + zero
        tl.store(history + ((index + 1) % HISTORY) * H * V * K + offsets, state, mask)


def packed_compose(zero_states, transitions, zero_pool, transition_pool, steps, rows):
    """Control: contiguous operands avoid dynamic cache-pointer/control flow.

    Packing and prefix-history workspace are intentionally timed. This is
    not a zero-copy replacement for the production segment-pool implementation.
    """
    first = zero_states[0]
    _, heads, value_dim, key_dim = first.shape
    if key_dim != 128:
        raise ValueError("the packed diagnostic currently requires key_dim=128")
    replays, finals = [], []
    for zeros, transforms, sequence in zip(zero_states, transitions, steps):
        replay = torch.zeros_like(zeros)
        if not sequence:
            replays.append(replay)
            finals.append(torch.zeros((heads, value_dim, key_dim), device=first.device, dtype=first.dtype))
            continue
        packed_s = torch.stack(
            [zero_pool[-step.source - 1] if step.source < 0 else zeros[step.source] for step in sequence]
        )
        packed_t = torch.stack(
            [transition_pool[-step.source - 1] if step.source < 0 else transforms[step.source] for step in sequence]
        )
        history_count = len(sequence) + 1 if any(step.replay >= 0 for step in sequence[:-1]) else 2
        history = torch.empty((history_count, heads, value_dim, key_dim), device=first.device, dtype=first.dtype)
        _packed_chain[(heads, triton.cdiv(value_dim, rows))](
            packed_s,
            packed_t,
            history,
            heads,
            value_dim,
            key_dim,
            len(sequence),
            rows,
            history_count,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for index, step in enumerate(sequence):
            if step.replay >= 0:
                replay[step.replay].copy_(history[index % history_count])
        replays.append(replay)
        finals.append(history[len(sequence) % history_count].clone())
        # No previous request's packed operands need to survive the next stack.
        del packed_s, packed_t, history
    return replays, torch.stack(finals)


def check_empty_cache_only():
    """Regression for the GDN fast path: no fresh S/T, including no Documents."""
    from vllm_ascend.hypic.compose import ComposeStep, compose_fused_batch

    fresh = torch.empty((0, 2, 128, 128), device="npu:0")
    pool = torch.ones((2, 2, 128, 128), device="npu:0")
    transitions = torch.eye(128, device="npu:0")[None, None].repeat(2, 2, 1, 1)
    sequences = [[], [ComposeStep(-1)], [ComposeStep(-2), ComposeStep(-1)], [ComposeStep(-1)] * 5]
    replay, final = compose_fused_batch([fresh] * 4, [fresh] * 4, pool, transitions, sequences)
    for index, sequence in enumerate(sequences):
        assert replay[index].shape == fresh.shape
        torch.testing.assert_close(final[index], torch.full_like(pool[0], len(sequence)), atol=0, rtol=0)
    torch.testing.assert_close(pool, torch.ones_like(pool), atol=0, rtol=0)
    print(json.dumps({"status": "PASS", "check": "empty-cache-only", "cases": 4}), flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cube-variant",
        choices=["persistent", "step", "direct", "packed", "batch", "window", "native", "indexed", "output", "cached"],
        default="persistent",
    )
    parser.add_argument("--cube-rows", type=int, choices=[16, 32, 64, 128], default=16)
    parser.add_argument("--cube-target", choices=["compose", "mock"], default="compose")
    parser.add_argument("--cube-window", type=int, default=16)
    parser.add_argument("--cube-baseline-module", help="Importable frozen native Cube module for same-process A/B")
    parser.add_argument("--cube-indexed-baseline-module", help="Frozen indexed Cube module for same-process A/B")
    parser.add_argument("--cube-output-baseline-module", help="Frozen output Cube module for same-process A/B")
    parser.add_argument("--cube-output-api-module", help="Optional frozen validation API for the output baseline")
    args, remaining = parser.parse_known_args()
    if args.cube_target == "mock":
        import hypic_mock_validate as bench

        backend_index = remaining.index("--compose-backend") + 1 if "--compose-backend" in remaining else len(remaining)
        if backend_index >= len(remaining) or remaining[backend_index] != "triton":
            parser.error("mock Cube validation requires --compose-backend triton")
    else:
        import hypic_compose_bench as bench

    original_bootstrap = bench.bootstrap
    if args.cube_target == "compose" and args.cube_variant in (
        "batch",
        "window",
        "native",
        "indexed",
        "output",
        "cached",
    ):
        from functools import partial

        original_run_case = bench.run_case

        def run_case(case_args, *case_values):
            case_args.packed_launch = partial(packed_compose, rows=128)
            if args.cube_baseline_module:
                import importlib

                baseline = importlib.import_module(args.cube_baseline_module)
                case_args.native_launch = partial(baseline.batch_compose, rows=128, pack_mode="native")
            if args.cube_indexed_baseline_module:
                import importlib

                baseline = importlib.import_module(args.cube_indexed_baseline_module)
                case_args.indexed_launch = partial(baseline.batch_compose, rows=128, pack_mode="indexed")
            if args.cube_output_baseline_module:
                import importlib

                baseline = importlib.import_module(args.cube_output_baseline_module)
                case_args.output_launch = partial(baseline.batch_compose, rows=128, pack_mode="output")
                if args.cube_output_api_module:
                    case_args.output_api = importlib.import_module(args.cube_output_api_module).compose_fused_batch
            return original_run_case(case_args, *case_values)

        bench.run_case = run_case

    def bootstrap(*bootstrap_args):
        original_bootstrap(*bootstrap_args)
        from vllm_ascend.hypic import compose_triton

        if args.cube_variant in ("direct", "packed", "batch", "window", "native", "indexed", "output", "cached"):
            from functools import partial

            candidate = direct_compose if args.cube_variant == "direct" else packed_compose
            if args.cube_variant in ("batch", "native", "indexed", "output", "cached"):
                from hypic_cube_batch import batch_compose

                candidate = batch_compose
                if args.cube_variant in ("native", "indexed", "output", "cached"):
                    candidate = partial(batch_compose, pack_mode=args.cube_variant)
            if args.cube_variant == "window":
                from hypic_cube_batch import window_compose

                candidate = partial(window_compose, window=args.cube_window)
            compose_triton.launch_compose = partial(candidate, rows=args.cube_rows)
        else:
            compose_triton._compose_kernel = _cube_kernel if args.cube_variant == "persistent" else StepLaunch()
            compose_triton._VALUE_TILE = args.cube_rows
        if args.cube_target == "compose":
            check_empty_cache_only()

    bench.bootstrap = bootstrap
    print(
        json.dumps(
            {
                "cube_variant": args.cube_variant,
                "cube_rows": args.cube_rows,
                "cube_target": args.cube_target,
                "experimental_only": True,
            }
        ),
        flush=True,
    )
    sys.argv = [sys.argv[0], *remaining]
    bench.main()


if __name__ == "__main__":
    main()
