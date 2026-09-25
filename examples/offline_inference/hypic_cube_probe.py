"""Ascend FP32 Cube recurrence diagnostics; mock tensors, never model weights."""

import argparse
import json

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _step(h, s, t, out, V: tl.constexpr, K: tl.constexpr, M: tl.constexpr, ACC_S: tl.constexpr):
    head = tl.program_id(0)
    rows = tl.program_id(1) * M + tl.arange(0, M)
    keys = tl.arange(0, K)
    offsets = head * V * K + rows[:, None] * K + keys[None, :]
    left = tl.load(h + offsets, rows[:, None] < V, other=0)
    zero = tl.load(s + offsets, rows[:, None] < V, other=0)
    right = tl.load(t + head * K * K + keys[:, None] * K + keys[None, :])
    if ACC_S:
        result = tl.dot(left, right, acc=zero, input_precision="ieee")
    else:
        result = tl.dot(left, right, input_precision="ieee") + zero
    tl.store(out + offsets, result, rows[:, None] < V)


@triton.jit
def _chain(
    s,
    t,
    out,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    ACC_S: tl.constexpr,
    UNROLL: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.program_id(1) * M + tl.arange(0, M)
    keys = tl.arange(0, K)
    offsets = head * V * K + rows[:, None] * K + keys[None, :]
    t_offsets = head * K * K + keys[:, None] * K + keys[None, :]
    state = tl.load(s + offsets, rows[:, None] < V, other=0)
    if UNROLL:
        for index in tl.static_range(1, N):
            zero = tl.load(s + index * H * V * K + offsets, rows[:, None] < V, other=0)
            right = tl.load(t + index * H * K * K + t_offsets)
            if ACC_S:
                state = tl.dot(state, right, acc=zero, input_precision="ieee")
            else:
                state = tl.dot(state, right, input_precision="ieee") + zero
    else:
        for index in range(1, N):
            zero = tl.load(s + index * H * V * K + offsets, rows[:, None] < V, other=0)
            right = tl.load(t + index * H * K * K + t_offsets)
            if ACC_S:
                state = tl.dot(state, right, acc=zero, input_precision="ieee")
            else:
                state = tl.dot(state, right, input_precision="ieee") + zero
    tl.store(out + offsets, state, rows[:, None] < V)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", nargs="+", type=int, default=[2, 4, 16])
    parser.add_argument("--rows", nargs="+", type=int, default=[16])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["step-add", "step-acc", "loop-add", "loop-acc", "unroll-add", "unroll-acc"],
        default=["step-add", "step-acc", "loop-add", "loop-acc", "unroll-add", "unroll-acc"],
    )
    parser.add_argument("--compiler-options", type=json.loads, default={})
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20260925)
    heads, dim, value_dim = 2, 128, 128
    failures = 0
    for count in args.segments:
        for kind in ("identity", "random"):
            s = torch.ones((count, heads, value_dim, dim), device="npu")
            t = torch.eye(dim, device="npu")[None, None].repeat(count, heads, 1, 1)
            if kind == "random":
                s.normal_(std=0.02)
                t.mul_(0.9).add_(torch.randn_like(t) * (0.03 / dim**0.5))
            expected = s[0].clone()
            for index in range(1, count):
                expected = expected @ t[index] + s[index]
            for rows in args.rows:
                for variant in args.variants:
                    out = torch.empty_like(expected)
                    acc_s = variant.endswith("acc")
                    try:
                        if variant.startswith("step"):
                            state = s[0]
                            for index in range(1, count):
                                out = torch.empty_like(expected)
                                compiled = _step[(heads, triton.cdiv(value_dim, rows))](
                                    state,
                                    s[index],
                                    t[index],
                                    out,
                                    value_dim,
                                    dim,
                                    rows,
                                    acc_s,
                                    num_warps=4,
                                    enable_fp_fusion=False,
                                    **args.compiler_options,
                                )
                                state = out
                        else:
                            compiled = _chain[(heads, triton.cdiv(value_dim, rows))](
                                s,
                                t,
                                out,
                                heads,
                                value_dim,
                                dim,
                                rows,
                                count,
                                acc_s,
                                variant.startswith("unroll"),
                                num_warps=4,
                                enable_fp_fusion=False,
                                **args.compiler_options,
                            )
                        torch.npu.synchronize()
                        delta = (out - expected).abs()
                        exact = kind == "identity"
                        passed = torch.allclose(out, expected, atol=0 if exact else 2e-5, rtol=0 if exact else 2e-4)
                        record = dict(
                            variant=variant,
                            segments=count,
                            rows=rows,
                            kind=kind,
                            passed=passed,
                            max_abs=delta.max().item(),
                            first_row=out[0, 0, :8].tolist(),
                            expected_row=expected[0, 0, :8].tolist(),
                            asm_keys=list(compiled.asm),
                            compiler_options=args.compiler_options,
                        )
                    except Exception as error:
                        passed = False
                        record = dict(
                            variant=variant, segments=count, rows=rows, kind=kind, passed=False, error=str(error)
                        )
                    failures += not passed
                    print(json.dumps(record), flush=True)
    print(json.dumps(dict(failures=failures, diagnostic_only=True)), flush=True)


if __name__ == "__main__":
    main()
