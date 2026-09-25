"""Ascend-only FP32 dot diagnostic; no model downloads or engine required."""

import json

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def dot_probe(a, b, out, M: tl.constexpr, K: tl.constexpr, REDUCE: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, K)
    inner = tl.arange(0, REDUCE)
    acc = tl.full((M, K), 0, tl.float32)
    for start in range(0, K, REDUCE):
        left = tl.load(a + rows[:, None] * K + start + inner[None, :])
        right = tl.load(b + (start + inner[:, None]) * K + cols[None, :])
        acc += tl.dot(left, right, input_precision="ieee")
    tl.store(out + rows[:, None] * K + cols[None, :], acc)


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260924)
    for rows in (16, 32):
        for reduce in (32, 64, 128):
            for kind in ("identity", "random"):
                a = torch.ones((rows, 128), device="npu")
                b = torch.eye(128, device="npu")
                if kind == "random":
                    a.normal_(std=0.1)
                    b.normal_(std=0.1)
                out = torch.empty_like(a)
                dot_probe[(1,)](a, b, out, rows, 128, reduce, num_warps=4, enable_fp_fusion=False)
                torch.npu.synchronize()
                expected = a @ b
                difference = (out - expected).abs().max().item()
                print(json.dumps({"rows": rows, "reduce": reduce, "kind": kind, "max_abs": difference}), flush=True)


if __name__ == "__main__":
    main()
