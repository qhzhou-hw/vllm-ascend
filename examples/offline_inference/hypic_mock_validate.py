"""Standalone HYPIC mock-weight validation; no model downloads or vLLM build.

Run on the Ascend server, from this checkout:
  python examples/offline_inference/hypic_mock_validate.py --unit-tests
  python examples/offline_inference/hypic_mock_validate.py --npu

The bootstrap bypasses package initializers, not HYPIC execution. NPU mode
uses real sgl-kernel-npu GDN and CANN attention, with eager PyTorch gating.
It validates operators/cache plumbing, NOT a full vLLM engine/model load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap(root):
    # Only the two DeviceOperator methods reached by HYPIC are supplied here.
    # Gating is the same formula on device; scatter calls the native operator.
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(root / "vllm_ascend")]
    sys.modules["vllm_ascend"] = package
    device = ModuleType("vllm_ascend.device")
    device.__path__ = []
    sys.modules["vllm_ascend.device"] = device
    operator = ModuleType("vllm_ascend.device.device_op")

    def gating(log, a, b, bias):
        return (-log.float().exp() * F.softplus(a.float() + bias.float()))[None], b.sigmoid()[None]

    def scatter(**kwargs):
        import torch_npu

        torch_npu.npu_scatter_pa_kv_cache(
            **{key: value.contiguous() for key, value in kwargs.items()}, cache_mode="Norm"
        )

    operator.DeviceOperator = SimpleNamespace(fused_gdn_gating=gating, reshape_and_cache=scatter)
    sys.modules[operator.__name__] = operator


def run_npu(root, doc_len):
    import torch_npu  # noqa: F401

    from vllm_ascend.hypic.attention import forward_hypic_attention, reference_suffix_attention
    from vllm_ascend.hypic.config import HypicConfig
    from vllm_ascend.hypic.gdn import _run_gdn, chunk_gated_delta_rule_npu, forward_hypic_gdn
    from vllm_ascend.hypic.planner import build_plan
    from vllm_ascend.hypic.runtime import HypicBatchContext

    if chunk_gated_delta_rule_npu is None:
        raise RuntimeError("Install sgl-kernel-npu before NPU validation; no reference-kernel fallback is allowed")
    training_path = root.parent / "MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5/block_native_pic.py"
    training = load_module("hypic_mock_training", training_path)
    device, dtype = "npu:0", torch.bfloat16
    torch.npu.set_device(0)
    torch.manual_seed(20260923)
    heads, dim, channels, width = 2, 128, 768, 4
    query_len = 9
    total = doc_len * 2 + query_len
    config = HypicConfig(mode="block_native_pic", chunk_size=doc_len, seam_sink_tokens=0)
    boundaries = [0, doc_len, doc_len * 2, total]
    raw = (torch.randn((total + doc_len, channels)) * 0.15).to(device, dtype)
    a = torch.randn((len(raw), heads)).to(device, dtype) * 0.1
    b = torch.randn_like(a) * 0.1
    weight = (torch.randn((channels, 1, width)) * 0.1).to(device, dtype)
    bias = torch.zeros(channels, device=device, dtype=dtype)
    results = []

    def training_rule(*args, **kwargs):
        # MindSpeed's rule contract returns (output, final_state); the serving
        # NPU entry point additionally returns an internal workspace.
        return chunk_gated_delta_rule_npu(*args, **kwargs)[:2]

    for batch_size in (1, 4):
        slots, ready = {}, {}
        layer = SimpleNamespace(
            prefix="mock.gdn",
            conv1d=SimpleNamespace(weight=weight, bias=bias),
            activation="silu",
            hypic_conv_pool=torch.empty((2, width - 1, channels), device=device, dtype=dtype),
            hypic_zero_state_pool=torch.empty((2, heads, dim, dim), device=device),
            hypic_transition_pool=torch.empty((2, heads, dim, dim), device=device),
            kv_cache=(
                torch.zeros((batch_size, width - 1, channels), device=device, dtype=dtype),
                torch.zeros((batch_size, heads, dim, dim), device=device),
            ),
            A_log=torch.full((heads,), -1.0, device=device),
            dt_bias=torch.zeros(heads, device=device),
            rearrange_mixed_qkv=lambda x: tuple(t.reshape(1, -1, heads, dim) for t in x.chunk(3, dim=-1)),
        )
        cases = {
            "cold": list(range(total)),
            "warm": list(range(total)),
            "reordered": [*range(doc_len, 2 * doc_len), *range(doc_len), *range(2 * doc_len, total)],
            "changed_prefix": [*range(total, total + doc_len), *range(doc_len, total)],
        }
        for case, tokens in cases.items():
            plans = {
                str(row): build_plan(tokens, ready, config, segment_boundaries=boundaries) for row in range(batch_size)
            }
            for row, plan in enumerate(plans.values()):
                for segment in plan["segments"]:
                    if segment["cacheable"] and not segment["hit"]:
                        if segment["hash"] not in slots and len(slots) < 2:
                            slots[segment["hash"]] = len(slots)
                        segment["store"] = row == 0 and segment["hash"] in slots
            context = HypicBatchContext(plans, tuple(plans), SimpleNamespace(lookup=slots.get))
            positions = plans["0"]["query_positions"]
            x, aa, bb = raw[tokens], a[tokens], b[tokens]
            packed = x[positions].repeat(batch_size, 1)
            out = torch.empty((len(packed), heads, dim), device=device, dtype=dtype)
            metadata = SimpleNamespace(
                num_actual_tokens=len(packed), non_spec_state_indices_tensor=torch.arange(batch_size, device=device)
            )
            forward_hypic_gdn(
                layer,
                packed,
                bb[positions].repeat(batch_size, 1),
                aa[positions].repeat(batch_size, 1),
                out,
                context,
                metadata,
            )
            ids = torch.tensor([[0] * doc_len + [1] * doc_len + [2] * query_len], device=device)
            transformed = training.segmented_causal_conv1d(x.T[None], weight, bias, ids)[0].T
            q, k, v = layer.rearrange_mixed_qkv(transformed)
            g, beta = sys.modules["vllm_ascend.device.device_op"].DeviceOperator.fused_gdn_gating(
                layer.A_log, aa, bb, layer.dt_bias
            )
            expected = training.block_native_gated_delta_rule(q, k, v, g, beta, ids, training_rule, True)[0]
            reference = expected[positions].repeat(batch_size, 1, 1)
            difference = (out.float() - reference.float()).abs()
            torch.testing.assert_close(out, reference, atol=1e-5, rtol=0.02)
            for row in range(batch_size):
                torch.testing.assert_close(layer.kv_cache[0][row], x[-(width - 1) :])
            # Independent recurrence through the same per-segment q/k/v checks
            # that composed final state is suitable for native decode.
            state = torch.zeros((1, heads, dim, dim), device=device)
            for start, end in zip(boundaries, boundaries[1:]):
                _, state = _run_gdn(
                    layer,
                    q[:, start:end],
                    k[:, start:end],
                    v[:, start:end],
                    g[:, start:end],
                    beta[:, start:end],
                    state,
                    torch.tensor([0, end - start], device=device, dtype=torch.int32),
                )
            torch.testing.assert_close(layer.kv_cache[1][0], state[0], atol=2e-4, rtol=0.02)
            results.append(
                {
                    "op": "gdn",
                    "batch_size": batch_size,
                    "case": case,
                    "max_abs": difference.max().item(),
                    "mean_abs": difference.mean().item(),
                    "state_max_abs": (layer.kv_cache[1][0] - state[0]).abs().max().item(),
                }
            )
            for segment in plans["0"]["segments"]:
                if segment.get("store", False):
                    ready[segment["hash"]] = tuple(segment["token_ids"])
            print(json.dumps(results[-1]), flush=True)

    # True CANN full-attention execution and KV pool restore with identity RoPE.
    # Nontrivial RoPE relocation is covered by the tensor unit tests.
    qtable = torch.randn((total, heads, dim), device=device, dtype=dtype) * 0.1
    ktable = torch.randn_like(qtable) * 0.1
    vtable = torch.randn_like(qtable) * 0.1
    rotary = SimpleNamespace(
        is_neox_style=True,
        rotary_dim=dim,
        cos_sin_cache=torch.cat((torch.ones((total, dim // 2)), torch.zeros((total, dim // 2))), -1).to(device, dtype),
    )
    attn = SimpleNamespace(
        layer_name="mock.attn",
        hypic_rotary_emb=rotary,
        hypic_key_pool=torch.empty((2, doc_len, heads, dim), device=device, dtype=dtype),
        hypic_value_pool=torch.empty((2, doc_len, heads, dim), device=device, dtype=dtype),
    )
    ready, slots = {}, {}
    for case in ("cold", "warm"):
        plan = build_plan(list(range(total)), ready, config, segment_boundaries=boundaries)
        for i, segment in enumerate(plan["segments"][:-1]):
            slots[segment["hash"]] = i
        context = HypicBatchContext({"r": plan}, ("r",), SimpleNamespace(lookup=slots.get))
        positions = plan["query_positions"]
        out = torch.empty((len(positions), heads * dim), device=device, dtype=dtype)
        num_blocks = (total + 63) // 64
        paged = tuple(torch.zeros((num_blocks, 64, heads, dim), device=device, dtype=dtype) for _ in range(2))
        forward_hypic_attention(
            attn,
            qtable[positions],
            ktable[positions],
            vtable[positions],
            out,
            context,
            scale=dim**-0.5,
            kv_cache=paged,
            attn_metadata=SimpleNamespace(block_tables=torch.arange(num_blocks, device=device)[None]),
        )
        expected_parts = []
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            kv_start = 0 if i == 2 else start
            expected_parts.append(
                reference_suffix_attention(
                    qtable[start:end].cpu(), ktable[kv_start:end].cpu(), vtable[kv_start:end].cpu(), scale=dim**-0.5
                )
            )
        expected = torch.cat(expected_parts)[positions].reshape_as(out.cpu())
        torch.testing.assert_close(out.cpu(), expected, atol=0.002, rtol=0.02)
        if case == "warm":
            torch.testing.assert_close(paged[0].flatten(0, 1)[: 2 * doc_len], ktable[: 2 * doc_len])
            torch.testing.assert_close(paged[1].flatten(0, 1)[: 2 * doc_len], vtable[: 2 * doc_len])
        ready = {s["hash"]: tuple(s["token_ids"]) for s in plan["segments"][:-1]}
        results.append(
            {"op": "attention", "case": case, "max_abs": (out.cpu().float() - expected.float()).abs().max().item()}
        )
        print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"status": "PASS", "checks": len(results), "mock_weights_only": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-tests", action="store_true")
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--doc-len", type=int, default=512)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    bootstrap(root)
    if args.unit_tests:
        import pytest

        folder = root / "tests/ut/hypic"
        raise SystemExit(
            pytest.main(
                [
                    "-q",
                    f"--confcutdir={folder}",
                    str(folder / "test_hypic.py"),
                    str(folder / "test_block_native_pic.py"),
                ]
            )
        )
    if args.npu:
        if args.doc_len <= 0:
            parser.error("--doc-len must be positive")
        run_npu(root, args.doc_len)
    else:
        parser.error("choose --unit-tests or --npu")


if __name__ == "__main__":
    main()
