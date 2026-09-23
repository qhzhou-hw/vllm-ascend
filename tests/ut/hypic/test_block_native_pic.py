"""Training-graph parity and cold/warm block-native PIC regressions.

Numerical tests use a small PyTorch recurrent oracle instead of the NPU kernel.
The direct MindSpeed comparison additionally requires a sibling MindSpeed-MM
checkout; it skips explicitly when that optional training source is absent.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.hypic.attention import forward_hypic_attention
from vllm_ascend.hypic.config import HypicConfig
from vllm_ascend.hypic.gdn import forward_hypic_gdn
from vllm_ascend.hypic.planner import build_plan, validate_plan
from vllm_ascend.hypic.runtime import HypicBatchContext


def _config():
    return HypicConfig(mode="block_native_pic", chunk_size=4, seam_sink_tokens=0)


def _rule(q, k, v, g, beta, initial_state=None, **kwargs):
    """Independent K,V-layout recurrence, matching the training convention."""
    q = F.normalize(q.float(), dim=-1, eps=1e-6) * q.shape[-1] ** -0.5
    k = F.normalize(k.float(), dim=-1, eps=1e-6)
    state = (
        q.new_zeros((q.shape[0], q.shape[2], q.shape[-1], v.shape[-1]))
        if initial_state is None
        else initial_state.clone()
    )
    outputs = []
    for index in range(q.shape[1]):
        state = state * g[:, index].exp()[..., None, None]
        memory = (state * k[:, index, :, :, None]).sum(-2)
        delta = (v[:, index] - memory) * beta[:, index, :, None]
        state = state + k[:, index, :, :, None] * delta[..., None, :]
        outputs.append((state * q[:, index, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1), state


def _kernel(layer, q, k, v, g, beta, initial, cu):
    outputs, states = [], []
    bounds = cu.tolist()
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        out, state = _rule(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial_state=initial[index : index + 1].transpose(-1, -2),
        )
        outputs.append(out)
        states.append(state[0].transpose(-1, -2))
    return torch.cat(outputs, dim=1), torch.stack(states)


def _layer():
    torch.manual_seed(37)
    return SimpleNamespace(
        prefix="gdn",
        conv1d=SimpleNamespace(weight=torch.randn((6, 1, 3)) * 0.2, bias=torch.randn(6) * 0.1),
        activation="silu",
        hypic_conv_pool=torch.empty((2, 2, 6)),
        hypic_zero_state_pool=torch.empty((2, 1, 2, 2)),
        hypic_transition_pool=torch.empty((2, 1, 2, 2)),
        kv_cache=(torch.zeros((1, 2, 6)), torch.zeros((1, 1, 2, 2))),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=lambda x: tuple(part.reshape(1, -1, 1, 2) for part in x.chunk(3, dim=-1)),
    )


def _expected_gdn(layer, raw, a, b):
    # Independent document conv/GDN outputs; only Query uses the accumulated
    # document state. Carrying state for this oracle's aggregate is equivalent
    # to S/T composition, but does not compute a transition matrix at all.
    outputs = []
    accumulated = None
    transformed = []
    for index, (start, end) in enumerate(((0, 2), (2, 4), (4, 6))):
        convolution = F.conv1d(F.pad(raw[start:end].T[None], (2, 0)), layer.conv1d.weight, layer.conv1d.bias, groups=6)
        x = F.silu(convolution).squeeze(0).T
        transformed.append(x)
        q, k, v = layer.rearrange_mixed_qkv(x)
        args = (q, k, v, a[start:end][None], b[start:end][None])
        out, _ = _rule(*args, initial_state=accumulated if index == 2 else None)
        _, accumulated = _rule(*args, initial_state=accumulated)
        outputs.append(out)
    return torch.cat(outputs, dim=1)[0], accumulated[0].transpose(-1, -2), torch.cat(transformed)


def test_block_native_config_and_boundaries():
    with pytest.raises(ValueError, match="seam_sink_tokens=0"):
        HypicConfig.from_dict({"mode": "block_native_pic"})
    config = _config()
    with pytest.raises(ValueError, match="explicit"):
        build_plan(list(range(6)), {}, config)
    for bounds in ([0, 6], [1, 3, 6], [0, 2, 5], [0, 2, 2, 6], [0, 4, 3, 6], [0, 2.5, 6]):
        with pytest.raises(ValueError):
            build_plan(list(range(6)), {}, config, segment_boundaries=bounds)
    with pytest.raises(ValueError, match="document exceeds chunk_size"):
        build_plan(list(range(8)), {}, config, segment_boundaries=[0, 5, 8])
    # A long Query stays intact; it does not occupy a document cache slot.
    plan = build_plan(list(range(10)), {}, config, segment_boundaries=[0, 2, 10])
    validate_plan(plan)
    assert len(plan["segments"]) == 2
    assert plan["reset_conv_history"] is True
    legacy = build_plan(list(range(10)), {}, HypicConfig(chunk_size=4, seam_sink_tokens=0))
    assert plan["mode"] != legacy["mode"]
    corrupted = {**plan, "reset_conv_history": False}
    with pytest.raises(ValueError, match="convolution reset"):
        validate_plan(corrupted)


def test_block_native_and_legacy_reset_have_separate_caches():
    tokens = list(range(6))
    native = build_plan(tokens, {}, _config(), segment_boundaries=[0, 2, 4, 6])
    ready = {s["hash"]: tuple(s["token_ids"]) for s in native["segments"][:-1]}
    legacy = build_plan(
        tokens,
        ready,
        HypicConfig(chunk_size=4, seam_sink_tokens=0, reset_conv_history=True),
        segment_boundaries=[0, 2, 4, 6],
    )
    assert not any(s["hit"] for s in legacy["segments"])


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4, 5], [2, 3, 0, 1, 4, 5], [6, 7, 2, 3, 4, 5]])
def test_block_native_gdn_cold_warm_reorder_and_changed_prefix(monkeypatch, order):
    layer = _layer()
    monkeypatch.setattr("vllm_ascend.hypic.gdn._run_gdn", _kernel)
    monkeypatch.setattr(
        "vllm_ascend.hypic.gdn.DeviceOperator.fused_gdn_gating", lambda log, a, b, bias: (a[None], b[None])
    )
    torch.manual_seed(11)
    raw_table = torch.randn((8, 6))
    a_table, b_table = -torch.rand((8, 1)), torch.rand((8, 1))
    slots, ready = {}, {}
    for tokens in (list(range(6)), order):
        plan = build_plan(tokens, ready, _config(), segment_boundaries=[0, 2, 4, 6])
        for segment in plan["segments"]:
            if segment["cacheable"] and segment["hash"] not in slots:
                if len(slots) < 2:
                    slots[segment["hash"]] = len(slots)
                else:
                    segment["store"] = False  # changed prefix is compute-only
        context = HypicBatchContext({"r": plan}, ("r",), SimpleNamespace(lookup=slots.get))
        raw, a, b = raw_table[tokens], a_table[tokens], b_table[tokens]
        positions = plan["query_positions"]
        out = torch.empty((len(positions), 1, 2))
        metadata = SimpleNamespace(num_actual_tokens=len(positions), non_spec_state_indices_tensor=torch.tensor([0]))
        forward_hypic_gdn(layer, raw[positions], b[positions], a[positions], out, context, metadata)
        expected, final_state, _ = _expected_gdn(layer, raw, a, b)
        torch.testing.assert_close(out, expected[positions], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(layer.kv_cache[1][0], final_state, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(layer.kv_cache[0][0], raw[-2:])
        for segment in plan["segments"]:
            if segment.get("store", segment["cacheable"]):
                ready[segment["hash"]] = tuple(segment["token_ids"])


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4, 5], [2, 3, 0, 1, 4, 5], [6, 7, 2, 3, 4, 5]])
def test_block_native_attention_cold_warm_and_document_isolation(monkeypatch, order):
    torch.manual_seed(23)
    q_table, k_table, v_table = (torch.randn((8, 1, 2)) for _ in range(3))
    angles = torch.arange(6, dtype=torch.float32) * 0.17
    rotary = SimpleNamespace(
        is_neox_style=True, rotary_dim=2, cos_sin_cache=torch.stack((angles.cos(), angles.sin()), dim=-1)
    )
    layer = SimpleNamespace(
        layer_name="attn",
        hypic_rotary_emb=rotary,
        hypic_key_pool=torch.empty((2, 4, 1, 2)),
        hypic_value_pool=torch.empty((2, 4, 1, 2)),
    )
    hydrated = []
    monkeypatch.setattr(
        "vllm_ascend.hypic.attention._hydrate_paged_kv_cache",
        lambda key, value, **kwargs: hydrated.append((key.clone(), value.clone(), kwargs["start"])),
    )

    def rotate(x):
        half = torch.stack((-x[..., 1], x[..., 0]), -1)
        return x * angles.cos()[:, None, None] + half * angles.sin()[:, None, None]

    slots, ready = {}, {}
    for tokens in (list(range(6)), order):
        plan = build_plan(tokens, ready, _config(), segment_boundaries=[0, 2, 4, 6])
        for s in plan["segments"]:
            if s["cacheable"] and s["hash"] not in slots:
                if len(slots) < 2:
                    slots[s["hash"]] = len(slots)
                else:
                    s["store"] = False
        q, k, v = rotate(q_table[tokens]), rotate(k_table[tokens]), v_table[tokens]
        positions = plan["query_positions"]
        context = HypicBatchContext({"r": plan}, ("r",), SimpleNamespace(lookup=slots.get))
        out = torch.empty((len(positions), 2))
        hydrated.clear()
        forward_hypic_attention(
            layer,
            q[positions],
            k[positions],
            v[positions],
            out,
            context,
            scale=2**-0.5,
            kv_cache=(),
            attn_metadata=None,
        )
        ids = torch.tensor([0, 0, 1, 1, 2, 2])
        allowed = ((ids[:, None] == ids[None, :]) | (ids[:, None] == 2)) & torch.ones((6, 6), dtype=torch.bool).tril()
        scores = torch.einsum("qhd,khd->hqk", q, k) * 2**-0.5
        expected = torch.einsum("hqk,khd->qhd", scores.masked_fill(~allowed[None], -torch.inf).softmax(-1), v)
        torch.testing.assert_close(out, expected[positions].reshape(-1, 2), atol=2e-5, rtol=2e-5)
        for keys, values, start in hydrated:
            torch.testing.assert_close(keys, k[start : start + len(keys)], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(values, v[start : start + len(values)])
        for s in plan["segments"]:
            if s.get("store", s["cacheable"]):
                ready[s["hash"]] = tuple(s["token_ids"])


def test_direct_mindspeed_training_primitives_match(monkeypatch):
    source = Path(__file__).resolve().parents[4] / "MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5/block_native_pic.py"
    if not source.is_file():
        pytest.skip("Direct training parity requires a sibling MindSpeed-MM checkout")
    spec = importlib.util.spec_from_file_location("training_block_native_pic", source)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    layer = _layer()
    torch.manual_seed(91)
    raw, a, b = torch.randn((6, 6)), -torch.rand((6, 1)), torch.rand((6, 1))
    ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    conv = training.segmented_causal_conv1d(raw.T[None], layer.conv1d.weight, layer.conv1d.bias, ids)[0].T
    q, k, v = layer.rearrange_mixed_qkv(conv)
    expected = training.block_native_gated_delta_rule(q, k, v, a[None], b[None], ids, _rule, False)[0]
    oracle, _, oracle_conv = _expected_gdn(layer, raw, a, b)
    torch.testing.assert_close(conv, oracle_conv, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(expected, oracle, atol=2e-5, rtol=2e-5)
    monkeypatch.setattr("vllm_ascend.hypic.gdn._run_gdn", _kernel)
    monkeypatch.setattr(
        "vllm_ascend.hypic.gdn.DeviceOperator.fused_gdn_gating", lambda log, a, b, bias: (a[None], b[None])
    )
    plan = build_plan(list(range(6)), {}, _config(), segment_boundaries=[0, 2, 4, 6])
    slots = {s["hash"]: i for i, s in enumerate(plan["segments"][:-1])}
    context = HypicBatchContext({"r": plan}, ("r",), SimpleNamespace(lookup=slots.get))
    output = torch.empty((6, 1, 2))
    forward_hypic_gdn(
        layer,
        raw,
        b,
        a,
        output,
        context,
        SimpleNamespace(num_actual_tokens=6, non_spec_state_indices_tensor=torch.tensor([0])),
    )
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
    mask = training.apply_block_native_attention_mask(None, ids, dtype=torch.float32)
    block_allowed = (ids[0, :, None] == ids[0, None, :]) | (ids[0, :, None] == 2)
    expected_mask = block_allowed & torch.ones((6, 6), dtype=torch.bool).tril()
    assert torch.equal(mask[0, 0].eq(0), expected_mask)
