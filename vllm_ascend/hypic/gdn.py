"""Transition/recompute HYPIC execution for Ascend Gated DeltaNet layers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
except ImportError:  # Optional until HYPIC is enabled.
    chunk_gated_delta_rule_npu = None

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.hypic.cache import LayerSegmentState
from vllm_ascend.hypic.runtime import HypicBatchContext


def _causal_conv(layer: Any, raw: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = int(layer.conv1d.weight.shape[-1])
    combined = torch.cat((history, raw), dim=0)
    convolution = F.conv1d(
        combined.transpose(0, 1).unsqueeze(0),
        layer.conv1d.weight,
        bias=layer.conv1d.bias,
        groups=combined.shape[-1],
    )
    output = convolution.squeeze(0).transpose(0, 1)
    if layer.activation:
        output = F.silu(output)
    return output, combined[-(width - 1) :].detach().clone()


def _run_gdn(
    layer: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_gated_delta_rule_npu is None:
        raise RuntimeError("HYPIC GDN requires sgl-kernel-npu 2026.5.1 or newer on Ascend")
    output, final_state, _ = chunk_gated_delta_rule_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.transpose(-1, -2).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return output, final_state.transpose(-1, -2).contiguous()


def _compose(state: torch.Tensor, transition: torch.Tensor, zero_state: torch.Tensor) -> torch.Tensor:
    return torch.bmm(state.float(), transition.float()) + zero_state.float()


def forward_hypic_gdn(
    layer: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    context: HypicBatchContext,
    attn_metadata: Any,
) -> None:
    """Execute HYPIC S/T composition and seeded replay for one GDN layer."""
    if len(context.request_ids) != 1:
        raise RuntimeError("HYPIC GDN currently requires max_num_seqs=1")
    request_id = context.request_ids[0]
    plan = context.plans[request_id]
    layer_name = str(layer.prefix)
    num_tokens = len(plan["query_positions"])
    mixed_qkv = mixed_qkv[:num_tokens]
    a = a[:num_tokens]
    b = b[:num_tokens]

    width = int(layer.conv1d.weight.shape[-1])
    history = mixed_qkv.new_zeros((width - 1, mixed_qkv.shape[-1]))
    transformed_parts: list[torch.Tensor] = []
    a_parts: list[torch.Tensor] = []
    b_parts: list[torch.Tensor] = []
    units: list[dict[str, Any]] = []
    cache_writes: list[tuple[dict[str, Any], int, torch.Tensor]] = []
    packed_offset = 0

    for segment in plan["segments"]:
        length = int(segment["end"]) - int(segment["start"])
        hit = bool(segment["hit"])
        query_len = int(segment["seam"]) if hit else length
        raw = mixed_qkv[packed_offset : packed_offset + query_len]
        part_a = a[packed_offset : packed_offset + query_len]
        part_b = b[packed_offset : packed_offset + query_len]
        cached = context.cache.get(segment["hash"], layer_name) if hit else None
        if hit and cached is None:
            raise RuntimeError(
                f"HYPIC scheduler/worker GDN cache divergence for segment {segment['hash']} at {layer_name}"
            )

        if query_len:
            transformed, raw_tail = _causal_conv(layer, raw, history)
            transformed_parts.append(transformed)
            a_parts.append(part_a)
            b_parts.append(part_b)
        else:
            raw_tail = history

        if hit:
            if query_len:
                units.append({"segment": segment, "kind": "seam", "length": query_len})
            history = cached.conv_tail
        else:
            history = raw_tail
            seam = int(segment["recompute_seam"])
            if seam:
                units.append({"segment": segment, "kind": "seam", "length": seam})
                interior_index = len(units)
                units.append({"segment": segment, "kind": "interior", "length": length - seam})
                cache_writes.append((segment, interior_index, history))
            else:
                unit_index = len(units)
                units.append({"segment": segment, "kind": "full", "length": length})
                if segment["cacheable"]:
                    cache_writes.append((segment, unit_index, history))
        packed_offset += query_len

    if packed_offset != num_tokens:
        raise RuntimeError("HYPIC GDN packed-token plan mismatch")
    transformed = torch.cat(transformed_parts, dim=0)
    q, k, v = layer.rearrange_mixed_qkv(transformed)
    joined_a = torch.cat(a_parts, dim=0)
    joined_b = torch.cat(b_parts, dim=0)
    g, beta = DeviceOperator.fused_gdn_gating(layer.A_log, joined_a, joined_b, layer.dt_bias)
    lengths = [int(unit["length"]) for unit in units]
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=q.device,
    )
    num_units = len(units)
    num_heads = v.shape[2]
    value_dim = v.shape[-1]
    key_dim = k.shape[-1]
    zero = torch.zeros(
        (num_units, num_heads, value_dim, key_dim),
        dtype=torch.float32,
        device=q.device,
    )
    _, zero_states = _run_gdn(layer, q, k, v, g, beta, zero, cu_seqlens)
    identity = torch.zeros_like(zero)
    identity.diagonal(dim1=-2, dim2=-1).fill_(1)
    v_zero = v.new_zeros((1, v.shape[1], num_heads, key_dim))
    _, transitions = _run_gdn(layer, q, k, v_zero, g, beta, identity, cu_seqlens)
    zero_states = zero_states.float()
    transitions = transitions.float()

    for segment, unit_index, conv_tail in cache_writes:
        context.cache.put(
            segment["hash"],
            layer_name,
            LayerSegmentState(
                conv_tail=conv_tail,
                zero_state=zero_states[unit_index].detach().clone(),
                transition=transitions[unit_index].detach().clone(),
            ),
        )

    accumulated = zero[0]
    replay_initial = torch.zeros_like(zero)
    unit_cursor = 0
    for segment in plan["segments"]:
        hit = bool(segment["hit"])
        if hit and int(segment["seam"]):
            replay_initial[unit_cursor] = accumulated
            accumulated = _compose(
                accumulated,
                transitions[unit_cursor],
                zero_states[unit_cursor],
            )
            unit_cursor += 1
        if hit:
            cached = context.cache.get(segment["hash"], layer_name)
            accumulated = _compose(accumulated, cached.transition, cached.zero_state)
            continue
        number_of_units = 2 if int(segment["recompute_seam"]) else 1
        for _ in range(number_of_units):
            replay_initial[unit_cursor] = accumulated
            accumulated = _compose(
                accumulated,
                transitions[unit_cursor],
                zero_states[unit_cursor],
            )
            unit_cursor += 1

    output, _ = _run_gdn(layer, q, k, v, g, beta, replay_initial, cu_seqlens)
    core_attn_out[:num_tokens] = output.squeeze(0)

    state_indices = attn_metadata.prefill_state_indices
    if state_indices is None or len(state_indices) != 1:
        raise RuntimeError("HYPIC GDN expected exactly one prefill state slot")
    state_index = state_indices[0].to(torch.long)
    layer.kv_cache[1][state_index] = accumulated.to(layer.kv_cache[1].dtype)
    layer.kv_cache[0][state_index] = history.to(layer.kv_cache[0].dtype)
