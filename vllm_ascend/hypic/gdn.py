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
    return output, combined[-(width - 1) :]


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


def _forward_hypic_gdn_request(
    layer: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    context: HypicBatchContext,
    attn_metadata: Any,
    request_id: str,
    request_index: int,
) -> None:
    """Execute HYPIC S/T composition for one request in a packed batch."""
    plan = context.plans[request_id]
    layer_name = str(layer.prefix)
    conv_pool = getattr(layer, "hypic_conv_pool", None)
    zero_state_pool = getattr(layer, "hypic_zero_state_pool", None)
    transition_pool = getattr(layer, "hypic_transition_pool", None)
    if conv_pool is None or zero_state_pool is None or transition_pool is None:
        raise RuntimeError(f"HYPIC static GDN pool is missing for {layer_name}")
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
    cache_writes: list[tuple[dict[str, Any], int, int]] = []
    packed_offset = 0

    for segment in plan["segments"]:
        length = int(segment["end"]) - int(segment["start"])
        hit = bool(segment["hit"])
        query_len = int(segment["seam"]) if hit else length
        raw = mixed_qkv[packed_offset : packed_offset + query_len]
        part_a = a[packed_offset : packed_offset + query_len]
        part_b = b[packed_offset : packed_offset + query_len]
        slot = context.cache.lookup(segment["hash"]) if segment["cacheable"] else None
        if segment["cacheable"] and slot is None:
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
            history = conv_pool[slot]
        else:
            history = raw_tail
            if segment["cacheable"]:
                conv_pool[slot].copy_(history)
                history = conv_pool[slot]
            seam = int(segment["recompute_seam"])
            if seam:
                units.append({"segment": segment, "kind": "seam", "length": seam})
                interior_index = len(units)
                units.append({"segment": segment, "kind": "interior", "length": length - seam})
                cache_writes.append((segment, interior_index, slot))
            else:
                unit_index = len(units)
                units.append({"segment": segment, "kind": "full", "length": length})
                if segment["cacheable"]:
                    cache_writes.append((segment, unit_index, slot))
        packed_offset += query_len

    if packed_offset != num_tokens:
        raise RuntimeError("HYPIC GDN packed-token plan mismatch")
    transformed = torch.cat(transformed_parts, dim=0)
    q, k, v = layer.rearrange_mixed_qkv(transformed)
    joined_a = torch.cat(a_parts, dim=0)
    joined_b = torch.cat(b_parts, dim=0)
    g, beta = DeviceOperator.fused_gdn_gating(layer.A_log, joined_a, joined_b, layer.dt_bias)
    # q/k/v and g/beta own everything needed below. Drop the per-segment
    # convolution outputs and concatenation inputs before allocating FP32 state
    # workspaces for long prompts.
    del transformed, transformed_parts, a_parts, b_parts, joined_a, joined_b
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
    del zero
    v_zero = v.new_zeros((1, v.shape[1], num_heads, key_dim))
    _, transitions = _run_gdn(layer, q, k, v_zero, g, beta, identity, cu_seqlens)
    del identity, v_zero
    zero_states = zero_states.float()
    transitions = transitions.float()

    for _, unit_index, slot in cache_writes:
        zero_state_pool[slot].copy_(zero_states[unit_index])
        transition_pool[slot].copy_(transitions[unit_index])

    accumulated = torch.zeros_like(zero_states[0])
    replay_initial = torch.empty_like(zero_states)
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
            slot = context.cache.lookup(segment["hash"])
            if slot is None:
                raise RuntimeError(
                    f"HYPIC GDN slot disappeared for segment {segment['hash']} at {layer_name}"
                )
            accumulated = _compose(
                accumulated,
                transition_pool[slot],
                zero_state_pool[slot],
            )
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

    state_indices = attn_metadata.prefill_state_indices
    if state_indices is None or request_index >= len(state_indices):
        raise RuntimeError(
            f"HYPIC GDN state metadata is missing request {request_index}"
        )
    state_index = state_indices[request_index].to(torch.long)
    layer.kv_cache[1][state_index] = accumulated.to(layer.kv_cache[1].dtype)
    layer.kv_cache[0][state_index] = history.to(layer.kv_cache[0].dtype)

    # Seeded replay only consumes q/k/v/g/beta and replay_initial. Release the
    # two per-segment FP32 result sets before entering the third GDN pass so the
    # kernel workspace can reuse their allocator blocks instead of pushing the
    # NPU to its memory limit.
    del zero_states, transitions, cache_writes, accumulated

    output, _ = _run_gdn(layer, q, k, v, g, beta, replay_initial, cu_seqlens)
    core_attn_out[:num_tokens] = output.squeeze(0)


def forward_hypic_gdn(
    layer: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    context: HypicBatchContext,
    attn_metadata: Any,
) -> None:
    """Execute independent HYPIC state composition for a packed request batch."""
    packed_offset = 0
    for request_index, request_id in enumerate(context.request_ids):
        plan = context.plans[request_id]
        num_tokens = len(plan["query_positions"])
        packed_end = packed_offset + num_tokens
        _forward_hypic_gdn_request(
            layer,
            mixed_qkv[packed_offset:packed_end],
            b[packed_offset:packed_end],
            a[packed_offset:packed_end],
            core_attn_out[packed_offset:packed_end],
            context,
            attn_metadata,
            request_id,
            request_index,
        )
        packed_offset = packed_end

    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    if packed_offset != num_actual_tokens:
        raise RuntimeError(
            f"HYPIC GDN consumed {packed_offset} of {num_actual_tokens} packed tokens"
        )
