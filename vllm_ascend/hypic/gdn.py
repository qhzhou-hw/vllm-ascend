"""Transition/recompute HYPIC execution for Ascend Gated DeltaNet layers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, cast

import torch
import torch.nn.functional as F

try:
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
except ImportError:  # Optional until HYPIC is enabled.
    chunk_gated_delta_rule_npu = None

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.hypic.compose import ComposeStep, build_compose_steps, compose_fused_batch, compose_reference
from vllm_ascend.hypic.runtime import HypicBatchContext

_CU_CACHE_SIZE = 8


def _cu_seqlens(layer: Any, lengths: list[int], device: torch.device) -> torch.Tensor:
    """Bounded, device-qualified cache of immutable layout metadata, not state.

    sgl-kernel-npu caches derived chunk indices by tensor identity. Retain the
    exact tensor for repeated layouts so S/T/Query do not repeat CPU-NPU syncs.
    """
    cache = getattr(layer, "_hypic_gdn_cu_cache", None)
    if cache is None:
        cache = layer._hypic_gdn_cu_cache = OrderedDict()
    key = (device, tuple(lengths))
    if key not in cache:
        cache[key] = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
        if len(cache) > _CU_CACHE_SIZE:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


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


@dataclass
class _PendingGdn:
    """Keep only one bounded group's inputs alive until the fused state update."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    cu_seqlens: torch.Tensor
    zero_states: torch.Tensor | None
    transitions: torch.Tensor | None
    history: torch.Tensor
    output: torch.Tensor
    request_index: int
    steps: tuple[ComposeStep, ...]
    summarizes_query: bool


def _finish_request(
    layer: Any, work: _PendingGdn, replay: torch.Tensor, document_state: torch.Tensor, attn_metadata: Any
) -> None:
    """Replay fresh units, or execute Query once for the cache-only fast path."""
    state_indices = getattr(attn_metadata, "non_spec_state_indices_tensor", None)
    if state_indices is None:
        state_indices = attn_metadata.prefill_state_indices
    if state_indices is None or work.request_index >= len(state_indices):
        raise RuntimeError(f"HYPIC GDN state metadata is missing request {work.request_index}")
    initial = replay if work.summarizes_query else document_state.unsqueeze(0)
    output, final = _run_gdn(
        layer,
        work.q,
        work.k,
        work.v,
        work.g,
        work.beta,
        initial,
        work.cu_seqlens,
    )
    work.output.copy_(output.squeeze(0))
    state_index = state_indices[work.request_index].to(torch.long)
    final_state = document_state if work.summarizes_query else final[-1]
    layer.kv_cache[1][state_index] = final_state.to(layer.kv_cache[1].dtype)
    layer.kv_cache[0][state_index] = work.history.to(layer.kv_cache[0].dtype)


def _finish_fused_group(layer: Any, pending: list[_PendingGdn], attn_metadata: Any) -> None:
    """Compose ordered summaries across requests/heads, then replay outputs."""
    if not pending or any(work.zero_states is None or work.transitions is None for work in pending):
        raise RuntimeError("HYPIC fused composition requires an unconsumed nonempty group")
    replay_states, final_states = compose_fused_batch(
        [cast(torch.Tensor, work.zero_states) for work in pending],
        [cast(torch.Tensor, work.transitions) for work in pending],
        layer.hypic_zero_state_pool,
        layer.hypic_transition_pool,
        [work.steps for work in pending],
    )
    # Drop all fresh S/T workspaces before replay, as in the reference path.
    for work in pending:
        work.zero_states = None
        work.transitions = None
    for row, (work, replay) in enumerate(zip(pending, replay_states)):
        _finish_request(layer, work, replay, final_states[row], attn_metadata)


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
    pending: list[_PendingGdn] | None = None,
) -> None:
    """Execute HYPIC S/T composition for one request in a packed batch."""
    plan = context.plans[request_id]
    query = plan["segments"][-1]
    if query["hit"] or query["cacheable"] or query["recompute_seam"]:
        raise RuntimeError("HYPIC GDN requires a final uncached, unsplit Query")
    block_native_pic = plan.get("mode") == "block_native_pic"
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
    zero_history = mixed_qkv.new_zeros((width - 1, mixed_qkv.shape[-1]))
    history = zero_history
    reset_conv_history = plan.get("reset_conv_history", False)
    transformed_parts: list[torch.Tensor] = []
    a_parts: list[torch.Tensor] = []
    b_parts: list[torch.Tensor] = []
    units: list[dict[str, Any]] = []
    cache_writes: list[tuple[dict[str, Any], int, int]] = []
    packed_offset = 0

    for segment in plan["segments"]:
        if reset_conv_history:
            # history may alias a cached tail. Never zero it in place: hits
            # and other requests still own that cache entry. Reset once per
            # segment, not between its seam and interior GDN compute units.
            history = zero_history
        length = int(segment["end"]) - int(segment["start"])
        hit = bool(segment["hit"])
        query_len = int(segment["seam"]) if hit else length
        raw = mixed_qkv[packed_offset : packed_offset + query_len]
        part_a = a[packed_offset : packed_offset + query_len]
        part_b = b[packed_offset : packed_offset + query_len]
        store = segment.get("store", segment["cacheable"])
        slot = context.cache.lookup(segment["hash"]) if hit or store else None
        if (hit or store) and slot is None:
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
            if store:
                conv_pool[slot].copy_(history)
                history = conv_pool[slot]
            seam = int(segment["recompute_seam"])
            if seam:
                units.append({"segment": segment, "kind": "seam", "length": seam})
                interior_index = len(units)
                units.append({"segment": segment, "kind": "interior", "length": length - seam})
                if store:
                    cache_writes.append((segment, interior_index, slot))
            else:
                unit_index = len(units)
                units.append({"segment": segment, "kind": "full", "length": length})
                if store:
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
    cu_seqlens = _cu_seqlens(layer, lengths, q.device)
    # Splitting fresh Document/Query passes regressed real-NPU latency due to
    # extra varlen metadata processing. Keep the original packed path there.
    # Without fresh Documents/seams, Query alone needs no affine summary.
    summarizes_query = len(units) > 1
    num_units = len(units) if summarizes_query else 0
    num_heads = v.shape[2]
    value_dim = v.shape[-1]
    key_dim = k.shape[-1]
    if num_units:
        zero = torch.zeros((num_units, num_heads, value_dim, key_dim), dtype=torch.float32, device=q.device)
        zero_output, zero_states = _run_gdn(layer, q, k, v, g, beta, zero, cu_seqlens)
        del zero_output, zero
        identity = torch.zeros((num_units, num_heads, key_dim, key_dim), dtype=torch.float32, device=q.device)
        identity.diagonal(dim1=-2, dim2=-1).fill_(1)
        v_zero = v.new_zeros((1, v.shape[1], num_heads, key_dim))
        transition_output, transitions = _run_gdn(layer, q, k, v_zero, g, beta, identity, cu_seqlens)
        del transition_output, identity, v_zero
        zero_states = zero_states.float()
        transitions = transitions.float()
    else:
        # Fully cached Documents (or a Query-only plan): no S/T GDN calls.
        zero_states = torch.empty((0, num_heads, value_dim, key_dim), dtype=torch.float32, device=q.device)
        transitions = zero_states.new_empty((0, num_heads, key_dim, key_dim))

    for _, unit_index, slot in cache_writes:
        zero_state_pool[slot].copy_(zero_states[unit_index])
        transition_pool[slot].copy_(transitions[unit_index])

    steps, expected_units = build_compose_steps(plan, context.cache.lookup, include_query=summarizes_query)
    if expected_units != num_units:
        raise RuntimeError("HYPIC composition unit count mismatch")
    work = _PendingGdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        zero_states=zero_states,
        transitions=transitions,
        history=history,
        output=core_attn_out[:num_tokens],
        request_index=request_index,
        steps=steps,
        summarizes_query=summarizes_query,
    )
    if pending is not None:
        pending.append(work)
        return
    replay_initial, document_state = compose_reference(
        zero_states, transitions, zero_state_pool, transition_pool, steps, zero_replay=block_native_pic
    )
    work.zero_states = work.transitions = None
    del zero_states, transitions, cache_writes
    _finish_request(layer, work, replay_initial, document_state, attn_metadata)


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
    backend = getattr(layer, "hypic_state_compose_backend", "torch")
    if backend not in {"torch", "triton"}:
        raise ValueError(f"Unsupported HYPIC state composition backend: {backend}")
    if backend == "triton" and mixed_qkv.device.type != "npu":
        raise ValueError("HYPIC triton composition requires an NPU")
    group_size = getattr(layer, "hypic_state_compose_batch_size", 1)
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("HYPIC state composition batch size must be a positive integer")
    pending: list[_PendingGdn] = []
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
            **({"pending": pending} if backend == "triton" else {}),
        )
        packed_offset = packed_end
        if pending and len(pending) >= group_size:
            _finish_fused_group(layer, pending, attn_metadata)
            pending.clear()

    if pending:
        _finish_fused_group(layer, pending, attn_metadata)
        pending.clear()

    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    if packed_offset != num_actual_tokens:
        raise RuntimeError(f"HYPIC GDN consumed {packed_offset} of {num_actual_tokens} packed tokens")
