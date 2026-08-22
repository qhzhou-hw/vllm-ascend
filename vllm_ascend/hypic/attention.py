"""Suffix-causal attention primitives for HYPIC full-attention layers."""

from __future__ import annotations

import math

import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.hypic.cache import LayerSegmentState
from vllm_ascend.hypic.runtime import HypicBatchContext


def reference_suffix_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    logit_cap: float = 0.0,
) -> torch.Tensor:
    """Compute GQA attention where ``query`` is the suffix of ``key/value``."""
    query_len, query_heads, head_dim = query.shape
    kv_len, kv_heads, _ = key.shape
    if kv_len < query_len:
        raise ValueError("HYPIC suffix attention requires kv_len >= query_len")
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if query_len == 0:
        return query.new_empty((0, query_heads, value.shape[-1]))

    groups = query_heads // kv_heads
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if logit_cap > 0:
        scores = logit_cap * torch.tanh(scores / logit_cap)
    query_position = torch.arange(query_len, device=query.device) + (kv_len - query_len)
    key_position = torch.arange(kv_len, device=query.device)
    mask = key_position.unsqueeze(0) > query_position.unsqueeze(1)
    scores.masked_fill_(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probability = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("hqk,khd->qhd", probability, value)


_NPU_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def suffix_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    logit_cap: float = 0.0,
) -> torch.Tensor:
    """Use CANN right-down causal attention with a reference fallback."""
    if query.device.type != "npu" or logit_cap > 0:
        return reference_suffix_attention(query, key, value, scale=scale, logit_cap=logit_cap)
    import torch_npu

    scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    mask = _NPU_CAUSAL_MASKS.get(query.device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        _NPU_CAUSAL_MASKS[query.device] = mask
    output = torch_npu.npu_prompt_flash_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.transpose(0, 1).unsqueeze(0).contiguous(),
        value.transpose(0, 1).unsqueeze(0).contiguous(),
        atten_mask=mask,
        num_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        scale_value=scale,
        input_layout="BNSD",
        sparse_mode=3,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()


def attention_from_segments(
    query: torch.Tensor,
    key_segments: list[torch.Tensor],
    value_segments: list[torch.Tensor],
    query_lengths: list[int],
    *,
    scale: float | None = None,
    logit_cap: float = 0.0,
) -> torch.Tensor:
    """Run every logical query segment against its ordered KV prefix."""
    if not (len(key_segments) == len(value_segments) == len(query_lengths)):
        raise ValueError("segment lists must have identical lengths")
    outputs: list[torch.Tensor] = []
    query_offset = 0
    prefix_keys: list[torch.Tensor] = []
    prefix_values: list[torch.Tensor] = []
    for key, value, query_len in zip(key_segments, value_segments, query_lengths):
        prefix_keys.append(key)
        prefix_values.append(value)
        if query_len:
            segment_query = query[query_offset : query_offset + query_len]
            outputs.append(
                suffix_attention(
                    segment_query,
                    torch.cat(prefix_keys),
                    torch.cat(prefix_values),
                    scale=scale,
                    logit_cap=logit_cap,
                )
            )
            query_offset += query_len
    if query_offset != query.shape[0]:
        raise ValueError("query_lengths do not consume the packed query tensor")
    if not outputs:
        value_dim = value_segments[-1].shape[-1]
        return query.new_empty((0, query.shape[1], value_dim))
    return torch.cat(outputs)


def _rotate_half_neox(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rerotate_keys(
    key: torch.Tensor,
    from_positions: torch.Tensor,
    to_positions: torch.Tensor,
    rotary_emb: object,
) -> torch.Tensor:
    """Move already-RoPE'd keys between absolute and segment-local positions."""
    if not getattr(rotary_emb, "is_neox_style", False):
        raise NotImplementedError("HYPIC currently requires NeoX-style RoPE")
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb.cos_sin_cache.to(key.device, key.dtype)
    from_cos, from_sin = cache.index_select(0, from_positions).chunk(2, dim=-1)
    to_cos, to_sin = cache.index_select(0, to_positions).chunk(2, dim=-1)
    delta_cos = to_cos * from_cos + to_sin * from_sin
    delta_sin = to_sin * from_cos - to_cos * from_sin
    rotated = key[..., :rotary_dim]
    # vLLM stores one frequency value per rotary pair. NeoX layout applies
    # that same value to the matching coordinates in both half vectors.
    delta_cos = torch.cat((delta_cos, delta_cos), dim=-1).unsqueeze(1)
    delta_sin = torch.cat((delta_sin, delta_sin), dim=-1).unsqueeze(1)
    rotated = rotated * delta_cos + _rotate_half_neox(rotated) * delta_sin
    return torch.cat((rotated, key[..., rotary_dim:]), dim=-1)


def _hydrate_paged_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    start: int,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: object,
) -> None:
    """Materialize a reused segment in vLLM's cache for subsequent decoding."""
    if len(kv_cache) < 2:
        raise RuntimeError("HYPIC requires a writable paged KV cache")
    block_tables = getattr(attn_metadata, "block_tables", None)
    if block_tables is None or block_tables.shape[0] != 1:
        raise RuntimeError("HYPIC expected exactly one paged-cache block table")

    block_size = int(kv_cache[0].shape[1])
    positions = torch.arange(start, start + key.shape[0], device=key.device)
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    physical_blocks = block_tables[0].index_select(0, logical_blocks.to(block_tables.device, torch.long))
    slots = physical_blocks.to(torch.long) * block_size + positions.remainder(block_size)
    if bool((slots < 0).any().item()):
        raise RuntimeError("HYPIC paged-cache block table contains an unallocated block")
    DeviceOperator.reshape_and_cache(
        key=key,
        value=value,
        key_cache=kv_cache[0],
        value_cache=kv_cache[1],
        slot_mapping=slots.to(torch.int32),
    )


def forward_hypic_attention(
    layer: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    context: HypicBatchContext,
    *,
    scale: float,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: object,
) -> torch.Tensor:
    """Execute packed HYPIC plans for one full-attention layer."""
    layer_name = str(layer.layer_name)
    rotary_emb = getattr(layer, "hypic_rotary_emb", None)
    if rotary_emb is None:
        raise RuntimeError(f"HYPIC RoPE metadata is missing for {layer_name}")

    packed_offset = 0
    output_offset = 0
    for request_id in context.request_ids:
        plan = context.plans.get(request_id)
        if plan is None:
            continue
        prefix_keys: list[torch.Tensor] = []
        prefix_values: list[torch.Tensor] = []
        for segment in plan["segments"]:
            start = int(segment["start"])
            end = int(segment["end"])
            length = end - start
            hit = bool(segment["hit"])
            query_len = int(segment["seam"]) if hit else length
            current_key = key[packed_offset : packed_offset + query_len]
            current_value = value[packed_offset : packed_offset + query_len]

            if hit:
                cached = context.cache.get(segment["hash"], layer_name)
                if cached is None or cached.key is None or cached.value is None:
                    raise RuntimeError(
                        f"HYPIC scheduler/worker cache divergence for segment {segment['hash']} at {layer_name}"
                    )
                local_positions = torch.arange(length, device=key.device)
                absolute_positions = local_positions + start
                segment_key = rerotate_keys(cached.key, local_positions, absolute_positions, rotary_emb)
                segment_value = cached.value
                if query_len:
                    segment_key = segment_key.clone()
                    segment_value = segment_value.clone()
                    segment_key[:query_len] = current_key
                    segment_value[:query_len] = current_value
                    attention_key = torch.cat(prefix_keys + [current_key])
                    attention_value = torch.cat(prefix_values + [current_value])
                else:
                    attention_key = None
                    attention_value = None
                # The custom prefill reads HYPIC's segment cache directly, but
                # ordinary vLLM decoding resumes immediately afterwards. Fill
                # every skipped segment into the standard paged cache so decode
                # observes exactly the same complete prefix as a cold prefill.
                _hydrate_paged_kv_cache(
                    segment_key,
                    segment_value,
                    start=start,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                )
            else:
                segment_key = current_key
                segment_value = current_value
                attention_key = torch.cat(prefix_keys + [segment_key])
                attention_value = torch.cat(prefix_values + [segment_value])
                if segment["cacheable"]:
                    absolute_positions = torch.arange(start, end, device=key.device)
                    local_positions = torch.arange(length, device=key.device)
                    public_key = rerotate_keys(
                        segment_key,
                        absolute_positions,
                        local_positions,
                        rotary_emb,
                    )
                    context.cache.put(
                        segment["hash"],
                        layer_name,
                        LayerSegmentState(
                            key=public_key.detach().clone(),
                            value=segment_value.detach().clone(),
                        ),
                    )

            if query_len:
                segment_query = query[packed_offset : packed_offset + query_len]
                segment_output = suffix_attention(
                    segment_query,
                    attention_key,
                    attention_value,
                    scale=scale,
                )
                target = output[output_offset : output_offset + query_len]
                target.copy_(segment_output.reshape_as(target))
                packed_offset += query_len
                output_offset += query_len
            prefix_keys.append(segment_key)
            prefix_values.append(segment_value)

    if packed_offset != query.shape[0]:
        raise RuntimeError(f"HYPIC consumed {packed_offset} of {query.shape[0]} packed tokens")
    return output
