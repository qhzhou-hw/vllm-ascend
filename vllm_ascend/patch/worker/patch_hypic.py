"""Model-runner and operator dispatch patches for HYPIC."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.hypic.attention import forward_hypic_attention
from vllm_ascend.hypic.cache import DeviceSegmentCache
from vllm_ascend.hypic.config import get_hypic_config
from vllm_ascend.hypic.gdn import forward_hypic_gdn
from vllm_ascend.hypic.runtime import (
    HypicBatchContext,
    current_hypic_context,
    set_hypic_context,
)
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

_ORIGINAL_QWEN_INIT = Qwen3NextAttention.__init__
_ORIGINAL_GDN_INIT = QwenGatedDeltaNetAttention.__init__
_ORIGINAL_UPDATE_STATES = NPUModelRunner._update_states
_ORIGINAL_PREPARE_INPUTS = NPUModelRunner._prepare_inputs
_ORIGINAL_ATTENTION_FORWARD = AscendAttentionBackendImpl.forward
_ORIGINAL_ATTENTION_IMPL = AscendAttentionBackendImpl.forward_impl
_ORIGINAL_GDN_CORE = QwenGatedDeltaNetAttention._forward_core


def _qwen_init(self: Qwen3NextAttention, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_QWEN_INIT(self, *args, **kwargs)
    self.attn.hypic_rotary_emb = self.rotary_emb
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    config = get_hypic_config(vllm_config)
    if config.enabled:
        shape = (
            config.max_cache_segments,
            config.chunk_size,
            self.num_kv_heads,
            self.head_dim,
        )
        device = current_platform.current_device()
        dtype = vllm_config.model_config.dtype
        self.attn.register_buffer(
            "hypic_key_pool",
            torch.empty(shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.attn.register_buffer(
            "hypic_value_pool",
            torch.empty(shape, dtype=dtype, device=device),
            persistent=False,
        )


def _gdn_init(
    self: QwenGatedDeltaNetAttention,
    config: Any,
    vllm_config: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    _ORIGINAL_GDN_INIT(self, config, vllm_config, *args, **kwargs)
    hypic_config = get_hypic_config(vllm_config)
    if not hypic_config.enabled:
        return
    slots = hypic_config.max_cache_segments
    local_value_heads = self.num_v_heads // self.tp_size
    state_shape = (slots, local_value_heads, self.head_v_dim, self.head_k_dim)
    width = int(self.conv1d.weight.shape[-1])
    mixed_dim = int(self.conv1d.weight.shape[0])
    device = current_platform.current_device()
    self.register_buffer(
        "hypic_conv_pool",
        torch.empty(
            (slots, width - 1, mixed_dim),
            dtype=vllm_config.model_config.dtype,
            device=device,
        ),
        persistent=False,
    )
    self.register_buffer(
        "hypic_zero_state_pool",
        torch.empty(state_shape, dtype=torch.float32, device=device),
        persistent=False,
    )
    self.register_buffer(
        "hypic_transition_pool",
        torch.empty(state_shape, dtype=torch.float32, device=device),
        persistent=False,
    )


def _update_states(self: NPUModelRunner, scheduler_output: Any) -> Any:
    config = get_hypic_config(self.vllm_config)
    if config.enabled:
        if not hasattr(self, "hypic_device_cache"):
            self.hypic_device_cache = DeviceSegmentCache(config.max_cache_segments)
            self.hypic_plans = {}
        for request_data in scheduler_output.scheduled_new_reqs:
            plan = getattr(request_data, "hypic_plan", None)
            if plan is not None:
                self.hypic_plans[request_data.req_id] = plan
        for request_id in scheduler_output.finished_req_ids:
            self.hypic_plans.pop(request_id, None)
    return _ORIGINAL_UPDATE_STATES(self, scheduler_output)


def _prepare_inputs(
    self: NPUModelRunner,
    scheduler_output: Any,
    num_scheduled_tokens: np.ndarray,
) -> Any:
    result = _ORIGINAL_PREPARE_INPUTS(self, scheduler_output, num_scheduled_tokens)
    config = get_hypic_config(self.vllm_config)
    if not config.enabled:
        set_hypic_context(None)
        return result

    planned_request_ids = tuple(
        request_data.req_id
        for request_data in scheduler_output.scheduled_new_reqs
        if request_data.req_id in self.hypic_plans
    )
    new_request_ids = set(planned_request_ids)
    active_ids = tuple(
        request_id
        for request_id in self.input_batch.req_ids
        if request_id in new_request_ids and request_id in self.hypic_plans
    )
    if not active_ids:
        set_hypic_context(None)
        return result
    if tuple(self.input_batch.req_ids) != active_ids:
        raise RuntimeError(
            "HYPIC requires a prefill-only batch; mixed prefill/decode "
            "forward detected"
        )
    if active_ids != planned_request_ids:
        raise RuntimeError(
            "HYPIC scheduler/worker request order divergence: "
            f"scheduler={planned_request_ids}, worker={active_ids}"
        )

    # Assign every active segment a stable slot before the first model layer.
    # All attention and GDN layers then read/write the same slot id without
    # mutating the LRU while a packed forward is in progress.
    self.hypic_device_cache.prepare(
        self.hypic_plans[request_id] for request_id in active_ids
    )

    total = int(scheduler_output.total_num_scheduled_tokens)
    packed_offset = 0
    for row_index, request_id in enumerate(active_ids):
        plan = self.hypic_plans[request_id]
        positions = np.asarray(plan["query_positions"], dtype=np.int64)
        scheduled = int(num_scheduled_tokens[row_index])
        if len(positions) != scheduled:
            raise RuntimeError(
                f"HYPIC request {request_id} scheduled {scheduled} tokens "
                f"for {len(positions)} query positions"
            )
        packed_end = packed_offset + scheduled
        token_row = self.input_batch.token_ids_cpu[row_index]
        selected_ids = torch.from_numpy(token_row[positions]).to(torch.int32)
        self.input_ids.cpu[packed_offset:packed_end].copy_(selected_ids)
        self.input_ids.gpu[packed_offset:packed_end].copy_(
            selected_ids.to(self.device)
        )
        position_tensor = torch.from_numpy(positions).to(self.device)
        self.positions[packed_offset:packed_end].copy_(position_tensor)
        packed_offset = packed_end
    if packed_offset != total:
        raise RuntimeError(
            f"HYPIC packed {packed_offset} tokens for a {total}-token batch"
        )
    self.input_batch.block_table.compute_slot_mapping(
        len(active_ids),
        self.query_start_loc.gpu[: len(active_ids) + 1],
        self.positions[:total],
    )
    set_hypic_context(
        HypicBatchContext(
            # Snapshot every plan in this packed prefill. The loop variables
            # above otherwise retain only the final request's plan.
            plans={
                request_id: self.hypic_plans[request_id]
                for request_id in active_ids
            },
            request_ids=active_ids,
            cache=self.hypic_device_cache,
        )
    )
    return result


def _attention_forward(self: AscendAttentionBackendImpl, layer: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    self.hypic_layer = layer
    context = current_hypic_context()
    if context is not None:
        kv_cache = kwargs.get("kv_cache", args[3] if len(args) > 3 else None)
        metadata = kwargs.get("attn_metadata", args[4] if len(args) > 4 else None)
        if kv_cache is None or metadata is None:
            raise RuntimeError("HYPIC attention cache metadata is missing")
        block_size = int(kv_cache[0].shape[1])
        packed_offset = 0
        for request_index, request_id in enumerate(context.request_ids):
            positions = torch.tensor(
                context.plans[request_id]["query_positions"],
                dtype=torch.long,
                device=metadata.block_tables.device,
            )
            logical_blocks = torch.div(
                positions, block_size, rounding_mode="floor"
            )
            physical_blocks = metadata.block_tables[request_index].index_select(
                0, logical_blocks
            )
            slots = (
                physical_blocks.to(torch.long) * block_size
                + positions.remainder(block_size)
            )
            packed_end = packed_offset + len(slots)
            metadata.slot_mapping[packed_offset:packed_end].copy_(
                slots.to(torch.int32)
            )
            packed_offset = packed_end
    return _ORIGINAL_ATTENTION_FORWARD(self, layer, *args, **kwargs)


def _attention_impl(
    self: AscendAttentionBackendImpl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: tuple[torch.Tensor],
    attn_metadata: Any,
    output: torch.Tensor,
) -> torch.Tensor:
    context = current_hypic_context()
    if context is None:
        return _ORIGINAL_ATTENTION_IMPL(self, query, key, value, kv_cache, attn_metadata, output)
    return forward_hypic_attention(
        self.hypic_layer,
        query[: attn_metadata.num_actual_tokens],
        key[: attn_metadata.num_actual_tokens],
        value[: attn_metadata.num_actual_tokens],
        output,
        context,
        scale=self.scale,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )


def _gdn_core(
    self: AscendGatedDeltaNetAttention,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    context = current_hypic_context()
    if context is None:
        return _ORIGINAL_GDN_CORE(self, mixed_qkv, b, a, core_attn_out)
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata[self.prefix]
    forward_hypic_gdn(self, mixed_qkv, b, a, core_attn_out, context, metadata)


Qwen3NextAttention.__init__ = _qwen_init
QwenGatedDeltaNetAttention.__init__ = _gdn_init
NPUModelRunner._update_states = _update_states
NPUModelRunner._prepare_inputs = _prepare_inputs
AscendAttentionBackendImpl.forward = _attention_forward
AscendAttentionBackendImpl.forward_impl = _attention_impl
QwenGatedDeltaNetAttention._forward_core = _gdn_core
