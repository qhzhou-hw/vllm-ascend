"""Scheduler and configuration patches for opt-in HYPIC execution."""

from __future__ import annotations

import inspect
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.hypic.config import get_hypic_config
from vllm_ascend.hypic.pic_cache import PicCatalog
from vllm_ascend.hypic.planner import build_plan
from vllm_ascend.hypic.protocol import request_policy
from vllm_ascend.hypic.vllm_adapter import (
    PicSchedulerOutput,
    logical_scheduler_output,
    pack_scheduler_output,
)
from vllm_ascend.platform import NPUPlatform

logger = init_logger(__name__)


_ORIGINAL_CHECK_CONFIG = NPUPlatform.check_and_update_config.__func__
_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__
_ORIGINAL_SCHEDULE = Scheduler.schedule
_ORIGINAL_MAMBA_SPLIT = Scheduler._mamba_block_aligned_split
_ORIGINAL_GET_COMPUTED_BLOCKS = KVCacheManager.get_computed_blocks
_ORIGINAL_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
_ORIGINAL_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
_ALLOCATE_SIGNATURE = inspect.signature(_ORIGINAL_ALLOCATE_SLOTS)


def _check_and_update_config(cls: type, vllm_config: Any) -> None:
    config = get_hypic_config(vllm_config)
    _ORIGINAL_CHECK_CONFIG(cls, vllm_config)
    if config.enabled:
        model_config = vllm_config.model_config
        architectures = set(getattr(model_config, "architectures", ()) or ())
        # vLLM exposes the text path of Qwen3.5 through the conditional-
        # generation wrappers, even when the request contains no vision input.
        supported = {
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }
        if not architectures.intersection(supported):
            raise ValueError(
                "HYPIC on vllm-ascend currently supports text-only Qwen3.5 "
                f"models; got architectures={sorted(architectures)}"
            )
        parallel = vllm_config.parallel_config
        unsupported_parallel = {
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "data_parallel_size": parallel.data_parallel_size,
            "prefill_context_parallel_size": (parallel.prefill_context_parallel_size),
            "decode_context_parallel_size": parallel.decode_context_parallel_size,
        }
        invalid = {name: value for name, value in unsupported_parallel.items() if value != 1}
        if invalid:
            raise ValueError(f"HYPIC does not yet support these parallel modes: {invalid}")
        if vllm_config.speculative_config is not None:
            raise ValueError("HYPIC does not support speculative decoding")
        if vllm_config.kv_transfer_config is not None:
            raise ValueError("HYPIC does not support KV transfer/disaggregation")

        if getattr(vllm_config, "lora_config", None) is not None:
            raise ValueError("PIC adapter-specific tensor pools are not implemented")
        if getattr(model_config, "enable_return_routed_experts", False):
            raise ValueError("PIC does not support returning routed experts")
        if getattr(model_config, "quantization", None) is not None:
            raise ValueError("PIC quantized payloads are not implemented")

        model_config.enforce_eager = True
        # SegmentCatalog commits misses only after the corresponding model
        # output. Async scheduling could plan another batch against that
        # uncommitted state while the worker has already reserved its slots.
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.scheduler_config.enable_chunked_prefill = False
        vllm_config.scheduler_config.long_prefill_token_threshold = 0
        # Hybrid models otherwise retain a 2048-token scheduling cap even when
        # max_num_batched_tokens is larger. HYPIC must plan and execute a whole
        # prompt atomically because its query positions are non-contiguous.
        vllm_config.scheduler_config.max_num_scheduled_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Keep vLLM prefix caching enabled so its hybrid-cache page-size
        # validation remains satisfied. PIC requests suppress both standard
        # APC lookup and publication; prefix_only requests retain native APC.
        vllm_config.cache_config.mamba_cache_mode = "align"
        logger.info(
            "Enabled HYPIC mode=%s with chunk_size=%d, seam=%d, reset_conv_history=%s",
            config.mode,
            config.chunk_size,
            config.seam_sink_tokens,
            config.reset_conv_history or config.mode == "block_native_pic",
        )


def _scheduler_init(self: Scheduler, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_SCHEDULER_INIT(self, *args, **kwargs)
    config = get_hypic_config(self.vllm_config)
    if config.enabled:
        self.hypic_config = config
        self.hypic_catalog = PicCatalog(config.max_cache_segments)
        self.hypic_batch_kind = None
        self.hypic_batch_units = 0
        self.kv_cache_manager.hypic_scheduler = self
        # Guard the final publication boundary, including allocate-time and
        # decode-time callers. Do not turn off hybrid block allocation itself.
        coordinator = self.kv_cache_manager.coordinator
        original_cache_blocks = coordinator.cache_blocks

        def cache_blocks(request: Any, num_computed_tokens: int) -> None:
            if config.mode != "block_native_pic" and request_policy(request) == "prefix_only":
                original_cache_blocks(request, num_computed_tokens)

        coordinator.cache_blocks = cache_blocks


def _schedule(self: Scheduler, *args: Any, **kwargs: Any) -> Any:
    """Keep HYPIC prefills isolated from in-flight decode requests.

    Multiple waiting requests may still be admitted together when the running
    set is empty. Once admitted, that request group drains before the scheduler
    admits another HYPIC prefill group. This preserves ordinary batched decode
    while avoiding a mixed custom-prefill/standard-decode model forward.
    """
    catalog = getattr(self, "hypic_catalog", None)
    if catalog is not None:
        if catalog.failed:
            raise RuntimeError("PIC execution failed; restart the engine")
        if catalog.pending is not None:
            raise RuntimeError("PIC previous step has not completed")
        self.hypic_batch_kind = None
        self.hypic_batch_units = 0
    if hasattr(self, "hypic_catalog") and self.running and self._pause_state == PauseState.UNPAUSED:
        self._pause_state = PauseState.PAUSED_NEW
        try:
            scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
        finally:
            self._pause_state = PauseState.UNPAUSED
    else:
        scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)

    if catalog is not None:
        plans = {
            data.req_id: self.requests[data.req_id].hypic_plan
            for data in scheduler_output.scheduled_new_reqs
            if getattr(self.requests[data.req_id], "hypic_plan", None) is not None
        }
        if plans:
            if set(plans) != set(scheduler_output.num_scheduled_tokens):
                raise RuntimeError("PIC requires a homogeneous prefill batch")
            prepared, step = catalog.prepare(plans)
            try:
                logger.debug(
                    "PIC step=%d logical_tokens=%d query_tokens=%d restore_tokens=%d "
                    "units=%d reads=%d fills=%d evictions=%d",
                    step.step_id,
                    sum(plan["logical_advance"] for plan in prepared.values()),
                    sum(plan["num_query_tokens"] for plan in prepared.values()),
                    sum(plan["num_restore_tokens"] for plan in prepared.values()),
                    sum(plan["num_prefill_units"] for plan in prepared.values()),
                    len(step.reads),
                    len(step.fills),
                    sum(fill.victim is not None for fill in step.fills),
                )
                return pack_scheduler_output(scheduler_output, prepared, step)
            except Exception:
                catalog.abort(dispatched=False)
                raise
    return scheduler_output


def _allocate_slots(self: KVCacheManager, request: Any, *args: Any, **kwargs: Any) -> Any:
    scheduler = getattr(self, "hypic_scheduler", None)
    if scheduler is None:
        return _ORIGINAL_ALLOCATE_SLOTS(self, request, *args, **kwargs)
    policy = request_policy(request)
    # A preempted request replays its full prompt + generated tokens through
    # native execution. It retains PIC publication isolation but no old lease.
    block_native = scheduler.hypic_config.mode == "block_native_pic"
    if block_native and (request.num_preemptions or policy == "prefix_only"):
        raise ValueError("block_native_pic cannot fall back to native prefix/preemption replay")
    kind = "pic" if (policy == "pic" or block_native) and not request.num_preemptions else "native"
    active_kind = scheduler.hypic_batch_kind
    if active_kind is not None and active_kind != kind:
        return None
    plan = getattr(request, "hypic_plan", None)
    units = plan["num_prefill_units"] if plan is not None and request.num_computed_tokens == 0 else 0
    if scheduler.hypic_batch_units + units > scheduler.hypic_config.max_prefill_units:
        return None
    bound = _ALLOCATE_SIGNATURE.bind(self, request, *args, **kwargs)
    if policy != "prefix_only":
        bound.arguments["delay_cache_blocks"] = True
    result = _ORIGINAL_ALLOCATE_SLOTS(*bound.args, **bound.kwargs)
    if result is not None:
        scheduler.hypic_batch_kind = kind
        scheduler.hypic_batch_units += units
    return result


def _mamba_block_aligned_split(
    self: Scheduler,
    request: Any,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    if (
        hasattr(self, "hypic_catalog")
        and (request_policy(request) == "pic" or self.hypic_config.mode == "block_native_pic")
        and not request.num_preemptions
    ):
        return num_new_tokens
    return _ORIGINAL_MAMBA_SPLIT(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens,
        num_external_computed_tokens,
    )


def _get_computed_blocks(self: KVCacheManager, request: Any) -> tuple[Any, int, int]:
    scheduler = getattr(self, "hypic_scheduler", None)
    if scheduler is None:
        return _ORIGINAL_GET_COMPUTED_BLOCKS(self, request)
    request.hypic_plan = None
    policy = request_policy(request)
    block_native = scheduler.hypic_config.mode == "block_native_pic"
    if block_native and (request.num_preemptions or policy == "prefix_only"):
        raise ValueError("block_native_pic cannot fall back to native prefix/preemption replay")
    if policy == "prefix_only":
        return _ORIGINAL_GET_COMPUTED_BLOCKS(self, request)
    if (policy == "full_recompute" and not block_native) or request.num_preemptions:
        return self.empty_kv_cache_blocks, 0, 0
    if request.prompt_token_ids is None:
        raise ValueError("HYPIC requires token-id prompts")
    if getattr(request, "mm_features", None) or getattr(request, "prompt_embeds", None) is not None:
        raise ValueError("PIC currently requires text-only token-id prompts")
    if request.num_tokens > scheduler.max_num_scheduled_tokens:
        raise ValueError(
            "PIC atomic admission requires max_num_batched_tokens >= prompt length; "
            f"got {scheduler.max_num_scheduled_tokens} < {request.num_tokens}"
        )
    if request.sampling_params is not None and getattr(request.sampling_params, "prompt_logprobs", None) is not None:
        raise ValueError("HYPIC does not support prompt logprobs")
    extra_args = (
        getattr(request.sampling_params, "extra_args", None) if request.sampling_params is not None else None
    ) or {}
    segment_boundaries = extra_args.get("hypic_segment_boundaries")
    plan = build_plan(
        request.prompt_token_ids,
        {} if policy == "full_recompute" else scheduler.hypic_catalog.ready,
        scheduler.hypic_config,
        segment_boundaries=segment_boundaries,
        cache_salt=getattr(request, "cache_salt", None),
    )
    if block_native and policy == "full_recompute":
        # The baseline must keep the training computation graph, not revert
        # to ordinary causal attention. Disable PIC reads AND publication.
        for segment in plan["segments"]:
            segment["cacheable"] = False
    if plan["num_prefill_units"] > scheduler.hypic_config.max_prefill_units:
        raise ValueError(
            f"PIC prompt needs {plan['num_prefill_units']} GDN units, exceeding "
            f"max_prefill_units={scheduler.hypic_config.max_prefill_units}; "
            "increase the workspace budget or reduce semantic fragmentation"
        )
    request.hypic_plan = plan
    # Sparse reuse is not a contiguous computed prefix. The unmodified vLLM
    # allocator reserves the full logical prompt; only worker query counts are
    # reduced after admission and slot planning.
    return self.empty_kv_cache_blocks, 0, 0


def _update_from_output(self: Scheduler, scheduler_output: Any, model_runner_output: Any) -> Any:
    if not isinstance(scheduler_output, PicSchedulerOutput):
        return _ORIGINAL_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)
    catalog = self.hypic_catalog
    ack = getattr(model_runner_output, "pic_ack", None)
    try:
        if ack is None:
            raise RuntimeError("PIC worker did not acknowledge all layer writes")
        catalog.commit(ack, expected_ranks=self.vllm_config.parallel_config.tensor_parallel_size)
        return _ORIGINAL_UPDATE_FROM_OUTPUT(self, logical_scheduler_output(scheduler_output), model_runner_output)
    except Exception:
        catalog.abort(dispatched=True)
        raise


NPUPlatform.check_and_update_config = classmethod(_check_and_update_config)
Scheduler.__init__ = _scheduler_init
Scheduler.schedule = _schedule
Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
KVCacheManager.get_computed_blocks = _get_computed_blocks
KVCacheManager.allocate_slots = _allocate_slots
Scheduler.update_from_output = _update_from_output
