"""Scheduler and configuration patches for opt-in HYPIC execution."""

from __future__ import annotations

import math
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.hypic.cache import SegmentCatalog
from vllm_ascend.hypic.config import get_hypic_config
from vllm_ascend.hypic.planner import build_plan
from vllm_ascend.platform import NPUPlatform

logger = init_logger(__name__)


_ORIGINAL_CHECK_CONFIG = NPUPlatform.check_and_update_config.__func__
_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__
_ORIGINAL_SCHEDULE = Scheduler.schedule
_ORIGINAL_MAMBA_SPLIT = Scheduler._mamba_block_aligned_split
_ORIGINAL_GET_COMPUTED_BLOCKS = KVCacheManager.get_computed_blocks
_ORIGINAL_NEW_REQUEST = NewRequestData.from_request.__func__
_ORIGINAL_UPDATE_FROM_OUTPUT = Scheduler.update_from_output


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

        max_batched_tokens = int(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        required_slots = max(
            1, math.ceil(max_batched_tokens / config.chunk_size) - 1
        )
        if config.max_cache_segments < required_slots:
            raise ValueError(
                "hypic_config.max_cache_segments must be at least "
                f"{required_slots} for max_num_batched_tokens="
                f"{max_batched_tokens} and chunk_size={config.chunk_size}; "
                "all cacheable segments in one packed forward need stable slots"
            )

        model_config.enforce_eager = True
        # SegmentCatalog commits misses only after the corresponding model
        # output. Async scheduling could plan another batch against that
        # uncommitted state while the worker has already reserved its slots.
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.scheduler_config.enable_chunked_prefill = False
        # Hybrid models otherwise retain a 2048-token scheduling cap even when
        # max_num_batched_tokens is larger. HYPIC must plan and execute a whole
        # prompt atomically because its query positions are non-contiguous.
        vllm_config.scheduler_config.max_num_scheduled_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Keep vLLM prefix caching enabled so its hybrid-cache page-size
        # validation remains satisfied.  HYPIC bypasses standard cache hits in
        # ``_get_computed_blocks`` and owns segment reuse independently.
        vllm_config.cache_config.mamba_cache_mode = "align"
        logger.info(
            "Enabled HYPIC transition_rope_recompute with chunk_size=%d, seam=%d",
            config.chunk_size,
            config.seam_sink_tokens,
        )


def _scheduler_init(self: Scheduler, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_SCHEDULER_INIT(self, *args, **kwargs)
    config = get_hypic_config(self.vllm_config)
    if config.enabled:
        self.hypic_config = config
        self.hypic_catalog = SegmentCatalog(config.max_cache_segments)
        self.kv_cache_manager.hypic_scheduler = self


def _schedule(self: Scheduler, *args: Any, **kwargs: Any) -> Any:
    """Keep HYPIC prefills isolated from in-flight decode requests.

    Multiple waiting requests may still be admitted together when the running
    set is empty. Once admitted, that request group drains before the scheduler
    admits another HYPIC prefill group. This preserves ordinary batched decode
    while avoiding a mixed custom-prefill/standard-decode model forward.
    """
    if (
        hasattr(self, "hypic_catalog")
        and self.running
        and self._pause_state == PauseState.UNPAUSED
    ):
        self._pause_state = PauseState.PAUSED_NEW
        try:
            scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
        finally:
            self._pause_state = PauseState.UNPAUSED
    else:
        scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)

    catalog = getattr(self, "hypic_catalog", None)
    if catalog is not None:
        # Cache lookup may happen for a waiting request that is later rejected
        # by token or block budgets. Only admitted requests may mutate the
        # scheduler LRU, because only those plans reach the worker.
        catalog.prepare_scheduled_plans(
            plan
            for request_data in scheduler_output.scheduled_new_reqs
            if (plan := getattr(request_data, "hypic_plan", None)) is not None
        )
    return scheduler_output


def _mamba_block_aligned_split(
    self: Scheduler,
    request: Any,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    if hasattr(self, "hypic_catalog"):
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
    if request.prompt_token_ids is None:
        raise ValueError("HYPIC requires token-id prompts")
    if request.sampling_params is not None and getattr(request.sampling_params, "prompt_logprobs", None) is not None:
        raise ValueError("HYPIC does not support prompt logprobs")
    extra_args = (
        getattr(request.sampling_params, "extra_args", None)
        if request.sampling_params is not None
        else None
    ) or {}
    segment_boundaries = extra_args.get("hypic_segment_boundaries")
    plan = build_plan(
        request.prompt_token_ids,
        scheduler.hypic_catalog.ready,
        scheduler.hypic_config,
        segment_boundaries=segment_boundaries,
    )
    request.hypic_plan = plan
    return self.empty_kv_cache_blocks, int(plan["num_computed_tokens"]), 0


def _new_request_from_request(
    cls: type,
    request: Any,
    block_ids: tuple[list[int], ...],
    prefill_token_ids: list[int] | None = None,
) -> NewRequestData:
    data = _ORIGINAL_NEW_REQUEST(cls, request, block_ids, prefill_token_ids=prefill_token_ids)
    plan = getattr(request, "hypic_plan", None)
    if plan is not None:
        data.hypic_plan = plan
    return data


def _update_from_output(self: Scheduler, scheduler_output: Any, model_runner_output: Any) -> Any:
    plans = [getattr(data, "hypic_plan", None) for data in scheduler_output.scheduled_new_reqs]
    result = _ORIGINAL_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)
    catalog = getattr(self, "hypic_catalog", None)
    if catalog is not None:
        expected_evicted = [
            segment["expected_eviction"]
            for plan in plans
            if plan is not None
            for segment in plan["segments"]
            if segment.get("expected_eviction") is not None
        ]
        evicted: list[str] = []
        for plan in plans:
            if plan is not None:
                evicted.extend(catalog.commit(plan))
        if evicted != expected_evicted:
            raise RuntimeError(
                "HYPIC scheduler cache projection divergence: "
                f"expected={expected_evicted}, actual={evicted}"
            )
        expected_order = next(
            (
                plan["cache_order_after_commit"]
                for plan in reversed(plans)
                if plan is not None and "cache_order_after_commit" in plan
            ),
            None,
        )
        if expected_order is not None and tuple(catalog.ready) != tuple(
            expected_order
        ):
            raise RuntimeError(
                "HYPIC scheduler cache order projection divergence: "
                f"expected={tuple(expected_order)}, "
                f"actual={tuple(catalog.ready)}"
            )
        if evicted:
            logger.debug("HYPIC scheduler evicted %d segments", len(evicted))
    return result


NPUPlatform.check_and_update_config = classmethod(_check_and_update_config)
Scheduler.__init__ = _scheduler_init
Scheduler.schedule = _schedule
Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
KVCacheManager.get_computed_blocks = _get_computed_blocks
NewRequestData.from_request = classmethod(_new_request_from_request)
Scheduler.update_from_output = _update_from_output
