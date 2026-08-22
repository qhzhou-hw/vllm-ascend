"""Scheduler and configuration patches for opt-in HYPIC execution."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.hypic.cache import SegmentCatalog
from vllm_ascend.hypic.config import get_hypic_config
from vllm_ascend.hypic.planner import build_plan
from vllm_ascend.platform import NPUPlatform

logger = init_logger(__name__)


_ORIGINAL_CHECK_CONFIG = NPUPlatform.check_and_update_config.__func__
_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__
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

        model_config.enforce_eager = True
        vllm_config.scheduler_config.enable_chunked_prefill = False
        vllm_config.scheduler_config.max_num_seqs = 1
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
    plan = build_plan(
        request.prompt_token_ids,
        scheduler.hypic_catalog.ready,
        scheduler.hypic_config,
    )
    scheduler.hypic_catalog.touch_plan(plan)
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
        evicted: list[str] = []
        for plan in plans:
            if plan is not None:
                evicted.extend(catalog.commit(plan))
        if evicted:
            logger.debug("HYPIC scheduler evicted %d segments", len(evicted))
    return result


NPUPlatform.check_and_update_config = classmethod(_check_and_update_config)
Scheduler.__init__ = _scheduler_init
Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
KVCacheManager.get_computed_blocks = _get_computed_blocks
NewRequestData.from_request = classmethod(_new_request_from_request)
Scheduler.update_from_output = _update_from_output
