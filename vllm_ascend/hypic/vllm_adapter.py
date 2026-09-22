"""Explicit dataclass messages for the synchronous vLLM V1 PIC adapter.

Keep scheduler accounting logical. Worker messages carry physical query counts
and an explicit copy of the logical counts for scheduler completion accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.hypic.protocol import PicAcknowledgement, PicStep


@dataclass
class PicSchedulerOutput(SchedulerOutput):
    pic_plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    pic_step: PicStep | None = None
    pic_logical_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class PicModelRunnerOutput(ModelRunnerOutput):
    pic_ack: PicAcknowledgement | None = None


def _preserve_extensions(source: Any, target: Any) -> Any:
    """Retain unrelated plugin metadata (e.g. profiling timings) on copies."""
    declared = {f.name for f in fields(target)}
    for name, value in vars(source).items():
        if name not in declared:
            setattr(target, name, value)
    return target


def pack_scheduler_output(
    output: SchedulerOutput, plans: dict[str, dict[str, Any]], step: PicStep
) -> PicSchedulerOutput:
    counts = dict(output.num_scheduled_tokens)
    for request_id, plan in plans.items():
        if counts[request_id] != plan["logical_advance"]:
            raise RuntimeError("PIC requires atomic full-prompt admission")
        counts[request_id] = plan["num_query_tokens"]
    result = PicSchedulerOutput(
        **{
            f.name: getattr(output, f.name)
            for f in fields(SchedulerOutput)
            if f.name not in {"num_scheduled_tokens", "total_num_scheduled_tokens"}
        },
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        pic_plans=plans,
        pic_step=step,
        pic_logical_tokens=dict(output.num_scheduled_tokens),
    )
    return _preserve_extensions(output, result)


def logical_scheduler_output(output: PicSchedulerOutput) -> PicSchedulerOutput:
    result = replace(
        output,
        num_scheduled_tokens=output.pic_logical_tokens,
        total_num_scheduled_tokens=sum(output.pic_logical_tokens.values()),
    )
    return _preserve_extensions(output, result)


def worker_input_output(output: PicSchedulerOutput) -> PicSchedulerOutput:
    """Adapt existing V1 suffix metadata without altering scheduler state.

    V1 constructs seq_len = metadata_base + query_len. Supply N-Q only in
    this worker-local view, then replace the gathered token IDs/positions with
    the explicit sparse plan before forward. This is not an APC hit count.
    """
    result = replace(
        output,
        scheduled_new_reqs=[
            replace(
                data,
                num_computed_tokens=(
                    output.pic_plans[data.req_id]["num_reused_tokens"]
                    if data.req_id in output.pic_plans
                    else data.num_computed_tokens
                ),
            )
            for data in output.scheduled_new_reqs
        ],
    )
    return _preserve_extensions(output, result)


def acknowledge_output(output: ModelRunnerOutput, ack: PicAcknowledgement) -> PicModelRunnerOutput:
    if not isinstance(output, ModelRunnerOutput):
        raise RuntimeError("PIC requires synchronous ModelRunnerOutput")
    result = PicModelRunnerOutput(
        **{f.name: getattr(output, f.name) for f in fields(ModelRunnerOutput)},
        pic_ack=ack,
    )
    return _preserve_extensions(output, result)
