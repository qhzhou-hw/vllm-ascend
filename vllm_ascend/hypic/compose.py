"""Ordered GDN state composition shared by the oracle and fused backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ComposeStep:
    """One affine update; negative sources encode cache slots as -(slot + 1)."""

    source: int
    # Save the incoming H for this replay unit; -1 means no replay output.
    replay: int = -1


def build_compose_steps(
    plan: dict[str, Any], lookup: Callable[[str], int | None], *, include_query: bool = True
) -> tuple[tuple[ComposeStep, ...], int]:
    """Keep hit interiors, fresh seams and misses in their original order."""
    block_native = plan.get("mode") == "block_native_pic"
    steps: list[ComposeStep] = []
    unit = 0
    segments = plan["segments"] if include_query else plan["segments"][:-1]
    for segment in segments:
        if segment["hit"]:
            if segment["seam"]:
                steps.append(ComposeStep(unit, unit))
                unit += 1
            slot = lookup(segment["hash"])
            if slot is None:
                raise RuntimeError(f"HYPIC GDN slot disappeared for segment {segment['hash']}")
            steps.append(ComposeStep(-int(slot) - 1))
        else:
            for _ in range(2 if segment["recompute_seam"] else 1):
                replay = unit if not block_native or segment["end"] == plan["num_tokens"] else -1
                steps.append(ComposeStep(unit, replay))
                unit += 1
    return tuple(steps), unit


def compose_reference(
    zero_states: torch.Tensor,
    transitions: torch.Tensor,
    zero_pool: torch.Tensor,
    transition_pool: torch.Tensor,
    steps: Sequence[ComposeStep],
    *,
    zero_replay: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unchanged ordered FP32 bmm/add oracle, including the first zero matmul."""
    replay = torch.zeros_like(zero_states) if zero_replay else torch.empty_like(zero_states)
    state = zero_states.new_zeros(zero_states.shape[1:], dtype=torch.float32)
    for step in steps:
        if step.replay >= 0:
            replay[step.replay].copy_(state)
        if step.source < 0:
            slot = -step.source - 1
            transition, zero = transition_pool[slot], zero_pool[slot]
        else:
            transition, zero = transitions[step.source], zero_states[step.source]
        state = torch.bmm(state.float(), transition.float()) + zero.float()
    return replay, state


def compose_fused_batch(
    zero_states: Sequence[torch.Tensor],
    transitions: Sequence[torch.Tensor],
    zero_pool: torch.Tensor,
    transition_pool: torch.Tensor,
    steps: Sequence[Sequence[ComposeStep]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Explicit opt-in NPU backend; never silently falls back after JIT failure.

    Pass pointers to the original per-request tensors and immutable cache
    pools. Do not gather/copy all hit S/T into another large temporary pool.
    """
    if not zero_states or len(zero_states) != len(transitions) or len(steps) != len(zero_states):
        raise ValueError("HYPIC composition requires equally sized nonempty request lists")
    first = zero_states[0]
    if first.ndim != 4:
        raise ValueError("HYPIC fresh zero states must have shape [units, heads, value_dim, key_dim]")
    _, heads, value_dim, key_dim = first.shape
    device = first.device
    if device.type != "npu":
        raise ValueError("HYPIC triton composition requires an NPU")
    if not (1 <= key_dim <= 128 and value_dim > 0 and heads > 0):
        raise ValueError("HYPIC triton composition supports key_dim in [1, 128] and nonempty heads/value_dim")
    tensors = [*zero_states, *transitions, zero_pool, transition_pool]
    if any(t.device != device or t.dtype != torch.float32 or not t.is_contiguous() for t in tensors):
        raise ValueError("HYPIC triton composition requires contiguous FP32 tensors on the same NPU")
    if zero_pool.ndim != 4 or zero_pool.shape[1:] != (heads, value_dim, key_dim):
        raise ValueError("HYPIC zero-state pool shape mismatch")
    if transition_pool.shape != (len(zero_pool), heads, key_dim, key_dim):
        raise ValueError("HYPIC transition pool shape mismatch")
    for zeros, transforms, sequence in zip(zero_states, transitions, steps):
        if zeros.ndim != 4 or zeros.shape[1:] != (heads, value_dim, key_dim):
            raise ValueError("HYPIC fresh zero-state shape mismatch")
        if transforms.shape != (len(zeros), heads, key_dim, key_dim):
            raise ValueError("HYPIC fresh transition shape mismatch")
        targets = [step.replay for step in sequence if step.replay >= 0]
        if len(targets) != len(set(targets)):
            raise ValueError("HYPIC composition cannot write a replay unit twice")
        for step in sequence:
            if not (-len(zero_pool) <= step.source < len(zeros)):
                raise ValueError("HYPIC composition source index out of bounds")
            if not (-1 <= step.replay < len(zeros)):
                raise ValueError("HYPIC composition replay index out of bounds")

    # Lazy import keeps the default path usable without a Triton compiler.
    from vllm_ascend.hypic.compose_triton import launch_compose

    return launch_compose(zero_states, transitions, zero_pool, transition_pool, steps)
