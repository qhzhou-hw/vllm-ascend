"""Versioned, pickle-safe PIC control messages (no device dependencies)."""

from __future__ import annotations

from dataclasses import dataclass

PIC_PROTOCOL_VERSION = 2


@dataclass(frozen=True)
class SlotRef:
    key: str
    slot: int
    generation: int


@dataclass(frozen=True)
class Reservation:
    target: SlotRef
    victim: SlotRef | None


@dataclass(frozen=True)
class PicStep:
    epoch: str
    step_id: int
    reads: tuple[SlotRef, ...]
    fills: tuple[Reservation, ...]
    version: int = PIC_PROTOCOL_VERSION


@dataclass(frozen=True)
class PicAcknowledgement:
    epoch: str
    step_id: int
    fills: tuple[SlotRef, ...]
    # All participating TP ranks must finish every pool-bearing layer.
    completed_ranks: int


def request_policy(request: object) -> str:
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    policy = extra.get("hypic_cache_policy", "pic")
    if policy not in {"pic", "prefix_only", "full_recompute"}:
        raise ValueError(f"Unknown hypic_cache_policy: {policy!r}")
    return policy
