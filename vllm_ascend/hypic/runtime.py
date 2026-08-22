"""Per-forward HYPIC context shared by attention and GDN layers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from vllm_ascend.hypic.cache import DeviceSegmentCache


@dataclass(frozen=True)
class HypicBatchContext:
    """Sparse prefill plans and device cache for the active model forward."""

    plans: dict[str, dict[str, Any]]
    request_ids: tuple[str, ...]
    cache: DeviceSegmentCache


_CURRENT: ContextVar[HypicBatchContext | None] = ContextVar("vllm_ascend_hypic_context", default=None)


def current_hypic_context() -> HypicBatchContext | None:
    """Return the active HYPIC context, if this is a HYPIC prefill."""
    return _CURRENT.get()


def set_hypic_context(context: HypicBatchContext | None) -> None:
    """Set or clear context at the model-runner input preparation boundary."""
    _CURRENT.set(context)


@contextmanager
def use_hypic_context(context: HypicBatchContext | None) -> Iterator[None]:
    """Install HYPIC context for exactly one model forward."""
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)
