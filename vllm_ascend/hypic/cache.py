"""Host and device cache registries used by HYPIC."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class SegmentCatalog:
    """Scheduler-side LRU catalog of worker-resident segment hashes."""

    def __init__(self, max_segments: int) -> None:
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.max_segments = max_segments
        self.ready: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    def touch_plan(self, plan: dict[str, Any]) -> None:
        """Refresh LRU state for hits selected by a plan."""
        for segment in plan["segments"]:
            digest = segment["hash"]
            if segment["hit"] and digest in self.ready:
                self.ready.move_to_end(digest)

    def commit(self, plan: dict[str, Any]) -> list[str]:
        """Mark completed misses ready and return hashes evicted by LRU."""
        for segment in plan["segments"]:
            if segment["cacheable"] and not segment["hit"]:
                digest = segment["hash"]
                self.ready[digest] = tuple(segment["token_ids"])
                self.ready.move_to_end(digest)
        evicted: list[str] = []
        while len(self.ready) > self.max_segments:
            digest, _ = self.ready.popitem(last=False)
            evicted.append(digest)
        return evicted


@dataclass
class LayerSegmentState:
    """One layer's public HYPIC state for a prompt segment."""

    key: Any | None = None
    value: Any | None = None
    conv_tail: Any | None = None
    zero_state: Any | None = None
    transition: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DeviceSegmentCache:
    """Worker-side per-layer tensor cache with deterministic segment LRU."""

    def __init__(self, max_segments: int) -> None:
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.max_segments = max_segments
        self.segments: OrderedDict[str, dict[str, LayerSegmentState]] = OrderedDict()

    def get(self, digest: str, layer_name: str) -> LayerSegmentState | None:
        """Return cached layer state and refresh the segment's LRU position."""
        layers = self.segments.get(digest)
        if layers is None:
            return None
        self.segments.move_to_end(digest)
        return layers.get(layer_name)

    def put(
        self,
        digest: str,
        layer_name: str,
        state: LayerSegmentState,
    ) -> list[str]:
        """Store layer state and evict complete least-recently-used segments."""
        self.segments.setdefault(digest, {})[layer_name] = state
        self.segments.move_to_end(digest)
        evicted: list[str] = []
        while len(self.segments) > self.max_segments:
            old_digest, _ = self.segments.popitem(last=False)
            evicted.append(old_digest)
        return evicted

    def discard(self, digests: Iterable[str]) -> None:
        """Discard scheduler-evicted segments if an eviction list is supplied."""
        for digest in digests:
            self.segments.pop(digest, None)
