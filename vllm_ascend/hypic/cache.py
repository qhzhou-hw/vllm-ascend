"""Host and device cache registries used by HYPIC."""

from __future__ import annotations

import heapq
from collections import OrderedDict
from collections.abc import Iterable
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


class DeviceSegmentCache:
    """Worker-side deterministic mapping from segment hashes to pool slots.

    Layer tensors live in fixed-size buffers allocated with the model. Cache
    entries contain only slot ids, matching SGLang PICache's ownership model and
    keeping runtime segment growth out of the dynamic NPU allocator.
    """

    def __init__(self, max_segments: int) -> None:
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.max_segments = max_segments
        self.segments: OrderedDict[str, int] = OrderedDict()
        self._free_slots = list(range(max_segments))
        heapq.heapify(self._free_slots)

    def lookup(self, digest: str) -> int | None:
        """Return a resident slot without changing the forward's LRU state."""
        return self.segments.get(digest)

    def touch(self, digest: str) -> int | None:
        """Return a resident slot and refresh its scheduler-matching LRU."""
        slot = self.segments.get(digest)
        if slot is not None:
            self.segments.move_to_end(digest)
        return slot

    def reserve(self, digest: str) -> int:
        """Return a stable slot, reusing the least-recently-used slot if full."""
        existing = self.touch(digest)
        if existing is not None:
            return existing
        if self._free_slots:
            slot = heapq.heappop(self._free_slots)
        else:
            _, slot = self.segments.popitem(last=False)
        self.segments[digest] = slot
        return slot

    def prepare(self, plans: Iterable[dict[str, Any]]) -> None:
        """Pin every cacheable segment used by one packed model forward.

        Slot assignment happens before the first layer executes. This keeps a
        miss in an early layer from evicting a hit that a later layer still has
        to read. The scheduler's token budget bounds the number of active
        cacheable segments, so they must all fit in the static pool.
        """
        active: set[str] = set()
        misses: list[str] = []
        seen_misses: set[str] = set()
        for plan in plans:
            for segment in plan["segments"]:
                if not segment["cacheable"]:
                    continue
                digest = str(segment["hash"])
                active.add(digest)
                if segment["hit"] and self.touch(digest) is None:
                    raise RuntimeError(
                        "HYPIC scheduler/worker cache divergence for "
                        f"segment {digest}"
                    )
                if not segment["hit"] and digest not in seen_misses:
                    misses.append(digest)
                    seen_misses.add(digest)
        if len(active) > self.max_segments:
            raise RuntimeError(
                "HYPIC packed prefill needs "
                f"{len(active)} cache slots, but only {self.max_segments} "
                "were configured"
            )
        for digest in misses:
            self.reserve(digest)

    def discard(self, digests: Iterable[str]) -> None:
        """Discard scheduler-evicted entries and return their slots to the pool."""
        for digest in digests:
            slot = self.segments.pop(digest, None)
            if slot is not None:
                heapq.heappush(self._free_slots, slot)
