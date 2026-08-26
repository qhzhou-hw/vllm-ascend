"""Host and device cache registries used by HYPIC."""

from __future__ import annotations

import heapq
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

_EXPECTED_EVICTION_UNSET = object()


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

    def plan_reservations(
        self, plans: Iterable[dict[str, Any]]
    ) -> tuple[str, ...]:
        """Attach scheduler-authoritative LRU victims to packed misses.

        Hit touches for the whole packed batch must be applied first, matching
        ``DeviceSegmentCache.prepare``. The returned order is the projected
        catalog state after all plans commit successfully.
        """
        projected: OrderedDict[str, None] = OrderedDict.fromkeys(self.ready)
        for plan in plans:
            for segment in plan["segments"]:
                if not segment["cacheable"] or segment["hit"]:
                    continue
                digest = str(segment["hash"])
                evicted: str | None = None
                if digest in projected:
                    projected.move_to_end(digest)
                else:
                    if len(projected) >= self.max_segments:
                        evicted, _ = projected.popitem(last=False)
                    projected[digest] = None
                segment["expected_eviction"] = evicted
        return tuple(projected)

    def prepare_scheduled_plans(
        self, plans: Iterable[dict[str, Any]]
    ) -> None:
        """Record admitted-plan touches and attach worker LRU expectations."""
        scheduled_plans = list(plans)
        for plan in scheduled_plans:
            plan["cache_order_before"] = tuple(self.ready)
            self.touch_plan(plan)
        projected_order = self.plan_reservations(scheduled_plans)
        if scheduled_plans:
            scheduled_plans[-1]["cache_order_after_commit"] = projected_order


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

    def reserve(
        self,
        digest: str,
        expected_eviction: str | None | object = _EXPECTED_EVICTION_UNSET,
    ) -> int:
        """Reserve a slot and validate any scheduler-selected LRU victim."""
        existing = self.segments.get(digest)
        if existing is not None:
            if (
                expected_eviction is not _EXPECTED_EVICTION_UNSET
                and expected_eviction is not None
            ):
                raise RuntimeError(
                    "HYPIC scheduler/worker eviction divergence for "
                    f"segment {digest}: scheduler={expected_eviction}, "
                    "worker=None"
                )
            self.segments.move_to_end(digest)
            return existing
        worker_eviction = (
            next(iter(self.segments)) if not self._free_slots else None
        )
        if (
            expected_eviction is not _EXPECTED_EVICTION_UNSET
            and expected_eviction != worker_eviction
        ):
            raise RuntimeError(
                "HYPIC scheduler/worker eviction divergence for "
                f"segment {digest}: scheduler={expected_eviction}, "
                f"worker={worker_eviction}"
            )
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
        misses: list[tuple[str, str | None | object]] = []
        expected_order_after: tuple[str, ...] | None = None
        for plan in plans:
            expected_order = plan.get("cache_order_before")
            if expected_order is not None and tuple(self.segments) != tuple(
                expected_order
            ):
                raise RuntimeError(
                    "HYPIC scheduler/worker cache order divergence before "
                    f"packed plan: scheduler={tuple(expected_order)}, "
                    f"worker={tuple(self.segments)}"
                )
            if "cache_order_after_commit" in plan:
                expected_order_after = tuple(plan["cache_order_after_commit"])
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
                if not segment["hit"]:
                    # Preserve every miss, including duplicate hashes in a
                    # packed batch. SegmentCatalog.commit() refreshes the LRU
                    # position for each miss in plan order, so deduplicating
                    # here would make the worker evict a different segment
                    # from the one the scheduler removes.
                    misses.append(
                        (
                            digest,
                            segment.get(
                                "expected_eviction", _EXPECTED_EVICTION_UNSET
                            ),
                        )
                    )
        if len(active) > self.max_segments:
            raise RuntimeError(
                "HYPIC packed prefill needs "
                f"{len(active)} cache slots, but only {self.max_segments} "
                "were configured"
            )
        for digest, expected_eviction in misses:
            self.reserve(digest, expected_eviction)
        if (
            expected_order_after is not None
            and tuple(self.segments) != expected_order_after
        ):
            raise RuntimeError(
                "HYPIC scheduler/worker cache order divergence after packed "
                f"plans: scheduler={expected_order_after}, "
                f"worker={tuple(self.segments)}"
            )

    def discard(self, digests: Iterable[str]) -> None:
        """Apply explicit scheduler invalidations and release their slots.

        This must not be used to reconcile an independently chosen worker LRU
        victim after ``reserve``: its pool slot may already contain another
        segment by then. Normal HYPIC operation instead keeps both LRUs in
        lockstep by replaying the scheduler's complete hit/miss sequence.
        """
        for digest in digests:
            slot = self.segments.pop(digest, None)
            if slot is not None:
                heapq.heappush(self._free_slots, slot)
