"""Scheduler-owned PIC slots and validated worker mirrors.

Only one step may be in flight. Tensor pools keep their existing fixed layout;
this module owns metadata, never tensors or allocation in the forward path.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from vllm_ascend.hypic.planner import refresh_plan_queries, validate_plan
from vllm_ascend.hypic.protocol import (
    PIC_PROTOCOL_VERSION,
    PicAcknowledgement,
    PicStep,
    Reservation,
    SlotRef,
)


@dataclass(frozen=True)
class Entry:
    ref: SlotRef
    tokens: tuple[int, ...]


class PicCatalog:
    """The sole authority for slot assignment, eviction and publication."""

    def __init__(self, capacity: int, *, epoch: str | None = None) -> None:
        if capacity <= 0:
            raise ValueError("PIC capacity must be positive")
        self.capacity = capacity
        self.epoch = epoch or uuid4().hex
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.generations = [0] * capacity
        self.pending: PicStep | None = None
        self._projected: OrderedDict[str, Entry] | None = None
        self._next_step = 1
        self.failed = False

    @property
    def ready(self) -> dict[str, tuple[int, ...]]:
        # Victims of an in-flight write must never be advertised as readable.
        unavailable = (
            {reservation.victim.key for reservation in self.pending.fills if reservation.victim is not None}
            if self.pending
            else set()
        )
        return {key: entry.tokens for key, entry in self.entries.items() if key not in unavailable}

    def prepare(self, plans: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], PicStep]:
        """Reserve admitted plans atomically; over-capacity misses compute only.

        Duplicate cold occurrences compute independently. The first occurrence
        in the scheduler's deterministic order is the only writer of a slot.
        Every hit is pinned until acknowledgement, regardless of worker order.
        """
        if self.failed:
            raise RuntimeError("PIC failed in flight; restart the engine")
        if self.pending is not None:
            raise RuntimeError("PIC cannot schedule before the previous acknowledgement")
        # Reservations belong to request occurrences, even if callers reuse
        # the same immutable input plan for identical requests. A single
        # deepcopy of the mapping would preserve cross-request aliases.
        prepared = {request_id: deepcopy(plan) for request_id, plan in plans.items()}
        projected = self.entries.copy()
        pinned: set[str] = set()
        reads: list[SlotRef] = []
        for plan in prepared.values():
            validate_plan(plan)
            for segment in plan["segments"]:
                segment["store"] = False
                if not segment["hit"]:
                    continue
                key = segment["hash"]
                entry = projected.get(key)
                if entry is None or entry.tokens != tuple(segment["token_ids"]):
                    raise RuntimeError("PIC hit changed between lookup and admission")
                if key not in pinned:
                    reads.append(entry.ref)
                    pinned.add(key)
                projected.move_to_end(key)
                self._assign(segment, entry.ref)

        fills: list[Reservation] = []
        occupied = {entry.ref.slot for entry in projected.values()}
        free = [slot for slot in range(self.capacity) if slot not in occupied]
        generations = self.generations.copy()
        for plan in prepared.values():
            for segment in plan["segments"]:
                if segment["hit"] or not segment["cacheable"]:
                    continue
                key = segment["hash"]
                # The token-equality collision guard also covers fills. Never
                # overwrite an existing entry just because its digest matches.
                if key in projected:
                    continue
                victim = None
                if free:
                    slot = free.pop(0)
                else:
                    candidate = next((k for k in projected if k not in pinned), None)
                    if candidate is None:
                        continue
                    victim = projected.pop(candidate).ref
                    slot = victim.slot
                generations[slot] += 1
                ref = SlotRef(key, slot, generations[slot])
                projected[key] = Entry(ref, tuple(segment["token_ids"]))
                pinned.add(key)
                fills.append(Reservation(ref, victim))
                segment["store"] = True
                self._assign(segment, ref)
            refresh_plan_queries(plan)

        step = PicStep(self.epoch, self._next_step, tuple(reads), tuple(fills))
        self._next_step += 1
        self.generations = generations
        self.pending = step
        self._projected = projected
        return prepared, step

    @staticmethod
    def _assign(segment: dict[str, Any], ref: SlotRef) -> None:
        segment["slot_id"] = ref.slot
        segment["slot_generation"] = ref.generation

    def commit(self, ack: PicAcknowledgement, *, expected_ranks: int) -> None:
        step = self.pending
        if step is None or (ack.epoch, ack.step_id) != (step.epoch, step.step_id):
            raise RuntimeError("PIC stale or unsolicited acknowledgement")
        if ack.fills != tuple(fill.target for fill in step.fills) or ack.completed_ranks != expected_ranks:
            raise RuntimeError("PIC incomplete fill acknowledgement")
        assert self._projected is not None
        self.entries = self._projected
        self.pending = None
        self._projected = None

    def abort(self, *, dispatched: bool) -> None:
        """Do not resurrect victims whose tensor slots may be overwritten.

        A dispatched failure requires an engine restart before more work:
        worker state may differ across ranks. The adapter fails closed.
        """
        if dispatched:
            self.failed = True
        if self.pending is not None and dispatched:
            for fill in self.pending.fills:
                if fill.victim is not None:
                    self.entries.pop(fill.victim.key, None)
        self.pending = None
        self._projected = None


class PicDeviceCache:
    """Worker mirror: explicit slots only, no independent LRU or eviction."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.epoch: str | None = None
        self.last_step = 0
        self.slots: dict[int, SlotRef] = {}
        self.by_key: dict[str, SlotRef] = {}
        self.pending: PicStep | None = None
        self.completed_layers: set[str] = set()

    def prepare(self, plans: dict[str, dict[str, Any]], step: PicStep) -> None:
        if step.version != PIC_PROTOCOL_VERSION:
            raise RuntimeError("PIC unsupported step version")
        if self.pending is not None:
            raise RuntimeError("PIC worker still has an incomplete step")
        if self.epoch is not None and step.epoch != self.epoch:
            raise RuntimeError("PIC engine epoch changed; restart worker")
        if step.step_id <= self.last_step:
            raise RuntimeError("PIC stale step")
        projected = self.slots.copy()
        touched: set[int] = set()
        for ref in step.reads:
            if projected.get(ref.slot) != ref:
                raise RuntimeError("PIC read slot/generation divergence")
            touched.add(ref.slot)
        for fill in step.fills:
            ref = fill.target
            if not 0 <= ref.slot < self.capacity or ref.slot in touched:
                raise RuntimeError("PIC write would overwrite a leased slot")
            if projected.get(ref.slot) != fill.victim:
                raise RuntimeError("PIC victim divergence before overwrite")
            if ref.generation <= (fill.victim.generation if fill.victim else 0):
                raise RuntimeError("PIC invalid slot generation")
            projected[ref.slot] = ref
            touched.add(ref.slot)
        if len({ref.key for ref in projected.values()}) != len(projected):
            raise RuntimeError("PIC duplicate entry assigned to different slots")
        read_set = set(step.reads)
        fill_set = {fill.target for fill in step.fills}
        writers: set[SlotRef] = set()
        used_reads: set[SlotRef] = set()
        for plan in plans.values():
            validate_plan(plan)
            for segment in plan["segments"]:
                if not segment["hit"] and not segment.get("store", False):
                    continue
                ref = SlotRef(segment["hash"], segment["slot_id"], segment["slot_generation"])
                if projected.get(ref.slot) != ref:
                    raise RuntimeError("PIC plan references an invalid slot")
                if segment["hit"]:
                    if ref not in read_set:
                        raise RuntimeError("PIC read has no lease")
                    used_reads.add(ref)
                else:
                    if ref not in fill_set or ref in writers:
                        raise RuntimeError("PIC fill has no unique writer")
                    writers.add(ref)
        if writers != fill_set or used_reads != read_set:
            raise RuntimeError("PIC step and request plans disagree")
        # All validation precedes changes to the mirror (and tensor writes).
        self.slots = projected
        self.by_key = {ref.key: ref for ref in projected.values()}
        self.epoch = step.epoch
        self.last_step = step.step_id
        self.pending = step
        self.completed_layers.clear()

    def lookup(self, key: str) -> int | None:
        ref = self.by_key.get(key)
        return ref.slot if ref else None

    def mark_layer(self, name: str) -> None:
        if self.pending is None or name in self.completed_layers:
            raise RuntimeError(f"PIC unexpected or duplicate layer completion: {name}")
        self.completed_layers.add(name)

    def acknowledge(self, *, completed_ranks: int, expected_layers: set[str]) -> PicAcknowledgement:
        if self.pending is None:
            raise RuntimeError("PIC no worker step to acknowledge")
        if not expected_layers or self.completed_layers != expected_layers or completed_ranks < 1:
            raise RuntimeError("PIC incomplete layer writes")
        step = self.pending
        self.pending = None
        return PicAcknowledgement(step.epoch, step.step_id, tuple(fill.target for fill in step.fills), completed_ranks)
