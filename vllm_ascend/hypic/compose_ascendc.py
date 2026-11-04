# SPDX-License-Identifier: Apache-2.0
"""Explicit, standalone Ascend C cached-only prototype; no engine default change.

Load the separately built shared library once with ``torch.ops.load_library``.
A prepared plan owns immutable metadata only, not pool contents or slot leases.
Callers must pin their slots and order writes until the consuming stream finishes.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch

_DIM = 128
_INT32_MAX = 2**31 - 1


def _check_pools(zero_pool: torch.Tensor, transition_pool: torch.Tensor) -> None:
    if zero_pool.device.type != "npu" or transition_pool.device != zero_pool.device:
        raise ValueError("cached compose requires S/T on the same NPU")
    if any(t.dtype != torch.float32 or not t.is_contiguous() for t in (zero_pool, transition_pool)):
        raise ValueError("cached compose requires contiguous FP32 pools")
    if (
        zero_pool.ndim != 4
        or zero_pool.shape[1] <= 0
        or zero_pool.shape[2:] != (_DIM, _DIM)
        or transition_pool.shape != zero_pool.shape
    ):
        raise ValueError("cached compose requires S/T [slots, heads, 128, 128]")
    if max(zero_pool.shape[:2]) > _INT32_MAX:
        raise ValueError("pool dimensions exceed int32")


def _packet_rows(slot_sequences: Sequence[Sequence[int]], pool_slots: int) -> list[list[int]]:
    # Freeze the caller's sequences; later mutation must not change a prepared plan.
    sequences = []
    for sequence in slot_sequences:
        slots = []
        for slot in sequence:
            if isinstance(slot, (bool, torch.Tensor)):
                raise ValueError("slot IDs must be host integers, not bools or tensors")
            try:
                slot = operator.index(slot)
            except TypeError as exc:
                raise ValueError("slot IDs must be integers") from exc
            if not 0 <= slot < pool_slots:
                raise ValueError(f"cache slot {slot} is outside [0, {pool_slots})")
            slots.append(slot)
        sequences.append(slots)
    width = max(map(len, sequences), default=0)
    if max(len(sequences), width + 1) > _INT32_MAX:
        raise ValueError("metadata dimensions exceed int32")
    # Padding is never read, including when pool_slots == 0.
    return [[len(slots), *slots, *([-1] * (width - len(slots)))] for slots in sequences]


class CachedComposeAscendC:
    """Prepared fixed-order plan for K=V=128 FP32 pools on Ascend910B.

    Every call allocates its own output and system workspace. No output aliasing,
    mutable global cache, cross-call workspace reuse, or silent backend fallback.
    Use on the creating stream; pool writers must obey normal stream ordering.
    """

    def __init__(
        self,
        zero_pool: torch.Tensor,
        transition_pool: torch.Tensor,
        slot_sequences: Sequence[Sequence[int]],
        *,
        tile_rows: int = 0,
        ops_namespace: str = "hypic_ascendc",
    ):
        _check_pools(zero_pool, transition_pool)
        if isinstance(tile_rows, (bool, torch.Tensor)):
            raise ValueError("tile_rows must be a host integer: 0 (auto), 32, 64 or 128")
        try:
            tile_rows = operator.index(tile_rows)
        except TypeError as exc:
            raise ValueError("tile_rows must be a host integer: 0 (auto), 32, 64 or 128") from exc
        if tile_rows not in (0, 32, 64, 128):
            raise ValueError("tile_rows must be 0 (auto), 32, 64 or 128")
        self._ops = getattr(torch.ops, ops_namespace)
        rows = _packet_rows(slot_sequences, zero_pool.shape[0])
        self._shape = zero_pool.shape
        self._device = zero_pool.device
        self._batch = len(rows)
        if self._batch * self._shape[1] * (_DIM // 32) > _INT32_MAX:
            raise ValueError("task count exceeds int32")
        # Lazy use of the explicitly loaded extension keeps CPU-side imports valid.
        if not hasattr(self._ops, "cached_compose"):
            raise RuntimeError("load libhypic_compose_ascendc.so with torch.ops.load_library before preparing a plan")
        with torch.npu.device(self._device):
            self._stream = torch.npu.current_stream(self._device)
            host_tiling, self._workspace_bytes, self._max_blocks, self._tile_rows = self._ops.make_tiling(
                tile_rows, self._batch * self._shape[1]
            )
            self._tiling = host_tiling.to(self._device)
            self._packet = torch.tensor(rows, dtype=torch.int32, device=self._device).reshape(
                self._batch, len(rows[0]) if rows else 1
            )

    def __call__(self, zero_pool: torch.Tensor, transition_pool: torch.Tensor) -> torch.Tensor:
        _check_pools(zero_pool, transition_pool)
        if zero_pool.device != self._device or zero_pool.shape != self._shape:
            raise ValueError("pool device/shape differs from the prepared plan")
        with torch.npu.device(self._device):
            if torch.npu.current_stream(self._device) != self._stream:
                raise ValueError("use a prepared cached-compose plan on its creating NPU stream")
            return self._ops.cached_compose(
                zero_pool,
                transition_pool,
                self._packet,
                self._tiling,
                self._workspace_bytes,
                self._max_blocks,
                self._tile_rows,
            )


def compose_cached_ascendc(
    zero_pool: torch.Tensor,
    transition_pool: torch.Tensor,
    slot_sequences: Sequence[Sequence[int]],
    *,
    tile_rows: int = 0,
    ops_namespace: str = "hypic_ascendc",
) -> torch.Tensor:
    """Allocation/metadata-inclusive API; accepts nonnegative pool slot IDs."""
    return CachedComposeAscendC(
        zero_pool, transition_pool, slot_sequences, tile_rows=tile_rows, ops_namespace=ops_namespace
    )(zero_pool, transition_pool)


def _tree_levels(sequences: list[list[int]], merge_levels: int) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Adjacent merges, odd carries, and dead-T pruning along each left spine."""
    levels = []
    while True:
        packets, next_sequences = [], []
        for seq in sequences:
            start = len(packets)
            if not seq:
                packets.append([-1, -1, 0])
            for i in range(0, len(seq), 2):
                packets.append([seq[i], seq[i + 1] if i + 1 < len(seq) else -1, int(i != 0)])
            next_sequences.append(list(range(start, len(packets))))
        levels.append(packets)
        if all(len(seq) <= 1 for seq in next_sequences):
            return levels, []
        if merge_levels and len(levels) == merge_levels:
            return levels, next_sequences
        sequences = next_sequences


class CachedComposeTreeAscendC:
    """Experimental ordered binary tree; FP32 reassociation is NOT bitwise exact.

    Pool ownership, fixed stream, and independent outputs match the sequential
    plan. Intermediate summaries are per-call, never cached between requests.
    ``single_kernel=True`` runs the full tree or tree/chain hybrid in one batch-
    scheduled launch on Ascend910B/CANN 9, with all-AIV barriers between levels.
    """

    def __init__(
        self,
        zero_pool: torch.Tensor,
        transition_pool: torch.Tensor,
        slot_sequences: Sequence[Sequence[int]],
        *,
        tile_rows: int = 0,
        ops_namespace: str = "hypic_ascendc",
        merge_levels: int = 0,
        single_kernel: bool = False,
    ):
        _check_pools(zero_pool, transition_pool)
        if isinstance(tile_rows, (bool, torch.Tensor)):
            raise ValueError("tile_rows must be a host integer")
        try:
            tile_rows = operator.index(tile_rows)
        except TypeError as exc:
            raise ValueError("tile_rows must be a host integer") from exc
        if tile_rows not in (0, 32, 64, 128):
            raise ValueError("tile_rows must be 0 (auto), 32, 64 or 128")
        if isinstance(merge_levels, (bool, torch.Tensor)):
            raise ValueError("merge_levels must be a host integer")
        try:
            merge_levels = operator.index(merge_levels)
        except TypeError as exc:
            raise ValueError("merge_levels must be a host integer") from exc
        if not 0 <= merge_levels <= 31:
            raise ValueError("merge_levels must be 0 (full tree) or 1..31")
        if not isinstance(single_kernel, bool):
            raise ValueError("single_kernel must be a bool")
        self._single_kernel = single_kernel
        frozen = _packet_rows(slot_sequences, zero_pool.shape[0])
        levels, tail = _tree_levels([row[1 : row[0] + 1] for row in frozen], merge_levels)
        self._shape, self._device = zero_pool.shape, zero_pool.device
        self._ops = getattr(torch.ops, ops_namespace)
        if not hasattr(self._ops, "cached_compose_tree"):
            raise RuntimeError("load the tree-enabled libhypic_compose_ascendc.so before preparing a tree plan")
        self._row_tiles = []
        self._packets, self._tilings = [], []
        with torch.npu.device(self._device):
            self._stream = torch.npu.current_stream(self._device)
            if single_kernel:
                self._prepare_single_tree(levels, tail, tile_rows)
                return
            parts, packet_slices, tiling_slices = [], [], []
            tiling_cache = {}
            byte_offset = 0
            for level in levels:
                independent_heads = len(level) * self._shape[1]
                if independent_heads * (_DIM // 32) > _INT32_MAX:
                    raise ValueError("tree task count exceeds int32")
                rows = tile_rows
                if not tiling_cache:
                    first, self._workspace_bytes, self._max_blocks, selected = self._ops.make_tree_tiling(
                        rows, independent_heads
                    )
                    tiling_cache[selected] = first
                if rows == 0:
                    rows = 32
                    while rows < _DIM and independent_heads * (_DIM // rows) > 2 * self._max_blocks:
                        rows *= 2
                if rows not in tiling_cache:
                    tiling_cache[rows] = self._ops.make_tree_tiling(rows, independent_heads)[0]
                host_packet = torch.tensor(level, dtype=torch.int32).reshape(-1, 3).view(torch.uint8).flatten()
                host_tiling = tiling_cache[rows]
                parts.extend((host_packet, host_tiling))
                packet_slices.append((byte_offset, host_packet.numel(), len(level)))
                byte_offset += host_packet.numel()
                tiling_slices.append((byte_offset, host_tiling.numel()))
                byte_offset += host_tiling.numel()
                self._row_tiles.append(rows)
            tail_offset = byte_offset
            self._tail_rows = 0
            tail_width, tail_bytes, tail_tiling_bytes = 1, 0, 0
            if tail:
                tail_data = _packet_rows(tail, len(levels[-1]))
                tail_width = len(tail_data[0])
                host_tail = torch.tensor(tail_data, dtype=torch.int32).view(torch.uint8).flatten()
                host_tail_tiling, _, _, self._tail_rows = self._ops.make_tiling(tile_rows, len(tail) * self._shape[1])
                parts.extend((host_tail, host_tail_tiling))
                tail_bytes, tail_tiling_bytes = host_tail.numel(), host_tail_tiling.numel()
            # One H2D copy for all immutable levels and tilings, no per-level sync.
            self._metadata = torch.cat(parts).to(self._device)
            for offset, size, nodes in packet_slices:
                self._packets.append(self._metadata[offset : offset + size].view(torch.int32).reshape(nodes, 3))
            for offset, size in tiling_slices:
                self._tilings.append(self._metadata[offset : offset + size])
            self._tail_packet = (
                self._metadata[tail_offset : tail_offset + tail_bytes].view(torch.int32).reshape(len(tail), tail_width)
            )
            self._tail_tiling = self._metadata[tail_offset + tail_bytes : tail_offset + tail_bytes + tail_tiling_bytes]

    def _prepare_single_tree(self, levels: list[list[list[int]]], tail: list[list[int]], tile_rows: int) -> None:
        if not hasattr(self._ops, "cached_compose_tree_single"):
            raise RuntimeError("load the single-kernel tree-enabled extension before preparing this plan")
        self._node_counts = [len(level) for level in levels]
        if max(self._node_counts) * self._shape[1] * (_DIM // 32) > _INT32_MAX:
            raise ValueError("single tree task count exceeds int32")
        host_tiling, self._workspace_bytes, self._max_blocks, _ = self._ops.make_tree_tiling(
            tile_rows, self._node_counts[0] * self._shape[1]
        )
        header, offset = [], len(levels)
        for nodes in self._node_counts:
            rows = tile_rows or 32
            if tile_rows == 0:
                while rows < _DIM and nodes * self._shape[1] * (_DIM // rows) > 2 * self._max_blocks:
                    rows *= 2
            self._row_tiles.append(rows)
            header.append([nodes, rows, offset])
            offset += nodes
        if offset > _INT32_MAX // 3:
            raise ValueError("single tree metadata exceeds int32")
        host_packet = torch.tensor(header + [node for level in levels for node in level], dtype=torch.int32)
        packet_bytes = host_packet.numel() * host_packet.element_size()
        parts = [host_packet.view(torch.uint8).flatten(), host_tiling]
        tail_offset = packet_bytes + host_tiling.numel()
        self._single_tail_packet, self._single_tail_tiling, self._single_tail_rows = None, None, 0
        if tail:
            if not hasattr(self._ops, "make_chain_tiling"):
                raise RuntimeError("rebuild the extension with single-kernel hybrid support")
            tail_data = _packet_rows(tail, len(levels[-1]))
            host_tail = torch.tensor(tail_data, dtype=torch.int32).view(torch.uint8).flatten()
            host_tail_tiling, workspace_bytes, _, self._single_tail_rows = self._ops.make_chain_tiling(
                tile_rows, len(tail) * self._shape[1], self._row_tiles[0]
            )
            self._workspace_bytes = max(self._workspace_bytes, workspace_bytes)
            parts.extend((host_tail, host_tail_tiling))
        self._metadata = torch.cat(parts).to(self._device)
        self._single_packet = self._metadata[:packet_bytes].view(torch.int32).reshape(-1, 3)
        self._single_tiling = self._metadata[packet_bytes:tail_offset]
        if tail:
            tail_end = tail_offset + host_tail.numel()
            self._single_tail_packet = self._metadata[tail_offset:tail_end].view(torch.int32).reshape(len(tail), -1)
            self._single_tail_tiling = self._metadata[tail_end:]

    def __call__(self, zero_pool: torch.Tensor, transition_pool: torch.Tensor) -> torch.Tensor:
        _check_pools(zero_pool, transition_pool)
        if zero_pool.shape != self._shape or zero_pool.device != self._device:
            raise ValueError("pool device/shape differs from the prepared tree plan")
        with torch.npu.device(self._device):
            if torch.npu.current_stream(self._device) != self._stream:
                raise ValueError("use a prepared tree plan on its creating NPU stream")
            if self._single_kernel:
                return self._ops.cached_compose_tree_single(
                    zero_pool,
                    transition_pool,
                    self._single_packet,
                    self._single_tiling,
                    self._node_counts,
                    self._row_tiles,
                    self._workspace_bytes,
                    self._max_blocks,
                    self._single_tail_packet,
                    self._single_tail_tiling,
                    self._single_tail_rows,
                )
            return self._ops.cached_compose_tree(
                zero_pool,
                transition_pool,
                self._packets,
                self._tilings,
                self._row_tiles,
                self._workspace_bytes,
                self._max_blocks,
                self._tail_packet,
                self._tail_tiling,
                self._tail_rows,
            )


def compose_cached_tree_ascendc(
    zero_pool: torch.Tensor,
    transition_pool: torch.Tensor,
    slot_sequences: Sequence[Sequence[int]],
    *,
    tile_rows: int = 0,
    ops_namespace: str = "hypic_ascendc",
    merge_levels: int = 0,
    single_kernel: bool = False,
) -> torch.Tensor:
    """Tree including metadata preparation; never silently replaces sequential composition."""
    return CachedComposeTreeAscendC(
        zero_pool,
        transition_pool,
        slot_sequences,
        tile_rows=tile_rows,
        ops_namespace=ops_namespace,
        merge_levels=merge_levels,
        single_kernel=single_kernel,
    )(zero_pool, transition_pool)
