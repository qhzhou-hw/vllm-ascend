"""Experimental batched FP32 Cube composition; invoked by hypic_compose_cube.

Indexed mode reads original S/T tensors through a pointer packet and uses each
request's actual length; no S/T packing. Output mode additionally writes replay
and final directly, without a history tensor or a separate scatter kernel.
Cached mode specializes empty fresh/replay inputs to compact pool-slot metadata.
Older packed controls use left identity padding and CANN or diagnostic gathers.
Not registered as a production backend; all allocation/metadata work is timed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    packet, zero_pool, transition_pool, packed, H: tl.constexpr, N: tl.constexpr, B: tl.constexpr, TILE: tl.constexpr
):
    request = tl.program_id(0)
    step = tl.program_id(1)
    lane = tl.program_id(2) * TILE + tl.arange(0, TILE)
    size: tl.constexpr = H * 128 * 128
    mask = lane < size
    zero = tl.full((TILE,), 0, tl.float32)
    transition = ((lane // 128) % 128 == lane % 128).to(tl.float32)
    address = tl.load(packet + B * 4 + (request * N + step) * 3)
    if address != 0:
        s = tl.multiple_of(address.to(tl.pointer_type(tl.float32)), 16)
        t = tl.multiple_of(tl.load(packet + B * 4 + (request * N + step) * 3 + 1).to(tl.pointer_type(tl.float32)), 16)
        zero = tl.load(s + lane, mask, other=0)
        transition = tl.load(t + lane, mask, other=0)
    offset = (request * N + step) * size + lane
    tl.store(packed + offset, zero, mask)
    tl.store(packed + B * N * size + offset, transition, mask)


@triton.jit
def _chain(packed, packed_t, history, H: tl.constexpr, N: tl.constexpr, B: tl.constexpr, HISTORY: tl.constexpr):
    request = tl.program_id(0)
    head = tl.program_id(1)
    keys = tl.arange(0, 128)
    offsets = head * 128 * 128 + keys[:, None] * 128 + keys[None, :]
    size: tl.constexpr = H * 128 * 128
    base = request * N * size + offsets
    output = history + request * HISTORY * size + offsets
    tl.store(output, tl.full((128, 128), 0, tl.float32))
    state = tl.load(packed + base)
    tl.store(output + size, state)
    for step in range(1, N):
        zero = tl.load(packed + base + step * size)
        transition = tl.load(packed_t + base + step * size)
        state = tl.dot(state, transition, input_precision="ieee") + zero
        tl.store(output + ((step + 1) % HISTORY) * size, state)


@triton.jit
def _indexed_chain(packet, history, H: tl.constexpr, N: tl.constexpr, B: tl.constexpr, HISTORY: tl.constexpr):
    """Full matrix tiles, uniform pointer loads, no hit/miss branch in Cube loop."""
    request = tl.program_id(0)
    head = tl.program_id(1)
    keys = tl.arange(0, 128)
    offsets = head * 128 * 128 + keys[:, None] * 128 + keys[None, :]
    size: tl.constexpr = H * 128 * 128
    output = history + request * HISTORY * size + offsets
    metadata = packet + B * 4 + request * N * 3
    count = tl.load(packet + request * 4 + 3).to(tl.int32)
    tl.store(output, tl.full((128, 128), 0, tl.float32))
    # Empty requests point their first S at an owned zero tensor. An if/else
    # initializer keeps too many full tiles live in this CANN compiler's UB.
    s = tl.load(metadata).to(tl.pointer_type(tl.float32))
    state = tl.load(s + offsets)
    tl.store(output + size, state)
    for step in range(1, count):
        s = tl.load(metadata + step * 3).to(tl.pointer_type(tl.float32))
        t = tl.load(metadata + step * 3 + 1).to(tl.pointer_type(tl.float32))
        zero = tl.load(s + offsets)
        transition = tl.load(t + offsets)
        state = tl.dot(state, transition, input_precision="ieee") + zero
        tl.store(output + ((step + 1) % HISTORY) * size, state)


@triton.jit
def _cached_chain(zero_pool, transition_pool, packet, final, H: tl.constexpr, N: tl.constexpr):
    """Cache-only chain: direct pool arguments and slot ids, no fresh/replay."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    metadata = packet + row * (N + 2)
    request = tl.load(metadata).to(tl.int64)
    count = tl.load(metadata + 1).to(tl.int32)
    keys = tl.arange(0, 128)
    offsets = head * 128 * 128 + keys[:, None] * 128 + keys[None, :]
    size: tl.constexpr = H * 128 * 128
    slot = tl.load(metadata + 2).to(tl.int64)
    state = tl.load(zero_pool + slot * size + offsets)
    for step in range(1, count):
        slot = tl.load(metadata + 2 + step).to(tl.int64)
        zero = tl.load(zero_pool + slot * size + offsets)
        transition = tl.load(transition_pool + slot * size + offsets)
        state = tl.dot(state, transition, input_precision="ieee") + zero
    tl.store(final + request * size + offsets, state)


def _cache_only_compose(zero_states, zero_pool, transition_pool, steps):
    """Called only after all sources are cached and all fresh/replay units absent."""
    batch = len(steps)
    heads = zero_pool.shape[1]
    active = [(row, sequence) for row, sequence in enumerate(steps) if sequence]
    shape = (batch, heads, 128, 128)
    allocate = torch.empty if len(active) == batch else torch.zeros
    final = allocate(shape, device=zero_pool.device, dtype=zero_pool.dtype)
    replay = [torch.empty_like(s) for s in zero_states]
    if not active:
        return replay, final
    count = max(len(sequence) for _, sequence in active)
    if max(batch, count, zero_pool.shape[0]) >= 2**31:
        raise ValueError("cache-only metadata exceeds int32 capacity")
    values = []
    for row, sequence in active:
        values.extend((row, len(sequence)))
        values.extend(-step.source - 1 for step in sequence)
        values.extend([0] * (count - len(sequence)))
    packet = torch.tensor(values, dtype=torch.int32, device=zero_pool.device)
    _cached_chain[(len(active), heads)](
        zero_pool, transition_pool, packet, final, heads, count, num_warps=4, enable_fp_fusion=False
    )
    return replay, final


@triton.jit
def _output_chain(packet, final, H: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    """Store only requested incoming states and the final state; no history."""
    request = tl.program_id(0)
    head = tl.program_id(1)
    keys = tl.arange(0, 128)
    offsets = head * 128 * 128 + keys[:, None] * 128 + keys[None, :]
    size: tl.constexpr = H * 128 * 128
    metadata = packet + B * 4 + request * N * 3
    count = tl.load(packet + request * 4 + 3).to(tl.int32)
    address = tl.load(metadata + 2)
    if address != 0:
        destination = address.to(tl.pointer_type(tl.float32))
        tl.store(destination + offsets, tl.full((128, 128), 0, tl.float32))
    s = tl.load(metadata).to(tl.pointer_type(tl.float32))
    state = tl.load(s + offsets)
    for step in range(1, count):
        address = tl.load(metadata + step * 3 + 2)
        if address != 0:
            destination = address.to(tl.pointer_type(tl.float32))
            tl.store(destination + offsets, state)
        s = tl.load(metadata + step * 3).to(tl.pointer_type(tl.float32))
        t = tl.load(metadata + step * 3 + 1).to(tl.pointer_type(tl.float32))
        zero = tl.load(s + offsets)
        transition = tl.load(t + offsets)
        state = tl.dot(state, transition, input_precision="ieee") + zero
    tl.store(final + request * size + offsets, state)


@triton.jit
def _scatter(
    packet,
    history,
    final,
    H: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    HISTORY: tl.constexpr,
    TILE: tl.constexpr,
    LAST_ONLY: tl.constexpr,
    RAGGED: tl.constexpr = False,
):
    request = tl.program_id(0)
    step = tl.program_id(1)
    count = N
    if RAGGED:
        count = tl.load(packet + request * 4 + 3).to(tl.int32)
    if LAST_ONLY:
        step += count - 1
    lane = tl.program_id(2) * TILE + tl.arange(0, TILE)
    size: tl.constexpr = H * 128 * 128
    mask = lane < size
    if step == count:
        state = tl.load(history + (request * HISTORY + count % HISTORY) * size + lane, mask, other=0)
        tl.store(final + request * size + lane, state, mask)
    elif step >= 0 and step < count:
        target = tl.load(packet + B * 4 + (request * N + step) * 3 + 2).to(tl.int32)
        if target >= 0:
            replay = tl.load(packet + request * 4 + 2).to(tl.pointer_type(tl.float32))
            state = tl.load(history + (request * HISTORY + step % HISTORY) * size + lane, mask, other=0)
            tl.store(replay + target * size + lane, state, mask)


def _native_pack(zero_states, transitions, zero_pool, transition_pool, steps, count):
    """CANN pool gather, followed by sparse fresh/padding updates; no pool clone."""
    first = zero_states[0]
    _, heads, _, _ = first.shape
    indices, padding = [], []
    fresh = [[] for _ in steps]
    for row, sequence in enumerate(steps):
        pad = count - len(sequence)
        padding.extend(range(row * count, row * count + pad))
        indices.extend([0] * pad)
        for index, step in enumerate(sequence):
            indices.append(-step.source - 1 if step.source < 0 else 0)
            if step.source >= 0:
                fresh[row].append((row * count + pad + index, step.source))
    select = torch.tensor(indices, dtype=torch.int64, device=first.device)
    if len(zero_pool):
        s, t = zero_pool.index_select(0, select), transition_pool.index_select(0, select)
    else:
        s = torch.empty((len(indices), heads, 128, 128), device=first.device, dtype=first.dtype)
        t = torch.empty_like(s)
    for row, entries in enumerate(fresh):
        if len(entries) == 1:
            target, source = entries[0]
            s[target].copy_(zero_states[row][source])
            t[target].copy_(transitions[row][source])
        elif entries:
            targets, sources = zip(*entries)
            target_index = torch.tensor(targets, dtype=torch.int64, device=first.device)
            if list(sources) == list(range(len(zero_states[row]))):
                fresh_s, fresh_t = zero_states[row], transitions[row]
            else:
                source_index = torch.tensor(sources, dtype=torch.int64, device=first.device)
                fresh_s = zero_states[row].index_select(0, source_index)
                fresh_t = transitions[row].index_select(0, source_index)
            s.index_copy_(0, target_index, fresh_s)
            t.index_copy_(0, target_index, fresh_t)
    if padding:
        pad_indices = torch.tensor(padding, dtype=torch.int64, device=first.device)
        s.index_fill_(0, pad_indices, 0)
        identity = torch.eye(128, device=first.device, dtype=first.dtype).expand(len(padding), heads, 128, 128)
        t.index_copy_(0, pad_indices, identity)
    return s, t


def batch_compose(zero_states, transitions, zero_pool, transition_pool, steps, rows=128, pack_mode="torch"):
    first = zero_states[0]
    _, heads, value_dim, key_dim = first.shape
    if (value_dim, key_dim, rows) != (128, 128, 128):
        raise ValueError("the batched Cube experiment requires K=V=rows=128")
    if pack_mode == "cached":
        if all(s.shape[0] == 0 for s in zero_states) and all(
            step.source < 0 and step.replay == -1 for sequence in steps for step in sequence
        ):
            return _cache_only_compose(zero_states, zero_pool, transition_pool, steps)
        # Mixed/fresh/replay plans keep the validated general path. Never treat
        # a miss as a hit or omit requested replay to force the specialization.
        return batch_compose(zero_states, transitions, zero_pool, transition_pool, steps, rows, pack_mode="output")
    batch = len(steps)
    indexed = pack_mode in ("indexed", "output")
    count = max(1, max(map(len, steps)))
    history_count = count + 1 if any(step.replay >= 0 for sequence in steps for step in sequence[:-1]) else 2
    # Only elide zeroing when every output unit is overwritten by a replay.
    # Untargeted units (including block-native fresh Documents) remain zero.
    replay = [
        torch.empty_like(s)
        if indexed and {step.replay for step in sequence if step.replay >= 0} == set(range(len(s)))
        else torch.zeros_like(s)
        for s, sequence in zip(zero_states, steps)
    ]
    final = torch.empty((batch, heads, 128, 128), device=first.device, dtype=first.dtype)
    headers = [
        value
        for s, t, r, seq in zip(zero_states, transitions, replay, steps)
        for value in (s.data_ptr(), t.data_ptr(), r.data_ptr(), len(seq))
    ]
    metadata = []
    state_bytes = heads * 128 * 128 * 4
    pool_s_base, pool_t_base = zero_pool.data_ptr(), transition_pool.data_ptr()
    empty_address = 0
    if indexed and any(not sequence for sequence in steps):
        empty_state = torch.zeros((heads, 128, 128), device=first.device, dtype=first.dtype)
        empty_address = empty_state.data_ptr()
    for row, (zeros, transforms, sequence) in enumerate(zip(zero_states, transitions, steps)):
        if not indexed:
            metadata.extend([0, 0, -1] * (count - len(sequence)))
        fresh_s_base, fresh_t_base = zeros.data_ptr(), transforms.data_ptr()
        if pack_mode == "output":
            replay_base = replay[row].data_ptr()
        for step in sequence:
            if step.source < 0:
                s_base, t_base, index = pool_s_base, pool_t_base, -step.source - 1
            else:
                s_base, t_base, index = fresh_s_base, fresh_t_base, step.source
            target = step.replay
            if pack_mode == "output":
                # A null destination means no replay write, not a valid slot.
                target = replay_base + target * state_bytes if target >= 0 else 0
            metadata.extend((s_base + index * state_bytes, t_base + index * state_bytes, target))
        if indexed:
            unused_target = 0 if pack_mode == "output" else -1
            metadata.extend([empty_address, 0, unused_target] * (count - len(sequence)))
    packet = torch.tensor(headers + metadata, device=first.device, dtype=torch.int64)
    if pack_mode == "output":
        _output_chain[(batch, heads)](packet, final, heads, count, batch, num_warps=4, enable_fp_fusion=False)
        return replay, final
    if pack_mode == "indexed":
        pass
    elif pack_mode == "native":
        packed, packed_t = _native_pack(zero_states, transitions, zero_pool, transition_pool, steps, count)
    elif pack_mode == "torch":
        padding_s = torch.zeros((heads, 128, 128), device=first.device, dtype=first.dtype)
        padding_t = torch.eye(128, device=first.device, dtype=first.dtype).expand(heads, 128, 128)
        source_s, source_t = [], []
        for zeros, transforms, sequence in zip(zero_states, transitions, steps):
            source_s.extend([padding_s] * (count - len(sequence)))
            source_t.extend([padding_t] * (count - len(sequence)))
            for step in sequence:
                source_s.append(zero_pool[-step.source - 1] if step.source < 0 else zeros[step.source])
                source_t.append(transition_pool[-step.source - 1] if step.source < 0 else transforms[step.source])
        packed = torch.stack(source_s + source_t)
    elif pack_mode == "gather":
        packed = torch.empty((2, batch, count, heads, 128, 128), device=first.device, dtype=first.dtype)
    else:
        raise ValueError("unknown Cube pack mode")
    if pack_mode not in ("native", "indexed"):
        packed_t = packed.reshape(2, batch * count, heads, 128, 128)[1]
    history = torch.empty((batch, history_count, heads, 128, 128), device=first.device, dtype=first.dtype)
    tile = 8192
    tiles = triton.cdiv(heads * 128 * 128, tile)
    if pack_mode == "gather":
        _gather[(batch, count, tiles)](
            packet, zero_pool, transition_pool, packed, heads, count, batch, tile, num_warps=4
        )
    if pack_mode == "indexed":
        _indexed_chain[(batch, heads)](
            packet, history, heads, count, batch, history_count, num_warps=4, enable_fp_fusion=False
        )
    else:
        _chain[(batch, heads)](
            packed, packed_t, history, heads, count, batch, history_count, num_warps=4, enable_fp_fusion=False
        )
    last_only = history_count == 2
    _scatter[(batch, 2 if last_only else count + 1, tiles)](
        packet,
        history,
        final,
        heads,
        count,
        batch,
        history_count,
        tile,
        last_only,
        RAGGED=pack_mode == "indexed",
        num_warps=4,
    )
    return replay, final


def window_compose(zero_states, transitions, zero_pool, transition_pool, steps, rows=128, window=16):
    """Batch across requests but bound packing/history by a segment window.

    A synthetic first S carries the preceding window's state. Its T is unused.
    Actual segments retain their original order and replay slot identities.
    """
    first = zero_states[0]
    _, heads, value_dim, key_dim = first.shape
    if (value_dim, key_dim, rows) != (128, 128, 128) or window < 1:
        raise ValueError("window Cube requires K=V=rows=128 and a positive window")
    batch = len(steps)
    length = max(1, max(map(len, steps)))
    replay = [torch.zeros_like(s) for s in zero_states]
    final = torch.zeros((batch, heads, 128, 128), device=first.device, dtype=first.dtype)
    padding_s = torch.zeros((heads, 128, 128), device=first.device, dtype=first.dtype)
    padding_t = torch.eye(128, device=first.device, dtype=first.dtype).expand(heads, 128, 128)
    headers = [
        value
        for s, t, r, seq in zip(zero_states, transitions, replay, steps)
        for value in (s.data_ptr(), t.data_ptr(), r.data_ptr(), len(seq))
    ]
    tiles = triton.cdiv(heads * 128 * 128, 8192)
    for start in range(0, length, window):
        width = min(window, length - start)
        count = width + 1
        sources_s, sources_t, metadata = [], [], []
        full_history = False
        for request, (zeros, transforms, sequence) in enumerate(zip(zero_states, transitions, steps)):
            sources_s.append(final[request])
            sources_t.append(padding_t)
            metadata.extend((0, 0, -1))
            padding = length - len(sequence)
            for position in range(start, start + width):
                if position < padding:
                    sources_s.append(padding_s)
                    sources_t.append(padding_t)
                    target = -1
                else:
                    step = sequence[position - padding]
                    sources_s.append(zero_pool[-step.source - 1] if step.source < 0 else zeros[step.source])
                    sources_t.append(transition_pool[-step.source - 1] if step.source < 0 else transforms[step.source])
                    target = step.replay
                metadata.extend((0, 0, target))
                full_history |= target >= 0 and position < start + width - 1
        history_count = count + 1 if full_history else 2
        packet = torch.tensor(headers + metadata, device=first.device, dtype=torch.int64)
        packed = torch.stack(sources_s + sources_t)
        history = torch.empty((batch, history_count, heads, 128, 128), device=first.device, dtype=first.dtype)
        _chain[(batch, heads)](
            packed,
            packed[batch * count :],
            history,
            heads,
            count,
            batch,
            history_count,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _scatter[(batch, count + 1 if full_history else 2, tiles)](
            packet, history, final, heads, count, batch, history_count, 8192, not full_history, num_warps=4
        )
        del packet, packed, history
    return replay, final
