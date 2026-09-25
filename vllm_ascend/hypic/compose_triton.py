"""Experimental persistent ordered composition on Ascend, with FP32 operands."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_VALUE_TILE = 8
_REDUCTION_TILE = 16


@triton.jit
def _compose_kernel(
    pointers,
    metadata,
    lengths,
    zero_pool,
    transition_pool,
    final_states,
    HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    KEY_DIM: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Requests, heads and value-row tiles are independent; segment order is
    # deliberately sequential. Every program retains its entire key axis.
    request = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    keys = tl.arange(0, BLOCK_K)
    state_offsets = head * VALUE_DIM * KEY_DIM + rows[:, None] * KEY_DIM + keys[None, :]
    state_mask = (rows[:, None] < VALUE_DIM) & (keys[None, :] < KEY_DIM)
    reduction_lane = tl.arange(0, BLOCK_R)
    fresh_s = tl.load(pointers + request * 3).to(tl.pointer_type(tl.float32))
    fresh_t = tl.load(pointers + request * 3 + 1).to(tl.pointer_type(tl.float32))
    replay = tl.load(pointers + request * 3 + 2).to(tl.pointer_type(tl.float32))
    count = tl.load(lengths + request).to(tl.int32)
    state = tl.full((BLOCK_V, BLOCK_K), 0, tl.float32)
    for step in range(count):
        offset = (request * MAX_STEPS + step) * 2
        source = tl.load(metadata + offset).to(tl.int32)
        target = tl.load(metadata + offset + 1).to(tl.int32)
        if target >= 0:
            tl.store(replay + target * HEADS * VALUE_DIM * KEY_DIM + state_offsets, state, state_mask)
        if source < 0:
            slot = -source - 1
            zero = tl.load(zero_pool + slot * HEADS * VALUE_DIM * KEY_DIM + state_offsets, state_mask, other=0)
        else:
            zero = tl.load(fresh_s + source * HEADS * VALUE_DIM * KEY_DIM + state_offsets, state_mask, other=0)
        if step == 0:
            # H0 is zero: skip a matrix multiply, without aliasing cache data.
            state = zero
        else:
            # Ascend cannot merge pointers with different allocation origins.
            # Load in each branch and merge values instead of pointer SSA nodes.
            # Vector FP32 reduction avoids the inaccurate FP32 Cube dot path
            # observed on 910B with Triton-Ascend 3.2.0. Tile the reduction to
            # bound compiler-generated UB scratch. No BF16/TF32 cast.
            # Reuse each contiguous T tile across BLOCK_V rows of H.
            product = tl.full((BLOCK_V, BLOCK_K), 0, tl.float32)
            for start in range(0, KEY_DIM, BLOCK_R):
                inner = start + reduction_lane
                offsets = head * KEY_DIM * KEY_DIM + inner[:, None] * KEY_DIM + keys[None, :]
                mask = (keys[None, :] < KEY_DIM) & (inner[:, None] < KEY_DIM)
                if source < 0:
                    transition = tl.load(
                        transition_pool + (-source - 1) * HEADS * KEY_DIM * KEY_DIM + offsets, mask, other=0
                    )
                else:
                    transition = tl.load(fresh_t + source * HEADS * KEY_DIM * KEY_DIM + offsets, mask, other=0)
                indices = tl.broadcast_to((inner % BLOCK_K)[None, :], (BLOCK_V, BLOCK_R))
                partial_state = tl.gather(state, indices, axis=1)
                product += tl.sum(partial_state[:, :, None] * transition[None, :, :], axis=1)
            state = product + zero
    tl.store(final_states + request * HEADS * VALUE_DIM * KEY_DIM + state_offsets, state, state_mask)


def launch_compose(zero_states, transitions, zero_pool, transition_pool, steps):
    first = zero_states[0]
    device = first.device
    _, heads, value_dim, key_dim = first.shape
    # Every targeted replay row is fully written by the kernel. Only document
    # rows deliberately not targeted in block-native mode need zero fill.
    replay = [
        torch.empty_like(zeros) if sum(step.replay >= 0 for step in sequence) == len(zeros) else torch.zeros_like(zeros)
        for zeros, sequence in zip(zero_states, steps)
    ]
    final = torch.empty((len(steps), heads, value_dim, key_dim), device=device, dtype=torch.float32)
    max_steps = max(1, max(map(len, steps)))
    # Only small integer metadata is uploaded. All large S/T tensors stay in
    # their existing pools; retain their Python owners until kernel enqueue.
    metadata = [
        [(step.source, step.replay) for step in sequence] + [(0, -1)] * (max_steps - len(sequence))
        for sequence in steps
    ]
    # A single small H2D transfer replaces three separately synchronized ones.
    batch = len(steps)
    pointer_values = [
        pointer
        for s, t, r in zip(zero_states, transitions, replay)
        for pointer in (s.data_ptr(), t.data_ptr(), r.data_ptr())
    ]
    packet = torch.tensor(
        pointer_values
        + [len(sequence) for sequence in steps]
        + [value for sequence in metadata for entry in sequence for value in entry],
        device=device,
        dtype=torch.int64,
    )
    pointers = packet[: 3 * batch]
    lengths = packet[3 * batch : 4 * batch]
    metadata_tensor = packet[4 * batch :]
    _compose_kernel[(len(steps), heads, triton.cdiv(value_dim, _VALUE_TILE))](
        pointers,
        metadata_tensor,
        lengths,
        zero_pool,
        transition_pool,
        final,
        HEADS=heads,
        VALUE_DIM=value_dim,
        KEY_DIM=key_dim,
        MAX_STEPS=max_steps,
        BLOCK_K=max(16, triton.next_power_of_2(key_dim)),
        BLOCK_V=_VALUE_TILE,
        BLOCK_R=_REDUCTION_TILE,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return replay, final
