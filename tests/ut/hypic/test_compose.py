"""Composition metadata/oracle regressions; NPU kernel checks live in the benchmark."""

from dataclasses import replace

import pytest
import torch

from vllm_ascend.hypic.compose import ComposeStep, build_compose_steps, compose_fused_batch, compose_reference
from vllm_ascend.hypic.config import HypicConfig
from vllm_ascend.hypic.planner import build_plan


@pytest.mark.parametrize(
    "mode,seam", [("transition_rope_recompute", 0), ("transition_rope_recompute", 1), ("block_native_pic", 0)]
)
@pytest.mark.parametrize("cache_mode", ["cold", "mixed", "warm"])
def test_steps_preserve_order_and_replay_targets(mode, seam, cache_mode):
    config = HypicConfig(mode=mode, chunk_size=4, seam_sink_tokens=seam)
    cold = build_plan(list(range(9)), {}, config, segment_boundaries=[0, 3, 6, 9])
    slots = {segment["hash"]: i for i, segment in enumerate(cold["segments"][:-1])}
    ready = {
        segment["hash"]: tuple(segment["token_ids"])
        for i, segment in enumerate(cold["segments"][:-1])
        if cache_mode == "warm" or (cache_mode == "mixed" and i == 1)
    }
    plan = build_plan(list(range(9)), ready, config, segment_boundaries=[0, 3, 6, 9])
    steps, units = build_compose_steps(plan, slots.get)
    # Every new S/T unit appears exactly once, ordered as the varlen GDN call.
    assert [step.source for step in steps if step.source >= 0] == list(range(units))
    assert units == plan["num_prefill_units"]
    doc_steps, doc_units = build_compose_steps(plan, slots.get, include_query=False)
    assert doc_steps == steps[:-1]
    assert doc_units == units - 1
    cached = [step.source for step in steps if step.source < 0]
    assert cached == [-slots[s["hash"]] - 1 for s in plan["segments"] if s["hit"]]
    targets = [step.replay for step in steps if step.replay >= 0]
    assert targets == ([units - 1] if mode == "block_native_pic" else list(range(units)))
    if cache_mode == "warm" and seam:
        assert steps == (ComposeStep(-1), ComposeStep(0, 0), ComposeStep(-2), ComposeStep(1, 1))


def test_missing_hit_slot_is_not_silently_dropped():
    config = HypicConfig(chunk_size=2, seam_sink_tokens=0)
    cold = build_plan([1, 2, 3], {}, config)
    ready = {s["hash"]: tuple(s["token_ids"]) for s in cold["segments"][:-1]}
    plan = build_plan([1, 2, 3], ready, config)
    with pytest.raises(RuntimeError, match="slot disappeared"):
        build_compose_steps(plan, lambda _: None)


def test_reference_preserves_noncommutative_order_and_does_not_mutate_pools():
    zeros = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]])
    transforms = torch.tensor([[[[1.0, 2.0], [0.0, 1.0]]], [[[1.0, 0.0], [3.0, 1.0]]]])
    zero_pool = zeros.flip(0).clone()
    transition_pool = transforms.flip(0).clone()
    original = [x.clone() for x in (zeros, transforms, zero_pool, transition_pool)]
    replay, final = compose_reference(
        zeros,
        transforms,
        zero_pool,
        transition_pool,
        [ComposeStep(-1), ComposeStep(0, 0), ComposeStep(-2), ComposeStep(1, 1)],
    )
    h0 = zero_pool[0]
    h1 = h0 @ transforms[0] + zeros[0]
    h2 = h1 @ transition_pool[1] + zero_pool[1]
    torch.testing.assert_close(replay[0], h0)
    torch.testing.assert_close(replay[1], h2)
    torch.testing.assert_close(final, h2 @ transforms[1] + zeros[1])
    for value, saved in zip((zeros, transforms, zero_pool, transition_pool), original):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


def test_backend_config_is_explicit_and_does_not_change_cache_identity():
    base = HypicConfig(chunk_size=4, seam_sink_tokens=0)
    assert base.state_compose_backend == "torch"
    assert base.state_compose_batch_size == 1
    fused = replace(base, state_compose_backend="triton", state_compose_batch_size=4)
    fused.validate()
    assert build_plan(list(range(9)), {}, base) == build_plan(list(range(9)), {}, fused)
    with pytest.raises(ValueError, match="backend"):
        replace(base, state_compose_backend="unknown").validate()
    for size in (True, 0, -1, 1.5, "4"):
        with pytest.raises(ValueError, match="positive integer"):
            replace(base, state_compose_batch_size=size).validate()


def test_fused_backend_does_not_silently_fall_back_on_cpu():
    states = torch.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="NPU"):
        compose_fused_batch([states], [states], states, states, [[ComposeStep(0, 0)]])
    with pytest.raises(ValueError, match="nonempty"):
        compose_fused_batch([], [], states, states, [])


@pytest.mark.parametrize("cached", [False, True])
def test_composition_without_fresh_documents(cached):
    fresh = torch.empty((0, 1, 2, 2))
    pool = torch.ones((1, 1, 2, 2))
    replay, state = compose_reference(fresh, fresh, pool, pool, [ComposeStep(-1)] if cached else [])
    assert replay.shape == fresh.shape
    torch.testing.assert_close(state, pool[0] if cached else torch.zeros_like(pool[0]))
