from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.hypic.attention import (
    _hydrate_paged_kv_cache,
    forward_hypic_attention,
    reference_suffix_attention,
    rerotate_keys,
)
from vllm_ascend.hypic.cache import DeviceSegmentCache, SegmentCatalog
from vllm_ascend.hypic.config import HypicConfig, get_hypic_config
from vllm_ascend.hypic.gdn import forward_hypic_gdn
from vllm_ascend.hypic.planner import (
    build_plan,
    segment_hash,
    split_segments,
    validate_plan,
)
from vllm_ascend.hypic.runtime import HypicBatchContext
from vllm_ascend.hypic.transition import compose_segments


def test_config_defaults_to_requested_chunk_size() -> None:
    config = get_hypic_config(SimpleNamespace(additional_config={"hypic_config": {"enabled": True}}))
    assert config.enabled
    assert config.chunk_size == 512
    assert config.seam_sink_tokens == 8
    assert config.max_cache_segments == 96


def test_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="transition_rope_recompute"):
        HypicConfig.from_dict({"mode": "addition"})


def test_plan_cold_then_warm_recomputes_middle_seam() -> None:
    config = HypicConfig(enabled=True, chunk_size=16, seam_sink_tokens=3)
    tokens = list(range(42))
    cold = build_plan(tokens, {}, config)
    validate_plan(cold)
    assert cold["query_positions"] == list(range(42))
    assert [segment["cacheable"] for segment in cold["segments"]] == [
        True,
        True,
        False,
    ]

    catalog = SegmentCatalog(8)
    catalog.commit(cold)
    warm = build_plan(tokens, catalog.ready, config)
    validate_plan(warm)
    assert warm["segments"][0]["hit"]
    assert warm["segments"][1]["hit"]
    assert warm["query_positions"] == [16, 17, 18, *range(32, 42)]
    assert warm["num_computed_tokens"] == 29


def test_semantic_boundaries_are_hard_cuts_with_chunk_size_as_maximum() -> None:
    assert split_segments(12, 4, boundaries=[0, 3, 8, 12]) == [
        (0, 3),
        (3, 7),
        (7, 8),
        (8, 12),
    ]


@pytest.mark.parametrize("boundaries", ([0, 5, 3], [0, 3, 3], [-1, 3], [0, 9]))
def test_semantic_boundaries_reject_invalid_offsets(boundaries) -> None:
    with pytest.raises(ValueError, match="segment boundar"):
        split_segments(8, 4, boundaries=boundaries)


def test_reordered_semantic_segments_hit_independently() -> None:
    config = HypicConfig(enabled=True, chunk_size=8, seam_sink_tokens=1)
    first_tokens = [10, 11, 20, 21, 99]
    cold = build_plan(
        first_tokens,
        {},
        config,
        segment_boundaries=[0, 2, 4, 5],
    )
    catalog = SegmentCatalog(8)
    catalog.commit(cold)

    reordered = build_plan(
        [20, 21, 10, 11, 98],
        catalog.ready,
        config,
        segment_boundaries=[0, 2, 4, 5],
    )
    assert [segment["hit"] for segment in reordered["segments"]] == [
        True,
        True,
        False,
    ]


def test_final_segment_never_hits() -> None:
    config = HypicConfig(enabled=True, chunk_size=4, seam_sink_tokens=1)
    tokens = [1, 2, 3, 4]
    digest = segment_hash(tokens)
    plan = build_plan(tokens, {digest: tuple(tokens)}, config)
    assert not plan["segments"][0]["hit"]
    assert plan["query_positions"] == [0, 1, 2, 3]


def test_catalog_collision_guard_and_lru() -> None:
    catalog = SegmentCatalog(1)
    first = build_plan(list(range(8)), {}, HypicConfig(chunk_size=4, seam_sink_tokens=1))
    catalog.commit(first)
    digest = first["segments"][0]["hash"]
    assert catalog.ready[digest] == (0, 1, 2, 3)
    collision = build_plan([9, 9, 9, 9, 5], {digest: (8, 8)}, HypicConfig(chunk_size=4, seam_sink_tokens=1))
    assert not collision["segments"][0]["hit"]


def test_suffix_attention_uses_right_down_causal_offset() -> None:
    query = torch.tensor([[[1.0]], [[1.0]]])
    key = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    value = torch.arange(4, dtype=torch.float32).view(4, 1, 1)
    actual = reference_suffix_attention(query, key, value, scale=1.0)
    first_expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), 0).dot(torch.tensor([0.0, 1.0, 2.0]))
    second_expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0, 3.0]), 0).dot(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(actual[:, 0, 0], torch.stack((first_expected, second_expected)))


def test_rerotate_keys_expands_neox_pair_frequencies() -> None:
    rotary = SimpleNamespace(
        is_neox_style=True,
        rotary_dim=4,
        cos_sin_cache=torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
    )
    key = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    actual = rerotate_keys(
        key,
        torch.tensor([0]),
        torch.tensor([1]),
        rotary,
    )
    torch.testing.assert_close(actual, torch.tensor([[[-3.0, 2.0, 1.0, 4.0]]]))


def test_transition_composition_order() -> None:
    initial = torch.tensor([[[1.0, 2.0]]])
    transitions = [
        torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]),
        torch.tensor([[[1.0, 1.0], [0.0, 1.0]]]),
    ]
    zero_states = [
        torch.tensor([[[4.0, 5.0]]]),
        torch.tensor([[[6.0, 7.0]]]),
    ]
    states = compose_segments(initial, transitions, zero_states)
    torch.testing.assert_close(states[0], torch.tensor([[[6.0, 11.0]]]))
    torch.testing.assert_close(states[1], torch.tensor([[[12.0, 24.0]]]))


def test_device_cache_prepares_stable_slots_and_reuses_lru() -> None:
    cache = DeviceSegmentCache(2)
    cold = {
        "segments": [
            {"hash": "a", "cacheable": True, "hit": False},
            {"hash": "tail", "cacheable": False, "hit": False},
        ]
    }
    cache.prepare([cold])
    slot_a = cache.lookup("a")
    assert slot_a is not None

    cache.prepare(
        [{"segments": [{"hash": "b", "cacheable": True, "hit": False}]}]
    )
    slot_b = cache.lookup("b")
    assert slot_b is not None and slot_b != slot_a
    cache.prepare(
        [{"segments": [{"hash": "c", "cacheable": True, "hit": False}]}]
    )
    assert cache.lookup("a") is None
    assert cache.lookup("c") == slot_a


def test_device_cache_rejects_unprepared_scheduler_hit() -> None:
    cache = DeviceSegmentCache(2)
    with pytest.raises(RuntimeError, match="cache divergence"):
        cache.prepare(
            [{"segments": [{"hash": "missing", "cacheable": True, "hit": True}]}]
        )


def test_device_cache_rejects_lru_order_divergence_before_reserve() -> None:
    cache = DeviceSegmentCache(2)
    cache.prepare(
        [{"segments": [{"hash": "A", "cacheable": True, "hit": False}]}]
    )
    cache.prepare(
        [{"segments": [{"hash": "B", "cacheable": True, "hit": False}]}]
    )

    with pytest.raises(RuntimeError, match="cache order divergence"):
        cache.prepare(
            [
                {
                    "cache_order_before": ("B", "A"),
                    "segments": [
                        {"hash": "C", "cacheable": True, "hit": False}
                    ],
                }
            ]
        )

    # The check happens before reserve can overwrite a resident slot.
    assert tuple(cache.segments) == ("A", "B")


def test_device_cache_rejects_scheduler_eviction_mismatch_before_reuse() -> None:
    cache = DeviceSegmentCache(2)
    cache.prepare(
        [
            {
                "segments": [
                    {"hash": "A", "cacheable": True, "hit": False},
                    {"hash": "B", "cacheable": True, "hit": False},
                ]
            }
        ]
    )

    with pytest.raises(RuntimeError, match="eviction divergence"):
        cache.prepare(
            [
                {
                    "segments": [
                        {
                            "hash": "C",
                            "cacheable": True,
                            "hit": False,
                            "expected_eviction": "B",
                        }
                    ]
                }
            ]
        )

    assert tuple(cache.segments) == ("A", "B")


def test_device_cache_replays_duplicate_misses_in_scheduler_lru_order() -> None:
    """A repeated packed miss must refresh LRU exactly like the scheduler."""
    catalog = SegmentCatalog(3)
    cache = DeviceSegmentCache(3)

    def plan(*digests: str, hit: bool = False) -> dict:
        return {
            "segments": [
                {
                    "hash": digest,
                    "cacheable": True,
                    "hit": hit,
                    "token_ids": [ord(digest)],
                }
                for digest in digests
            ]
        }

    # The second request sees A as cold because scheduler commits happen only
    # after the entire packed forward. Its repeated miss must still move A to
    # MRU when both sides apply the completed plans.
    packed_plans = [plan("A", "B"), plan("A", "C")]
    catalog.prepare_scheduled_plans(packed_plans)
    assert packed_plans[-1]["cache_order_after_commit"] == ("B", "A", "C")
    cache.prepare(packed_plans)
    for packed_plan in packed_plans:
        catalog.commit(packed_plan)

    assert tuple(cache.segments) == tuple(catalog.ready) == ("B", "A", "C")

    next_miss = plan("D")
    catalog.prepare_scheduled_plans([next_miss])
    assert next_miss["cache_order_after_commit"] == ("A", "C", "D")
    assert next_miss["segments"][0]["expected_eviction"] == "B"
    cache.prepare([next_miss])
    scheduler_evictions = catalog.commit(next_miss)

    assert scheduler_evictions == ["B"]
    assert tuple(cache.segments) == tuple(catalog.ready) == ("A", "C", "D")
    cache.prepare([plan("A", hit=True)])


def test_hypic_attention_keeps_batched_request_prefixes_independent() -> None:
    config = HypicConfig(enabled=True, chunk_size=2, seam_sink_tokens=1)
    plans = {
        "first": build_plan([1, 2, 3], {}, config),
        "second": build_plan([4, 5, 6], {}, config),
    }
    context = HypicBatchContext(
        plans=plans,
        request_ids=("first", "second"),
        cache=DeviceSegmentCache(8),
    )
    rotary = SimpleNamespace(
        is_neox_style=True,
        rotary_dim=2,
        cos_sin_cache=torch.tensor([[1.0, 0.0]] * 8),
    )
    layer = SimpleNamespace(
        layer_name="attn",
        hypic_rotary_emb=rotary,
        hypic_key_pool=torch.empty((8, 2, 1, 2)),
        hypic_value_pool=torch.empty((8, 2, 1, 2)),
    )
    query = torch.tensor(
        [[[1.0, 0.0]], [[0.5, 1.0]], [[1.0, 1.0]],
         [[-1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 1.0]]]
    )
    key = query.clone()
    value = torch.arange(12, dtype=torch.float32).view(6, 1, 2)
    output = torch.empty((6, 2))
    context.cache.prepare(plans.values())

    forward_hypic_attention(
        layer,
        query,
        key,
        value,
        output,
        context,
        scale=1.0,
        kv_cache=(torch.empty(0), torch.empty(0)),
        attn_metadata=SimpleNamespace(block_tables=torch.empty((2, 0))),
    )

    expected = torch.cat(
        [
            reference_suffix_attention(
                query[start : start + 3],
                key[start : start + 3],
                value[start : start + 3],
                scale=1.0,
            ).reshape(3, 2)
            for start in (0, 3)
        ]
    )
    torch.testing.assert_close(output, expected)


def test_hydrate_paged_cache_selects_request_block_table(monkeypatch) -> None:
    captured = {}

    def fake_reshape_and_cache(**kwargs) -> None:
        captured["slots"] = kwargs["slot_mapping"].clone()

    monkeypatch.setattr(
        "vllm_ascend.hypic.attention.DeviceOperator.reshape_and_cache",
        fake_reshape_and_cache,
    )
    key = torch.zeros((2, 1, 1))
    value = torch.zeros_like(key)
    key_cache = torch.zeros((10, 4, 1, 1))
    value_cache = torch.zeros_like(key_cache)
    metadata = SimpleNamespace(block_tables=torch.tensor([[5], [9]]))

    _hydrate_paged_kv_cache(
        key,
        value,
        start=0,
        request_index=1,
        kv_cache=(key_cache, value_cache),
        attn_metadata=metadata,
    )

    torch.testing.assert_close(captured["slots"], torch.tensor([36, 37], dtype=torch.int32))


def test_hypic_gdn_dispatches_each_packed_request(monkeypatch) -> None:
    plans = {
        "first": {"query_positions": [0, 1]},
        "second": {"query_positions": [0, 1, 2]},
    }
    context = HypicBatchContext(
        plans=plans,
        request_ids=("first", "second"),
        cache=DeviceSegmentCache(8),
    )
    calls = []

    def fake_forward(
        layer,
        mixed_qkv,
        b,
        a,
        core_attn_out,
        context,
        attn_metadata,
        request_id,
        request_index,
    ) -> None:
        calls.append((request_id, request_index, len(mixed_qkv)))

    monkeypatch.setattr(
        "vllm_ascend.hypic.gdn._forward_hypic_gdn_request",
        fake_forward,
    )
    tensors = [torch.zeros((5, 1)) for _ in range(4)]
    forward_hypic_gdn(
        object(),
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        context,
        SimpleNamespace(num_actual_tokens=5),
    )

    assert calls == [("first", 0, 2), ("second", 1, 3)]
