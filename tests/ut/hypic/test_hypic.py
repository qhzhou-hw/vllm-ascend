from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.hypic.attention import reference_suffix_attention, rerotate_keys
from vllm_ascend.hypic.cache import SegmentCatalog
from vllm_ascend.hypic.config import HypicConfig, get_hypic_config
from vllm_ascend.hypic.planner import build_plan, segment_hash, validate_plan
from vllm_ascend.hypic.transition import compose_segments


def test_config_defaults_to_requested_chunk_size() -> None:
    config = get_hypic_config(SimpleNamespace(additional_config={"hypic_config": {"enabled": True}}))
    assert config.enabled
    assert config.chunk_size == 512
    assert config.seam_sink_tokens == 8


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
