"""Pure-Python HYPIC prompt segmentation and sparse recompute planning."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from typing import Any

from vllm_ascend.hypic.config import HypicConfig


def segment_hash(token_ids: Sequence[int]) -> str:
    """Return the portable 128-bit HYPIC hash for a token segment."""
    payload = b"".join(struct.pack("<i", int(token)) for token in token_ids)
    return hashlib.sha256(payload).digest()[:16].hex()


def split_segments(
    num_tokens: int,
    chunk_size: int,
    boundaries: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Split tokens without crossing optional semantic boundaries.

    ``chunk_size`` remains the maximum device-slot length. Explicit boundaries
    are hard cuts, so a tool schema can be cached independently even when tools
    are reordered between requests. Regions longer than ``chunk_size`` are
    split further without being merged with a neighboring semantic region.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens cannot be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if boundaries is None:
        anchors = [0, num_tokens]
    else:
        anchors = [int(position) for position in boundaries]
        if anchors != sorted(set(anchors)):
            raise ValueError("HYPIC segment boundaries must be sorted and unique")
        if anchors and (anchors[0] < 0 or anchors[-1] > num_tokens):
            raise ValueError("HYPIC segment boundary is outside the prompt")
        if not anchors or anchors[0] != 0:
            anchors.insert(0, 0)
        if anchors[-1] != num_tokens:
            anchors.append(num_tokens)

    ranges: list[tuple[int, int]] = []
    for region_start, region_end in zip(anchors, anchors[1:]):
        for start in range(region_start, region_end, chunk_size):
            ranges.append((start, min(start + chunk_size, region_end)))
    return ranges


def build_plan(
    token_ids: Sequence[int],
    ready_segments: dict[str, tuple[int, ...]],
    config: HypicConfig,
    segment_boundaries: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a process-safe sparse prefill plan for one request.

    The final segment always executes to produce logits. A cache hit on the
    first segment executes no queries; later hits execute the seam sink at the
    segment start so boundary behavior matches the SGLang implementation.
    """
    tokens = [int(token) for token in token_ids]
    ranges = split_segments(
        len(tokens), config.chunk_size, boundaries=segment_boundaries
    )
    query_positions: list[int] = []
    segments: list[dict[str, Any]] = []

    for index, (start, end) in enumerate(ranges):
        segment_tokens = tuple(tokens[start:end])
        digest = segment_hash(segment_tokens)
        is_last = index == len(ranges) - 1
        # Full equality protects the deliberately truncated hash collision path.
        hit = not is_last and ready_segments.get(digest) == segment_tokens
        seam = 0
        if hit and index > 0:
            seam = min(config.seam_sink_tokens, end - start)
            query_positions.extend(range(start, start + seam))
        elif not hit:
            query_positions.extend(range(start, end))
        segments.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "hash": digest,
                "hit": hit,
                "cacheable": not is_last,
                "seam": seam,
                "recompute_seam": (min(config.seam_sink_tokens, end - start) if not is_last and index > 0 else 0),
                "token_ids": list(segment_tokens),
            }
        )

    return {
        "version": 1,
        "num_tokens": len(tokens),
        "query_positions": query_positions,
        "num_computed_tokens": len(tokens) - len(query_positions),
        "segments": segments,
    }


def validate_plan(plan: dict[str, Any]) -> None:
    """Validate plan invariants before a worker trusts scheduler metadata."""
    if plan.get("version") != 1:
        raise ValueError(f"Unsupported HYPIC plan version: {plan.get('version')}")
    num_tokens = int(plan["num_tokens"])
    positions = [int(pos) for pos in plan["query_positions"]]
    if positions != sorted(set(positions)):
        raise ValueError("HYPIC query positions must be sorted and unique")
    if positions and (positions[0] < 0 or positions[-1] >= num_tokens):
        raise ValueError("HYPIC query position is outside the prompt")
    if int(plan["num_computed_tokens"]) != num_tokens - len(positions):
        raise ValueError("HYPIC computed-token count does not match sparse queries")
