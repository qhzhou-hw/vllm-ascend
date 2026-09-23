"""Pure-Python HYPIC prompt segmentation and sparse recompute planning."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from typing import Any

from vllm_ascend.hypic.config import HypicConfig
from vllm_ascend.hypic.protocol import PIC_PROTOCOL_VERSION


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
    *,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    """Build a process-safe sparse prefill plan for one request.

    Query always executes. Legacy HYPIC recomputes hit seams; block-native PIC
    uses explicit training boundaries and skips every hit document completely.
    """
    tokens = [int(token) for token in token_ids]
    block_native_pic = config.mode == "block_native_pic"
    if block_native_pic:
        config.validate()
        if segment_boundaries is None:
            raise ValueError("block_native_pic requires explicit hypic_segment_boundaries matching training")
        boundaries = list(segment_boundaries)
        if any(isinstance(position, bool) or not isinstance(position, int) for position in boundaries):
            raise ValueError("block_native_pic boundaries must be integer token offsets")
        if len(boundaries) < 3 or boundaries[0] != 0 or boundaries[-1] != len(tokens):
            raise ValueError("block_native_pic boundaries must include 0, document/query boundary, and prompt length")
        # A storage limit must never introduce extra reset/attention boundaries.
        # Query has no persistent segment slot and may exceed chunk_size.
        ranges = split_segments(len(tokens), max(len(tokens), 1), boundaries=boundaries)
        if any(end - start > config.chunk_size for start, end in ranges[:-1]):
            raise ValueError(
                "block_native_pic document exceeds chunk_size; increase the pool slot size, do not split it"
            )
    else:
        ranges = split_segments(len(tokens), config.chunk_size, boundaries=segment_boundaries)
    reset_conv_history = config.reset_conv_history or block_native_pic
    query_positions: list[int] = []
    segments: list[dict[str, Any]] = []

    for index, (start, end) in enumerate(ranges):
        segment_tokens = tuple(tokens[start:end])
        is_last = index == len(ranges) - 1
        coverage_start = min(config.seam_sink_tokens, end - start) if index > 0 and not is_last else 0
        content_hash = segment_hash(segment_tokens)
        # Each engine has private pools (model/dtype/TP/RoPE lifetime). Within
        # that namespace, isolate caller salt and training-aligned semantics.
        identity = json.dumps(
            [
                PIC_PROTOCOL_VERSION,
                config.mode,
                reset_conv_history,
                cache_salt,
                content_hash,
                len(segment_tokens),
                coverage_start,
            ],
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        cacheable = not is_last and coverage_start < end - start
        # Full equality protects the deliberately truncated hash collision path.
        hit = cacheable and ready_segments.get(digest) == segment_tokens
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
                "content_hash": content_hash,
                "hit": hit,
                "cacheable": cacheable,
                "seam": seam,
                "recompute_seam": coverage_start if cacheable else 0,
                "token_ids": list(segment_tokens),
            }
        )

    return {
        "version": PIC_PROTOCOL_VERSION,
        "mode": config.mode,
        "reset_conv_history": reset_conv_history,
        "num_tokens": len(tokens),
        "logical_advance": len(tokens),
        "query_positions": query_positions,
        "num_query_tokens": len(query_positions),
        "num_reused_tokens": len(tokens) - len(query_positions),
        "num_restore_tokens": sum(s["end"] - s["start"] for s in segments if s["hit"]),
        "num_prefill_units": count_prefill_units(segments),
        # Kept for readers of old diagnostic plans. Never passed to the
        # scheduler as a contiguous computed prefix by the PIC adapter.
        "num_computed_tokens": len(tokens) - len(query_positions),
        "segments": segments,
    }


def count_prefill_units(segments: Sequence[dict[str, Any]]) -> int:
    """Number of S/T workspaces used by the existing packed GDN algorithm."""
    return sum(
        int(bool(segment["seam"])) if segment["hit"] else 1 + int(bool(segment["recompute_seam"]))
        for segment in segments
    )


def refresh_plan_queries(plan: dict[str, Any]) -> None:
    """Recompute accounting after resource planning changes a segment action."""
    positions = []
    restore_tokens = 0
    for segment in plan["segments"]:
        start, end = segment["start"], segment["end"]
        if segment["hit"]:
            positions.extend(range(start, start + segment["seam"]))
            restore_tokens += end - start
        else:
            positions.extend(range(start, end))
    plan["query_positions"] = positions
    plan["num_query_tokens"] = len(positions)
    plan["num_reused_tokens"] = plan["num_tokens"] - len(positions)
    plan["num_computed_tokens"] = plan["num_reused_tokens"]
    plan["num_restore_tokens"] = restore_tokens
    plan["num_prefill_units"] = count_prefill_units(plan["segments"])


def validate_plan(plan: dict[str, Any]) -> None:
    """Validate plan invariants before a worker trusts scheduler metadata."""
    if plan.get("version") != PIC_PROTOCOL_VERSION:
        raise ValueError(f"Unsupported HYPIC plan version: {plan.get('version')}")
    if plan.get("mode") not in {"transition_rope_recompute", "block_native_pic"}:
        raise ValueError("HYPIC plan mode is missing or unsupported")
    if not isinstance(plan.get("reset_conv_history"), bool):
        raise ValueError("HYPIC plan reset_conv_history must be a boolean")
    if plan["mode"] == "block_native_pic":
        if not plan["reset_conv_history"] or len(plan["segments"]) < 2:
            raise ValueError("block_native_pic plan requires convolution reset and documents plus query")
        if any(segment["seam"] or segment["recompute_seam"] for segment in plan["segments"]):
            raise ValueError("block_native_pic plan cannot recompute seams")
    num_tokens = int(plan["num_tokens"])
    positions = [int(pos) for pos in plan["query_positions"]]
    if positions != sorted(set(positions)):
        raise ValueError("HYPIC query positions must be sorted and unique")
    if positions and (positions[0] < 0 or positions[-1] >= num_tokens):
        raise ValueError("HYPIC query position is outside the prompt")
    if int(plan["num_computed_tokens"]) != num_tokens - len(positions):
        raise ValueError("HYPIC computed-token count does not match sparse queries")
    cursor = 0
    expected = []
    for segment in plan["segments"]:
        start, end = segment["start"], segment["end"]
        if start != cursor or end <= start or len(segment["token_ids"]) != end - start:
            raise ValueError("PIC segments must partition the prompt")
        if segment["hit"] and not segment["cacheable"]:
            raise ValueError("PIC cannot hit an uncacheable segment")
        seam = segment["seam"] if segment["hit"] else end - start
        if not 0 <= seam <= end - start:
            raise ValueError("PIC seam outside segment")
        expected.extend(range(start, start + seam))
        cursor = end
    if cursor != num_tokens or positions != expected:
        raise ValueError("PIC sparse queries disagree with segments")
    if not positions or positions[-1] != num_tokens - 1 or plan["segments"][-1]["cacheable"]:
        raise ValueError("PIC final segment must execute for logits")
    if plan["num_query_tokens"] != len(positions) or plan["logical_advance"] != num_tokens:
        raise ValueError("PIC logical/query accounting mismatch")
