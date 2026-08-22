"""HYPIC segment reuse support for hybrid Qwen3.5 models."""

from vllm_ascend.hypic.config import HypicConfig, get_hypic_config
from vllm_ascend.hypic.planner import build_plan, segment_hash

__all__ = ["HypicConfig", "build_plan", "get_hypic_config", "segment_hash"]
