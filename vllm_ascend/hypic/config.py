"""Configuration and validation for the HYPIC execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HypicConfig:
    """HYPIC settings stored under ``additional_config.hypic_config``."""

    enabled: bool = False
    chunk_size: int = 512
    seam_sink_tokens: int = 8
    max_cache_segments: int = 128
    mode: str = "transition_rope_recompute"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> HypicConfig:
        """Build and validate a HYPIC configuration."""
        raw = raw or {}
        unknown = set(raw).difference(cls.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown hypic_config option(s): {names}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject settings outside the implemented algorithm contract."""
        if self.mode != "transition_rope_recompute":
            raise ValueError("vllm-ascend HYPIC currently supports only mode='transition_rope_recompute'")
        if self.chunk_size <= 0:
            raise ValueError("hypic_config.chunk_size must be positive")
        if self.seam_sink_tokens < 0:
            raise ValueError("hypic_config.seam_sink_tokens cannot be negative")
        if self.seam_sink_tokens >= self.chunk_size:
            raise ValueError("hypic_config.seam_sink_tokens must be smaller than chunk_size")
        if self.max_cache_segments <= 0:
            raise ValueError("hypic_config.max_cache_segments must be positive")


def get_hypic_config(vllm_config: Any) -> HypicConfig:
    """Read HYPIC settings from a vLLM config-like object."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    raw = additional.get("hypic_config")
    if raw is not None and not isinstance(raw, dict):
        raise TypeError("additional_config.hypic_config must be a dictionary")
    return HypicConfig.from_dict(raw)
