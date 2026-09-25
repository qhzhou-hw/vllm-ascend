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
    # Segment tensors use fixed model-owned pools, so vLLM accounts for their
    # memory before sizing the ordinary hybrid KV cache.
    max_cache_segments: int = 96
    # Bounds the FP32 S/T workspaces independently of token/slot budgets.
    # Semantic boundaries may create many short units even in a short prompt.
    max_prefill_units: int = 256
    mode: str = "transition_rope_recompute"
    # Only affects legacy HYPIC. block_native_pic always resets convolution.
    reset_conv_history: bool = False
    # Experimental execution backend, not a change to cached S/T semantics.
    state_compose_backend: str = "torch"
    # Bound live request workspaces when opting into the fused backend.
    state_compose_batch_size: int = 1

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
        if self.state_compose_backend not in {"torch", "triton"}:
            raise ValueError("hypic_config.state_compose_backend must be 'torch' or 'triton'")
        if (
            isinstance(self.state_compose_batch_size, bool)
            or not isinstance(self.state_compose_batch_size, int)
            or self.state_compose_batch_size <= 0
        ):
            raise ValueError("hypic_config.state_compose_batch_size must be a positive integer")
        if not isinstance(self.reset_conv_history, bool):
            raise ValueError("hypic_config.reset_conv_history must be a boolean")
        if self.mode not in {"transition_rope_recompute", "block_native_pic"}:
            raise ValueError("HYPIC supports only mode='transition_rope_recompute' or 'block_native_pic'")
        if self.mode == "block_native_pic" and self.seam_sink_tokens != 0:
            raise ValueError("hypic_config mode='block_native_pic' requires seam_sink_tokens=0")
        if self.chunk_size <= 0:
            raise ValueError("hypic_config.chunk_size must be positive")
        if self.seam_sink_tokens < 0:
            raise ValueError("hypic_config.seam_sink_tokens cannot be negative")
        if self.seam_sink_tokens >= self.chunk_size:
            raise ValueError("hypic_config.seam_sink_tokens must be smaller than chunk_size")
        if self.max_cache_segments <= 0:
            raise ValueError("hypic_config.max_cache_segments must be positive")
        if self.max_prefill_units <= 0:
            raise ValueError("hypic_config.max_prefill_units must be positive")


def get_hypic_config(vllm_config: Any) -> HypicConfig:
    """Read HYPIC settings from a vLLM config-like object."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    raw = additional.get("hypic_config")
    if raw is not None and not isinstance(raw, dict):
        raise TypeError("additional_config.hypic_config must be a dictionary")
    return HypicConfig.from_dict(raw)
