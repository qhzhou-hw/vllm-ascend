"""GDN affine-state composition used by transition-mode HYPIC."""

from __future__ import annotations

import torch


def compose_state(
    previous: torch.Tensor,
    transition: torch.Tensor,
    zero_state: torch.Tensor,
) -> torch.Tensor:
    """Compose ``H_next = H_previous @ T + S_zero`` in logical layout."""
    if previous.shape[:-2] != transition.shape[:-2]:
        raise ValueError("state and transition batch/head dimensions must match")
    if transition.shape[-2] != transition.shape[-1]:
        raise ValueError("HYPIC transition matrices must be square")
    if previous.shape[-1] != transition.shape[-2]:
        raise ValueError("state K dimension does not match transition")
    if previous.shape != zero_state.shape:
        raise ValueError("previous and zero-state shapes must match")
    return torch.matmul(previous.float(), transition.float()).add_(zero_state.float())


def compose_segments(
    initial_state: torch.Tensor,
    transitions: list[torch.Tensor],
    zero_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Return the accumulated state after every segment in prompt order."""
    if len(transitions) != len(zero_states):
        raise ValueError("transition and zero-state counts must match")
    states: list[torch.Tensor] = []
    accumulated = initial_state
    for transition, zero_state in zip(transitions, zero_states):
        accumulated = compose_state(accumulated, transition, zero_state)
        states.append(accumulated)
    return states
