"""NPU-only regressions for the example Cube prototype, not a default backend."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vllm_ascend.hypic.compose import ComposeStep, compose_reference

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")


@pytest.fixture(scope="module")
def cube_candidate():
    source = Path(__file__).resolve().parents[3] / "examples/offline_inference/hypic_cube_batch.py"
    spec = importlib.util.spec_from_file_location("hypic_cube_npu_regression", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pack_mode", ["torch", "native", "indexed"])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("empty_pool", [False, True])
def test_ragged_empty_permuted_replay(cube_candidate, pack_mode, heads, empty_pool):
    torch.npu.set_device(0)
    torch.manual_seed(20260926)
    device = "npu:0"
    identity = torch.eye(128, device=device)[None, None]
    zeros = [torch.randn((3, heads, 128, 128), device=device) * 0.02 for _ in range(4)]
    transforms = [identity * 0.9 + torch.randn_like(s) * 0.003 for s in zeros]
    pool = torch.randn((0 if empty_pool else 2, heads, 128, 128), device=device) * 0.02
    transitions = identity * 0.9 + torch.randn_like(pool) * 0.003
    a, b = (0, 1) if empty_pool else (-1, -2)
    steps = [
        [],
        [ComposeStep(a), ComposeStep(b), ComposeStep(a)],
        [ComposeStep(0, 2), ComposeStep(1, 0), ComposeStep(2, 1)],
        [ComposeStep(a), ComposeStep(1, 1), ComposeStep(b), ComposeStep(0, 0), ComposeStep(2)],
    ]
    inputs = [*zeros, *transforms, pool, transitions]
    saved = [x.clone() for x in inputs]
    expected = [
        compose_reference(s, t, pool, transitions, sequence) for s, t, sequence in zip(zeros, transforms, steps)
    ]
    for _ in range(3):
        replay, final = cube_candidate.batch_compose(zeros, transforms, pool, transitions, steps, pack_mode=pack_mode)
        for row, (expected_replay, expected_final) in enumerate(expected):
            torch.testing.assert_close(replay[row], expected_replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(final[row], expected_final, atol=2e-5, rtol=2e-4)
    for actual, original in zip(inputs, saved):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 16, 65])
@pytest.mark.parametrize("replay_all", [False, True])
def test_indexed_chain_lengths(cube_candidate, length, replay_all):
    """Gate short Cube recurrence and ring parity, not just identity transforms."""
    torch.npu.set_device(0)
    torch.manual_seed(20260927 + length)
    device = "npu:0"
    heads = 2
    identity = torch.eye(128, device=device)[None, None]
    pool = torch.randn((5, heads, 128, 128), device=device) * 0.02
    pool_t = identity * 0.9 + torch.randn_like(pool) * 0.003
    zeros = [torch.randn((length, heads, 128, 128), device=device) * 0.02 for _ in range(2)]
    transforms = [identity * 0.9 + torch.randn_like(s) * 0.003 for s in zeros]
    steps = []
    for row in range(2):
        sequence = []
        for index in range(length - row):
            source = index if index % 2 == 0 else -(index % 5) - 1
            target = index if replay_all or index == length - row - 1 else -1
            sequence.append(ComposeStep(source, target))
        steps.append(sequence)
    expected = [compose_reference(s, t, pool, pool_t, seq) for s, t, seq in zip(zeros, transforms, steps)]
    saved = [x.clone() for x in [*zeros, *transforms, pool, pool_t]]
    for _ in range(3):
        replay, final = cube_candidate.batch_compose(zeros, transforms, pool, pool_t, steps, pack_mode="indexed")
        for row, (expected_replay, expected_final) in enumerate(expected):
            torch.testing.assert_close(replay[row], expected_replay, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(final[row], expected_final, atol=2e-5, rtol=2e-4)
    for actual, original in zip([*zeros, *transforms, pool, pool_t], saved):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
