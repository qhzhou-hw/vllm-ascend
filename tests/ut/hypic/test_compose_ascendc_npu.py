# SPDX-License-Identifier: Apache-2.0
"""Run on NPU after explicitly loading the standalone Ascend C extension."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")


@pytest.fixture(scope="module")
def impl():
    if not hasattr(torch.ops.hypic_ascendc, "cached_compose"):
        pytest.fail("load the Ascend C library first; use hypic_compose_ascendc_bench.py --tests")
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/hypic/compose_ascendc.py"
    spec = importlib.util.spec_from_file_location("test_cached_ascendc", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    return module


@pytest.fixture(params=[0, 32, 64, 128])
def tile_rows(request):
    return request.param


def pools(slots=5, heads=2):
    torch.manual_seed(20261103)
    s = torch.randn((slots, heads, 128, 128), device="npu:0") * 0.02
    t = torch.eye(128, device=s.device)[None, None] * 0.9 + torch.randn_like(s) * 0.003
    return s, t


def oracle(s, t, sequences):
    expected = torch.zeros((len(sequences), *s.shape[1:]), device=s.device)
    for row, seq in enumerate(sequences):
        for slot in seq:
            expected[row] = torch.bmm(expected[row], t[slot]) + s[slot]
    return expected


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 16, 65])
@pytest.mark.parametrize("batch,heads", [(1, 1), (4, 2), (17, 4)])
def test_ragged_reordered_repeated(impl, count, batch, heads, tile_rows):
    s, t = pools(heads=heads)
    seqs = [[(i * 3 + row) % 5 for i in range(max(0, count - row))] for row in range(batch)]
    expected = oracle(s, t, seqs)
    saved_s, saved_t = s.clone(), t.clone()
    plan = impl.CachedComposeAscendC(s, t, seqs, tile_rows=tile_rows)
    old = plan(s, t)
    for _ in range(3):
        actual = plan(s, t)
        assert old.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(old, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(s, saved_s, atol=0, rtol=0)
    torch.testing.assert_close(t, saved_t, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["identity", "zero", "noncommuting"])
def test_structured_transitions(impl, kind, tile_rows):
    s, t = pools(slots=3, heads=1)
    if kind == "identity":
        t.copy_(torch.eye(128, device=s.device).expand_as(t))
    elif kind == "zero":
        t.zero_()
    else:
        # Cyclic permutation and nonuniform diagonal: exposes reversed/transposed products.
        t[0, 0] = torch.eye(128, device=s.device).roll(1, 1)
        t[1, 0] = torch.diag(torch.linspace(0.5, 1.0, 128, device=s.device))
    seqs = [[2, 0, 1, 0], [2, 1, 0, 1], [], [0]]
    actual = impl.compose_cached_ascendc(s, t, seqs, tile_rows=tile_rows)
    torch.testing.assert_close(actual, oracle(s, t, seqs), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("batch", [0, 1, 4])
def test_empty_pool(impl, batch, tile_rows):
    s, t = pools(slots=0)
    actual = impl.compose_cached_ascendc(s, t, [[] for _ in range(batch)], tile_rows=tile_rows)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("slot", [-1, 5, 2**32, 1.5, True])
def test_bad_slot(impl, slot):
    s, t = pools()
    with pytest.raises(ValueError, match="slot"):
        impl.CachedComposeAscendC(s, t, [[slot]])


def test_pool_contract_and_plan_ownership(impl, tile_rows):
    s, t = pools()
    for invalid in (s.half(), s.transpose(-1, -2), s[..., :64, :64].contiguous()):
        with pytest.raises(ValueError):
            impl.CachedComposeAscendC(invalid, t, [[0]])
    seqs = [[0, 1], []]
    plan = impl.CachedComposeAscendC(s, t, seqs, tile_rows=tile_rows)
    seqs[0][:] = [3, 4]
    # Reusing metadata must still read CURRENT pool data, not cache a prior result.
    s.add_(0.01)
    torch.testing.assert_close(plan(s, t), oracle(s, t, [[0, 1], []]), atol=2e-5, rtol=2e-4)
    with pytest.raises(ValueError, match="shape"):
        plan(s[:2], t[:2])
    with torch.npu.stream(torch.npu.Stream()), pytest.raises(ValueError, match="stream"):
        plan(s, t)


@pytest.mark.parametrize("packet", [[[2, 0]], [[1, -1]], [[1, 5]], [[-1, 0]]])
def test_low_level_bad_metadata_does_not_read_pool(impl, packet, tile_rows):
    s, t = pools()
    plan = impl.CachedComposeAscendC(s, t, [[0]], tile_rows=tile_rows)
    invalid = torch.tensor(packet, device=s.device, dtype=torch.int32)
    actual = torch.ops.hypic_ascendc.cached_compose(
        s, t, invalid, plan._tiling, plan._workspace_bytes, plan._max_blocks, plan._tile_rows
    )
    assert torch.isnan(actual).all().item()


@pytest.mark.parametrize("rows", [-1, 16, 256, True, "64", 64.0])
def test_bad_row_tile(impl, rows):
    s, t = pools()
    with pytest.raises(ValueError, match="tile_rows"):
        impl.CachedComposeAscendC(s, t, [[0]], tile_rows=rows)


def test_repeated_launch_memory_is_bounded(impl, tile_rows):
    s, t = pools(heads=1)
    seqs = [[0, 3, 1, 4], [2, 2], []]
    expected = oracle(s, t, seqs)
    plan = impl.CachedComposeAscendC(s, t, seqs, tile_rows=tile_rows)
    for _ in range(4):
        plan(s, t)
    torch.npu.synchronize()
    initial = torch.npu.memory_allocated()
    # More than 4 GiB would accumulate here if each callback retained workspace.
    for _ in range(8):
        for _ in range(32):
            actual = plan(s, t)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        del actual
    torch.npu.synchronize()
    assert torch.npu.memory_allocated() - initial < 2 * 1024 * 1024


def test_ephemeral_plans_keep_queued_metadata_valid(impl, tile_rows):
    s, t = pools(heads=1)
    sequences = [[[0, 1, 4], []], [[3, 0], [1]], [[2], [4, 3, 1, 0]]]
    expected = [oracle(s, t, seqs) for seqs in sequences]
    # Destroy every plan immediately, with several differently shaped metadata
    # packets and workspaces queued before synchronizing or checking results.
    outputs = [impl.compose_cached_ascendc(s, t, sequences[i % 3], tile_rows=tile_rows) for i in range(48)]
    torch.npu.synchronize()
    for i, actual in enumerate(outputs):
        torch.testing.assert_close(actual, expected[i % 3], atol=2e-5, rtol=2e-4)


def test_pool_allocation_stream_lifetime(impl, tile_rows):
    producer = torch.npu.Stream()
    with torch.npu.stream(producer):
        s, t = pools(heads=1)
    torch.npu.current_stream().wait_stream(producer)
    seqs = [[4, 1, 0, 3] * 16]
    expected = oracle(s, t, seqs)
    plan = impl.CachedComposeAscendC(s, t, seqs, tile_rows=tile_rows)
    actual = plan(s, t)
    del s, t
    # Reusing the producer's allocator must not overwrite a still-consumed pool.
    with torch.npu.stream(producer):
        for _ in range(16):
            torch.empty((5, 1, 128, 128), device="npu:0").fill_(123)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 16, 65, 129])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_tree_ragged_odd_and_cache_ownership(impl, tile_rows, count, depth):
    s, t = pools(slots=7, heads=2)
    seqs = [[(i * 3 + row) % 7 for i in range(max(0, count - row))] for row in range(4)]
    expected = oracle(s, t, seqs)
    saved_s, saved_t = s.clone(), t.clone()
    plan = impl.CachedComposeTreeAscendC(s, t, seqs, tile_rows=tile_rows, merge_levels=depth)
    first = plan(s, t)
    second = plan(s, t)
    assert first.data_ptr() != second.data_ptr()
    for actual in (first, second):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(s, saved_s, atol=0, rtol=0)
    torch.testing.assert_close(t, saved_t, atol=0, rtol=0)
    if count:
        seqs[0].clear()
        s.add_(0.01)
        frozen = [[(i * 3 + row) % 7 for i in range(max(0, count - row))] for row in range(4)]
        torch.testing.assert_close(plan(s, t), oracle(s, t, frozen), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("kind", ["identity", "zero", "noncommuting", "near_identity"])
@pytest.mark.parametrize("depth", [0, 2])
def test_tree_transition_semantics(impl, tile_rows, kind, depth):
    s, t = pools(slots=3, heads=1)
    if kind == "identity":
        t.copy_(torch.eye(128, device=s.device).expand_as(t))
    elif kind == "zero":
        t.zero_()
    elif kind == "noncommuting":
        t[0, 0] = torch.eye(128, device=s.device).roll(1, 1)
        t[1, 0] = torch.diag(torch.linspace(0.5, 1.0, 128, device=s.device))
    else:
        t.add_(torch.eye(128, device=s.device)[None, None] * 0.102)
    seqs = [[2, 0, 1, 0] * 17, [2, 1, 0, 1] * 16, [2], []]
    actual = impl.compose_cached_tree_ascendc(s, t, seqs, tile_rows=tile_rows, merge_levels=depth)
    torch.testing.assert_close(actual, oracle(s, t, seqs), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("batch", [0, 1, 4])
def test_tree_empty_pool(impl, tile_rows, batch):
    s, t = pools(slots=0)
    actual = impl.compose_cached_tree_ascendc(s, t, [[] for _ in range(batch)], tile_rows=tile_rows)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("depth", [0, 2])
def test_tree_queued_plan_and_workspace_lifetime(impl, tile_rows, depth):
    s, t = pools(heads=1)
    seqs = [[0, 3, 1, 4] * 16, [2, 2], []]
    expected = oracle(s, t, seqs)
    plan = impl.CachedComposeTreeAscendC(s, t, seqs, tile_rows=tile_rows, merge_levels=depth)
    for _ in range(4):
        plan(s, t)
    torch.npu.synchronize()
    initial = torch.npu.memory_allocated()
    for _ in range(4):
        for _ in range(32):
            actual = plan(s, t)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        del actual
    torch.npu.synchronize()
    assert torch.npu.memory_allocated() - initial < 2 * 1024 * 1024
    queued = [impl.compose_cached_tree_ascendc(s, t, seqs, tile_rows=tile_rows, merge_levels=depth) for _ in range(16)]
    torch.npu.synchronize()
    for actual in queued:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    with torch.npu.stream(torch.npu.Stream()), pytest.raises(ValueError, match="stream"):
        plan(s, t)


@pytest.mark.parametrize("slot", [-1, 5, 2**32, 1.5, True])
def test_tree_bad_slot(impl, slot):
    s, t = pools()
    with pytest.raises(ValueError, match="slot"):
        impl.CachedComposeTreeAscendC(s, t, [[slot]])


@pytest.mark.parametrize("depth", [-1, 32, True, 1.5, "2"])
def test_tree_bad_depth(impl, depth):
    s, t = pools()
    with pytest.raises(ValueError, match="merge_levels"):
        impl.CachedComposeTreeAscendC(s, t, [[0]], merge_levels=depth)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 16, 65, 129])
@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_single_tree_ragged_and_level_barriers(impl, tile_rows, count, heads, depth):
    s, t = pools(slots=7, heads=heads)
    seqs = [[(i * 3 + row) % 7 for i in range(max(0, count - row))] for row in range(4)]
    expected = oracle(s, t, seqs)
    saved_s, saved_t = s.clone(), t.clone()
    plan = impl.CachedComposeTreeAscendC(s, t, seqs, tile_rows=tile_rows, single_kernel=True, merge_levels=depth)
    layered = impl.compose_cached_tree_ascendc(s, t, seqs, tile_rows=tile_rows, merge_levels=depth)
    first, second = plan(s, t), plan(s, t)
    assert first.data_ptr() != second.data_ptr()
    for actual in (first, second):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual, layered, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(s, saved_s, atol=0, rtol=0)
    torch.testing.assert_close(t, saved_t, atol=0, rtol=0)
    seqs[0].clear()
    s.add_(0.01)
    frozen = [[(i * 3 + row) % 7 for i in range(max(0, count - row))] for row in range(4)]
    torch.testing.assert_close(plan(s, t), oracle(s, t, frozen), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("kind", ["identity", "zero", "noncommuting", "near_identity"])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_single_tree_transition_semantics(impl, tile_rows, kind, depth):
    s, t = pools(slots=3, heads=1)
    if kind == "identity":
        t.copy_(torch.eye(128, device=s.device).expand_as(t))
    elif kind == "zero":
        t.zero_()
    elif kind == "noncommuting":
        t[0, 0] = torch.eye(128, device=s.device).roll(1, 1)
        t[1, 0] = torch.diag(torch.linspace(0.5, 1.0, 128, device=s.device))
    else:
        t.add_(torch.eye(128, device=s.device)[None, None] * 0.102)
    seqs = [[2, 0, 1, 0] * 17, [2, 1, 0, 1] * 16, [2], []]
    actual = impl.compose_cached_tree_ascendc(s, t, seqs, tile_rows=tile_rows, single_kernel=True, merge_levels=depth)
    torch.testing.assert_close(actual, oracle(s, t, seqs), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("batch", [0, 1, 4])
def test_single_tree_empty_pool(impl, tile_rows, batch):
    s, t = pools(slots=0)
    actual = impl.compose_cached_tree_ascendc(s, t, [[] for _ in range(batch)], tile_rows=tile_rows, single_kernel=True)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_single_tree_queued_lifetime_and_memory(impl, tile_rows, depth):
    s, t = pools(heads=1)
    seqs = [[0, 3, 1, 4] * 16, [2, 2], []]
    expected = oracle(s, t, seqs)
    plan = impl.CachedComposeTreeAscendC(s, t, seqs, tile_rows=tile_rows, single_kernel=True, merge_levels=depth)
    for _ in range(4):
        plan(s, t)
    torch.npu.synchronize()
    initial = torch.npu.memory_allocated()
    for _ in range(4):
        for _ in range(32):
            actual = plan(s, t)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        del actual
    torch.npu.synchronize()
    assert torch.npu.memory_allocated() - initial < 2 * 1024 * 1024
    outputs = [
        impl.compose_cached_tree_ascendc(s, t, seqs, single_kernel=True, tile_rows=tile_rows, merge_levels=depth)
        for _ in range(16)
    ]
    torch.npu.synchronize()
    for actual in outputs:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    with torch.npu.stream(torch.npu.Stream()), pytest.raises(ValueError, match="stream"):
        plan(s, t)


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_single_tree_two_streams(impl, depth):
    s, t = pools(slots=65, heads=8)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    sequences = [[list(range(65)), list(range(64, -1, -1))], [list(range(64, 0, -1)), []]]
    expected = [oracle(s, t, seqs) for seqs in sequences]
    plans = []
    for index, stream in enumerate(streams):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            plans.append(impl.CachedComposeTreeAscendC(s, t, sequences[index], single_kernel=True, merge_levels=depth))
    outputs = []
    for _ in range(16):
        for index, (stream, plan) in enumerate(zip(streams, plans)):
            with torch.npu.stream(stream):
                outputs.append((index, plan(s, t)))
    torch.npu.synchronize()
    for index, actual in outputs:
        torch.testing.assert_close(actual, expected[index], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("options", [{"single_kernel": 1}, {"single_kernel": None}, {"single_kernel": "yes"}])
def test_single_tree_invalid_options(impl, options):
    s, t = pools()
    with pytest.raises(ValueError, match="single_kernel"):
        impl.CachedComposeTreeAscendC(s, t, [[0]], **({"single_kernel": True} | options))


@pytest.mark.parametrize("field,value", [(0, 100000), (1, 16), (2, 100000)])
def test_single_tree_bad_header_is_bounded(impl, field, value):
    s, t = pools(heads=1)
    plan = impl.CachedComposeTreeAscendC(s, t, [[0, 1, 2, 3]], single_kernel=True)
    packet = plan._single_packet.clone()
    packet[0, field] = value
    actual = torch.ops.hypic_ascendc.cached_compose_tree_single(
        s, t, packet, plan._single_tiling, plan._node_counts, plan._row_tiles, plan._workspace_bytes, plan._max_blocks
    )
    assert torch.isnan(actual).all().item()


def test_single_tree_rejects_nonresident_workers(impl):
    s, t = pools(heads=1)
    plan = impl.CachedComposeTreeAscendC(s, t, [[0, 1]], single_kernel=True)
    with pytest.raises(RuntimeError, match="physical cores"):
        torch.ops.hypic_ascendc.cached_compose_tree_single(
            s,
            t,
            plan._single_packet,
            plan._single_tiling,
            plan._node_counts,
            plan._row_tiles,
            plan._workspace_bytes,
            plan._max_blocks + 1,
        )


@pytest.mark.parametrize("field,value", [(0, -1), (0, 100000), (1, -1), (1, 100000)])
def test_single_hybrid_bad_tail_is_bounded(impl, field, value):
    s, t = pools(heads=1)
    plan = impl.CachedComposeTreeAscendC(s, t, [[0, 1, 2, 3, 4], [4, 3, 2]], single_kernel=True, merge_levels=1)
    packet = plan._single_tail_packet.clone()
    packet[0, field] = value
    actual = torch.ops.hypic_ascendc.cached_compose_tree_single(
        s,
        t,
        plan._single_packet,
        plan._single_tiling,
        plan._node_counts,
        plan._row_tiles,
        plan._workspace_bytes,
        plan._max_blocks,
        packet,
        plan._single_tail_tiling,
        plan._single_tail_rows,
    )
    assert torch.isnan(actual[0]).all().item()
    torch.testing.assert_close(actual[1:], oracle(s, t, [[4, 3, 2]]), atol=2e-5, rtol=2e-4)
