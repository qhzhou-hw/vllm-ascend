import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from vllm.config import ParallelConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    ReqMeta,
)

_MASTER_READY_TIMEOUT_SECONDS = 30
_HASH_BLOCK_SIZE = 16
_GROUP_BLOCK_SIZES = [16, 32]
_NUM_LAYERS = 2
_NUM_CACHE_BLOCKS = 4
_SENTINEL = -123.0
_DSV4_GROUP_BLOCK_SIZES = [128, 4096, 32, 32, 32, 2, 8]
_DSV4_CACHE_FAMILIES = ["c4", "c128", "mixed", "mixed", "mixed", "c4", "c128"]
_DSV4_HASH_BLOCK_SIZE = 2
_DSV4_NUM_LAYERS = 3
_DSV4_TOKEN_COUNT = 4096


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + _MASTER_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"mooncake_master exited with {process.returncode}:\n{log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"mooncake_master did not listen on port {port}")


@pytest.fixture
def mooncake_store_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    master_port = _free_port()
    metrics_port = _free_port()
    config_path = tmp_path / "mooncake.json"
    log_path = tmp_path / "mooncake-master.log"
    config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "device_name": "",
                "master_server_address": f"127.0.0.1:{master_port}",
                "global_segment_size": "1GB",
                "local_buffer_size": "64MB",
                "preferred_segment": False,
                "prefer_alloc_in_same_node": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MOONCAKE_MASTER", f"127.0.0.1:{master_port}")

    inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    mooncake_library_paths = [
        "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake",
        "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
    ]
    if inherited_library_path:
        mooncake_library_paths.append(inherited_library_path)
    process_env = {
        **os.environ,
        "LD_LIBRARY_PATH": os.pathsep.join(mooncake_library_paths),
    }
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                "mooncake_master",
                "--port",
                str(master_port),
                "--metrics_port",
                str(metrics_port),
                "--eviction_high_watermark_ratio",
                "0.9",
                "--eviction_ratio",
                "0.1",
                "--default_kv_lease_ttl",
                "11000",
            ],
            env=process_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_port(master_port, process, log_path)
            yield config_path
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=10)


def _new_cache(block_size: int, width: int, num_blocks: int = _NUM_CACHE_BLOCKS) -> torch.Tensor:
    return torch.empty(
        (num_blocks, block_size, width),
        dtype=torch.float16,
        device="npu",
    ).uniform_(-1.0, 1.0)


def _block_bytes(tensor: torch.Tensor) -> int:
    return tensor[0].numel() * tensor.element_size()


def _make_transfer_tasks(request: ReqMeta, physical_layer: int) -> list[LayerTransferTask]:
    return [
        LayerTransferTask(
            layer_id=physical_layer,
            group_id=0,
            layer_idx_in_group=physical_layer,
            block_ranges=[LayerBlockRange(request, 0, 2)],
        ),
        LayerTransferTask(
            layer_id=physical_layer,
            group_id=1,
            layer_idx_in_group=physical_layer,
            block_ranges=[LayerBlockRange(request, 0, 1)],
        ),
    ]


def test_mooncake_hybrid_layerwise_kv_roundtrip(mooncake_store_config: Path) -> None:
    del mooncake_store_config
    torch.npu.set_device(0)
    torch.manual_seed(20260829)

    # Group 0 models the normal SFA/DSA KV layout: one cache entry per layer.
    # Group 1 models the GLM-5.2/DSV4 indexer layout: the first layer has two
    # cache entries while the second has one, and it uses a larger block size.
    group_caches = {
        0: [_new_cache(16, 8), _new_cache(16, 8)],
        1: [_new_cache(32, 6), _new_cache(32, 4), _new_cache(32, 5)],
    }
    torch.npu.synchronize()
    originals = {group_id: [tensor.cpu().clone() for tensor in tensors] for group_id, tensors in group_caches.items()}

    test_namespace = f"dsv4-layerwise-roundtrip-{uuid.uuid4().hex}"
    database = ChunkedTokenDatabase(
        [
            KeyMetadata(test_namespace, 0, 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata(test_namespace, 0, 0, 0, 0, kv_cache_group_id=1),
        ],
        _GROUP_BLOCK_SIZES,
        partitions=None,
        hash_block_size=_HASH_BLOCK_SIZE,
    )
    group_base_addrs = {
        group_id: [tensor.data_ptr() for tensor in tensors] for group_id, tensors in group_caches.items()
    }
    group_block_lengths = {
        group_id: [_block_bytes(tensor) for tensor in tensors] for group_id, tensors in group_caches.items()
    }
    database.set_group_buffers(
        group_base_addrs,
        group_block_lengths,
        group_block_stride=group_block_lengths,
        group_num_layers={0: 2, 1: 2},
        group_layer_cache_entry_offsets={0: [0, 1, 2], 1: [0, 2, 3]},
    )

    backend = MooncakeBackend(ParallelConfig())
    all_caches = [tensor for tensors in group_caches.values() for tensor in tensors]
    backend.register_buffer(
        [tensor.data_ptr() for tensor in all_caches],
        [tensor.numel() * tensor.element_size() for tensor in all_caches],
    )
    io_stats = {
        "put_calls": 0,
        "put_keys": 0,
        "put_bytes": 0,
        "get_calls": 0,
        "get_keys": 0,
        "get_bytes": 0,
    }
    real_put = backend.put
    real_get = backend.get

    def counted_put(keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> None:
        io_stats["put_calls"] += 1
        io_stats["put_keys"] += len(keys)
        io_stats["put_bytes"] += sum(sum(key_sizes) for key_sizes in sizes)
        real_put(keys, addrs, sizes)

    def counted_get(keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None:
        io_stats["get_calls"] += 1
        io_stats["get_keys"] += len(keys)
        io_stats["get_bytes"] += sum(sum(key_sizes) for key_sizes in sizes)
        return real_get(keys, addrs, sizes)

    # Count the public backend calls while still invoking the real Mooncake
    # batch_put_from_multi_buffers / batch_get_into_multi_buffers methods.
    backend.put = counted_put  # type: ignore[method-assign]
    backend.get = counted_get  # type: ignore[method-assign]

    request = ReqMeta(
        req_id="roundtrip",
        token_len_chunk=32,
        save_end_token=32,
        block_ids_by_group=[[1, 2], [1]],
        block_hashes=["block-hash-0", "block-hash-1"],
        is_last_chunk=True,
    )
    layer_save_events = [threading.Event() for _ in range(_NUM_LAYERS)]
    sync_save_events = [torch.npu.Event() for _ in range(_NUM_LAYERS)]
    for event in sync_save_events:
        event.record()
    sender = KVCacheStoreKeyLayerSendingThread(
        backend,
        database,
        _GROUP_BLOCK_SIZES,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        ready_event=threading.Event(),
        num_layers=_NUM_LAYERS,
        layer_save_finished_events=layer_save_events,
        sync_save_events=sync_save_events,
    )

    for physical_layer in range(_NUM_LAYERS):
        tasks = _make_transfer_tasks(request, physical_layer)
        for task in tasks:
            task.cached_process_tokens = sender.build_cached_process_tokens(task)
            sender.add_stored_request(request.req_id)
        sender.request_queue.put(tasks)
        sender._handle_request(tasks)

    expected_keys = []
    for group_id in range(2):
        for _, _, key in database.process_tokens(
            request.save_end_token,
            request.block_hashes,
            kv_cache_group_id=group_id,
        ):
            expected_keys.extend(layer_key.to_string() for layer_key in key.split_layers(_NUM_LAYERS))
    assert len(expected_keys) == 6
    assert all(backend.exists(expected_keys)), "not all per-group, per-layer keys were published"

    for tensor in all_caches:
        tensor.fill_(_SENTINEL)
    torch.npu.synchronize()

    layer_load_events = [threading.Event() for _ in range(_NUM_LAYERS)]
    receiver = KVCacheStoreKeyLayerRecvingThread(
        backend,
        database,
        _GROUP_BLOCK_SIZES,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        ready_event=threading.Event(),
        get_event=threading.Event(),
        layer_load_finished_events=layer_load_events,
        layer_save_finished_events=[threading.Event() for _ in range(_NUM_LAYERS)],
        num_layers=_NUM_LAYERS,
    )
    for physical_layer in range(_NUM_LAYERS):
        load_task = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=_make_transfer_tasks(request, physical_layer),
            layer_id=physical_layer,
        )
        receiver.request_queue.put(load_task)
        receiver._handle_request(load_task)
    torch.npu.synchronize()

    max_abs_diff = 0.0
    restored_blocks = {0: (1, 2), 1: (1,)}
    untouched_blocks = {0: (0, 3), 1: (0, 2, 3)}
    for group_id, tensors in group_caches.items():
        for cache_index, tensor in enumerate(tensors):
            actual = tensor.cpu()
            expected = originals[group_id][cache_index]
            for block_id in restored_blocks[group_id]:
                difference = (actual[block_id].float() - expected[block_id].float()).abs().max().item()
                max_abs_diff = max(max_abs_diff, difference)
                assert torch.equal(actual[block_id], expected[block_id])
            for block_id in untouched_blocks[group_id]:
                assert torch.all(actual[block_id] == _SENTINEL)

    assert max_abs_diff == 0.0
    assert io_stats["put_calls"] > 0
    assert io_stats["get_calls"] > 0
    assert io_stats["put_keys"] == len(expected_keys)
    assert io_stats["get_keys"] == len(expected_keys)
    assert io_stats["put_bytes"] == io_stats["get_bytes"]
    print(
        json.dumps(
            {
                "backend": "mooncake",
                "cache_groups": 2,
                "physical_layers": _NUM_LAYERS,
                "stored_keys": len(expected_keys),
                **io_stats,
                "max_abs_diff": max_abs_diff,
                "bitwise_equal": True,
            },
            sort_keys=True,
        )
    )


def test_mooncake_dsv4_layerwise_group_geometry_roundtrip(mooncake_store_config: Path) -> None:
    del mooncake_store_config
    torch.npu.set_device(0)
    torch.manual_seed(20260901)

    blocks_per_group = [_DSV4_TOKEN_COUNT // block_size for block_size in _DSV4_GROUP_BLOCK_SIZES]
    group_caches = {
        group_id: [
            _new_cache(
                block_size,
                width=group_id + layer_id + 1,
                num_blocks=blocks_per_group[group_id] + 2,
            )
            for layer_id in range(_DSV4_NUM_LAYERS)
        ]
        for group_id, block_size in enumerate(_DSV4_GROUP_BLOCK_SIZES)
    }
    torch.npu.synchronize()
    originals = {group_id: [tensor.cpu().clone() for tensor in tensors] for group_id, tensors in group_caches.items()}

    test_namespace = f"dsv4-seven-group-roundtrip-{uuid.uuid4().hex}"
    database = ChunkedTokenDatabase(
        [
            KeyMetadata(
                test_namespace,
                0,
                0,
                0,
                0,
                kv_cache_group_id=group_id,
                cache_family=_DSV4_CACHE_FAMILIES[group_id],
            )
            for group_id in range(len(_DSV4_GROUP_BLOCK_SIZES))
        ],
        _DSV4_GROUP_BLOCK_SIZES,
        partitions=None,
        hash_block_size=_DSV4_HASH_BLOCK_SIZE,
    )
    group_block_lengths = {
        group_id: [_block_bytes(tensor) for tensor in tensors] for group_id, tensors in group_caches.items()
    }
    database.set_group_buffers(
        {group_id: [tensor.data_ptr() for tensor in tensors] for group_id, tensors in group_caches.items()},
        group_block_lengths,
        group_block_stride=group_block_lengths,
        group_num_layers={group_id: _DSV4_NUM_LAYERS for group_id in group_caches},
        group_layer_cache_entry_offsets={group_id: list(range(_DSV4_NUM_LAYERS + 1)) for group_id in group_caches},
    )

    backend = MooncakeBackend(ParallelConfig())
    all_caches = [tensor for tensors in group_caches.values() for tensor in tensors]
    backend.register_buffer(
        [tensor.data_ptr() for tensor in all_caches],
        [tensor.numel() * tensor.element_size() for tensor in all_caches],
    )
    io_stats = {"put_calls": 0, "get_calls": 0, "put_bytes": 0, "get_bytes": 0}
    real_put = backend.put
    real_get = backend.get

    def counted_put(keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> None:
        io_stats["put_calls"] += 1
        io_stats["put_bytes"] += sum(sum(key_sizes) for key_sizes in sizes)
        real_put(keys, addrs, sizes)

    def counted_get(keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None:
        io_stats["get_calls"] += 1
        io_stats["get_bytes"] += sum(sum(key_sizes) for key_sizes in sizes)
        return real_get(keys, addrs, sizes)

    backend.put = counted_put  # type: ignore[method-assign]
    backend.get = counted_get  # type: ignore[method-assign]
    request = ReqMeta(
        req_id="dsv4-seven-group-roundtrip",
        token_len_chunk=_DSV4_TOKEN_COUNT,
        save_end_token=_DSV4_TOKEN_COUNT,
        block_ids_by_group=[list(range(1, num_blocks + 1)) for num_blocks in blocks_per_group],
        block_hashes=[f"dsv4-hash-{index}" for index in range(_DSV4_TOKEN_COUNT // _DSV4_HASH_BLOCK_SIZE)],
        is_last_chunk=True,
    )

    def make_tasks(physical_layer: int) -> list[LayerTransferTask]:
        return [
            LayerTransferTask(
                layer_id=physical_layer,
                group_id=group_id,
                layer_idx_in_group=physical_layer,
                block_ranges=[LayerBlockRange(request, 0, blocks_per_group[group_id])],
            )
            for group_id in range(len(_DSV4_GROUP_BLOCK_SIZES))
        ]

    layer_save_events = [threading.Event() for _ in range(_DSV4_NUM_LAYERS)]
    sync_save_events = [torch.npu.Event() for _ in range(_DSV4_NUM_LAYERS)]
    for event in sync_save_events:
        event.record()
    sender = KVCacheStoreKeyLayerSendingThread(
        backend,
        database,
        _DSV4_GROUP_BLOCK_SIZES,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        ready_event=threading.Event(),
        num_layers=_DSV4_NUM_LAYERS,
        layer_save_finished_events=layer_save_events,
        sync_save_events=sync_save_events,
    )
    for physical_layer in range(_DSV4_NUM_LAYERS):
        tasks = make_tasks(physical_layer)
        for task in tasks:
            task.cached_process_tokens = sender.build_cached_process_tokens(task)
            sender.add_stored_request(request.req_id)
        sender.request_queue.put(tasks)
        sender._handle_request(tasks)

    expected_keys = []
    for group_id in range(len(_DSV4_GROUP_BLOCK_SIZES)):
        for _, _, key in database.process_tokens(
            request.save_end_token,
            request.block_hashes,
            kv_cache_group_id=group_id,
        ):
            expected_keys.extend(layer_key.to_string() for layer_key in key.split_layers(_DSV4_NUM_LAYERS))
    assert len(expected_keys) == sum(blocks_per_group) * _DSV4_NUM_LAYERS
    assert all(backend.exists(expected_keys))

    for tensor in all_caches:
        tensor.fill_(_SENTINEL)
    torch.npu.synchronize()

    receiver = KVCacheStoreKeyLayerRecvingThread(
        backend,
        database,
        _DSV4_GROUP_BLOCK_SIZES,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        ready_event=threading.Event(),
        get_event=threading.Event(),
        layer_load_finished_events=[threading.Event() for _ in range(_DSV4_NUM_LAYERS)],
        layer_save_finished_events=[threading.Event() for _ in range(_DSV4_NUM_LAYERS)],
        num_layers=_DSV4_NUM_LAYERS,
    )
    for physical_layer in range(_DSV4_NUM_LAYERS):
        load_task = LayerLoadTask(None, make_tasks(physical_layer), layer_id=physical_layer)
        receiver.request_queue.put(load_task)
        receiver._handle_request(load_task)
    torch.npu.synchronize()

    for group_id, tensors in group_caches.items():
        restored_blocks = range(1, blocks_per_group[group_id] + 1)
        untouched_blocks = (0, blocks_per_group[group_id] + 1)
        for cache_index, tensor in enumerate(tensors):
            actual = tensor.cpu()
            expected = originals[group_id][cache_index]
            for block_id in restored_blocks:
                assert torch.equal(actual[block_id], expected[block_id])
            for block_id in untouched_blocks:
                assert torch.all(actual[block_id] == _SENTINEL)

    assert io_stats["put_calls"] == _DSV4_NUM_LAYERS
    assert io_stats["get_calls"] == _DSV4_NUM_LAYERS
    assert io_stats["put_bytes"] == io_stats["get_bytes"]
    print(
        json.dumps(
            {
                "backend": "mooncake",
                "cache_groups": len(_DSV4_GROUP_BLOCK_SIZES),
                "group_block_sizes": _DSV4_GROUP_BLOCK_SIZES,
                "physical_layers": _DSV4_NUM_LAYERS,
                "stored_keys": len(expected_keys),
                **io_stats,
                "max_abs_diff": 0.0,
                "bitwise_equal": True,
            },
            sort_keys=True,
        )
    )


def test_mooncake_kimi_k3_aligned_kda_state_roundtrip(mooncake_store_config: Path) -> None:
    del mooncake_store_config
    torch.npu.set_device(0)
    torch.manual_seed(20260830)

    block_size = 32
    conv_state = _new_cache(block_size, 4)
    ssm_state = _new_cache(block_size, 7)
    states = [conv_state, ssm_state]
    torch.npu.synchronize()
    originals = [state.cpu().clone() for state in states]

    database = ChunkedTokenDatabase(
        [KeyMetadata(f"kimi-k3-kda-{uuid.uuid4().hex}", 0, 0, 0, 0)],
        [block_size],
        partitions=None,
        hash_block_size=_HASH_BLOCK_SIZE,
    )
    state_block_lengths = [_block_bytes(state) for state in states]
    database.set_group_buffers(
        {0: [state.data_ptr() for state in states]},
        {0: state_block_lengths},
        group_block_stride={0: state_block_lengths},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0, 2]},
    )

    backend = MooncakeBackend(ParallelConfig())
    backend.register_buffer(
        [state.data_ptr() for state in states],
        [state.numel() * state.element_size() for state in states],
    )
    io_stats = {
        "put_calls": 0,
        "put_keys": 0,
        "put_bytes": 0,
        "get_calls": 0,
        "get_keys": 0,
        "get_bytes": 0,
    }
    real_put = backend.put
    real_get = backend.get

    def counted_put(keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> None:
        io_stats["put_calls"] += 1
        io_stats["put_keys"] += len(keys)
        io_stats["put_bytes"] += sum(sum(key_sizes) for key_sizes in sizes)
        real_put(keys, addrs, sizes)

    def counted_get(keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None:
        io_stats["get_calls"] += 1
        io_stats["get_keys"] += len(keys)
        io_stats["get_bytes"] += sum(sum(key_sizes) for key_sizes in sizes)
        return real_get(keys, addrs, sizes)

    backend.put = counted_put  # type: ignore[method-assign]
    backend.get = counted_get  # type: ignore[method-assign]
    request = ReqMeta(
        req_id="kimi-k3-kda-roundtrip",
        token_len_chunk=64,
        save_end_token=64,
        block_ids_by_group=[[0, 2]],
        block_hashes=["mamba-h0", "mamba-h1", "mamba-h2", "mamba-h3"],
        is_last_chunk=True,
        skip_null_blocks_by_group=[True],
    )

    def make_task() -> LayerTransferTask:
        return LayerTransferTask(
            layer_id=0,
            group_id=0,
            layer_idx_in_group=0,
            block_ranges=[LayerBlockRange(request, 0, 2)],
        )

    save_event = threading.Event()
    sync_event = torch.npu.Event()
    sync_event.record()
    sender = KVCacheStoreKeyLayerSendingThread(
        backend,
        database,
        [block_size],
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        ready_event=threading.Event(),
        num_layers=1,
        layer_save_finished_events=[save_event],
        sync_save_events=[sync_event],
    )
    save_task = make_task()
    save_task.cached_process_tokens = sender.build_cached_process_tokens(save_task)
    sender.add_stored_request(request.req_id)
    sender.request_queue.put([save_task])
    sender._handle_request([save_task])

    all_keys = []
    for _, _, key in database.process_tokens(
        request.save_end_token,
        request.block_hashes,
    ):
        all_keys.append(key.split_layers(1)[0].to_string())
    exists = backend.exists(all_keys)
    assert exists == [0, 1], "only the live aligned Mamba state must be stored"

    for state in states:
        state.fill_(_SENTINEL)
    torch.npu.synchronize()

    receiver = KVCacheStoreKeyLayerRecvingThread(
        backend,
        database,
        [block_size],
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        ready_event=threading.Event(),
        get_event=threading.Event(),
        layer_load_finished_events=[threading.Event()],
        layer_save_finished_events=[threading.Event()],
        num_layers=1,
    )
    load_task = LayerLoadTask(None, [make_task()], layer_id=0)
    receiver.request_queue.put(load_task)
    receiver._handle_request(load_task)
    torch.npu.synchronize()

    for actual, expected in zip(states, originals):
        actual_cpu = actual.cpu()
        assert torch.all(actual_cpu[0] == _SENTINEL)
        assert torch.equal(actual_cpu[2], expected[2])
        assert torch.all(actual_cpu[1] == _SENTINEL)
        assert torch.all(actual_cpu[3] == _SENTINEL)

    assert io_stats["put_calls"] == 1
    assert io_stats["get_calls"] == 1
    assert io_stats["put_keys"] == 1
    assert io_stats["get_keys"] == 1
    assert io_stats["put_bytes"] == io_stats["get_bytes"]
    print(
        json.dumps(
            {
                "backend": "mooncake",
                "model_cache": "kimi-k3-kda",
                "state_entries": len(states),
                **io_stats,
                "max_abs_diff": 0.0,
                "bitwise_equal": True,
            },
            sort_keys=True,
        )
    )
