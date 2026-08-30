# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import json
from pathlib import Path

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import MooncakeLauncher, RemoteOpenAIServer, wait_until_npu_memory_free

_MODEL = "Qwen/Qwen3.5-0.8B"
_NUM_PHYSICAL_LAYERS = 24
_MIN_TRANSFER_TOKENS = 1024
_METRICS_TIMEOUT_SECONDS = 5
_PROMPT = (
    "Qwen3.5 hybrid Mooncake validation must restore full-attention KV and "
    "GatedDeltaNet recurrent state. Use deterministic tokens and repeat this "
    "sentence so the prefix spans a complete hybrid cache transfer unit. "
) * 32


def _metric(metrics_port: int, name: str) -> float:
    response = requests.get(
        f"http://127.0.0.1:{metrics_port}/metrics",
        timeout=_METRICS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    prefix = f"{name} "
    for line in response.text.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    raise AssertionError(f"Mooncake metric not found: {name}")


@pytest.mark.e2e_model(_MODEL)
@pytest.mark.e2e_coverage(
    arch="hybrid",
    feature="mamba_ssm",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_qwen3_5_mooncake_layerwise_reuses_hybrid_cache(tmp_path: Path) -> None:
    server_port = get_open_port()
    mooncake_port = get_open_port()
    metrics_port = get_open_port()
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "device_name": "",
                "master_server_address": f"127.0.0.1:{mooncake_port}",
                "global_segment_size": "1GB",
                "local_buffer_size": "64MB",
                "preferred_segment": False,
                "prefer_alloc_in_same_node": True,
            }
        ),
        encoding="utf-8",
    )
    env_dict = {
        "PYTHONHASHSEED": "0",
        "OMP_PROC_BIND": "false",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "MOONCAKE_CONFIG_PATH": str(config_path),
        "MOONCAKE_MASTER": f"127.0.0.1:{mooncake_port}",
    }
    kv_transfer_config = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "lookup_rpc_port": "0",
            "use_layerwise": True,
        },
    }
    server_args = [
        "--served-model-name",
        _MODEL,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1536",
        "--max-num-seqs",
        "1",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.3",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--enforce-eager",
        "--port",
        str(server_port),
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]

    with (
        MooncakeLauncher(mooncake_port, metrics_port),
        RemoteOpenAIServer(
            _MODEL,
            server_args,
            server_port=server_port,
            env_dict=env_dict,
            auto_port=False,
        ) as server,
    ):
        client = server.get_client()
        request_args = {
            "model": _MODEL,
            "prompt": _PROMPT,
            "max_tokens": 16,
            "temperature": 0,
            "seed": 123,
        }
        first = client.completions.create(**request_args)
        second = client.completions.create(**request_args)

        assert first.choices[0].text == second.choices[0].text
        assert first.usage is not None and second.usage is not None
        assert first.usage.prompt_tokens == second.usage.prompt_tokens
        assert first.usage.completion_tokens == second.usage.completion_tokens
        assert first.usage.prompt_tokens >= _MIN_TRANSFER_TOKENS

        metrics = {
            name: _metric(metrics_port, name)
            for name in (
                "master_batch_put_start_requests_total",
                "master_batch_put_end_requests_total",
                "master_batch_get_replica_list_requests_total",
                "master_batch_put_start_failures_total",
                "master_batch_put_end_failures_total",
                "master_batch_get_replica_list_failures_total",
                "master_key_count",
            )
        }
        assert metrics["master_batch_put_start_requests_total"] >= _NUM_PHYSICAL_LAYERS
        assert metrics["master_batch_put_end_requests_total"] >= _NUM_PHYSICAL_LAYERS
        assert metrics["master_batch_get_replica_list_requests_total"] >= _NUM_PHYSICAL_LAYERS
        assert metrics["master_batch_put_start_failures_total"] == 0
        assert metrics["master_batch_put_end_failures_total"] == 0
        assert metrics["master_batch_get_replica_list_failures_total"] == 0
        assert metrics["master_key_count"] >= _NUM_PHYSICAL_LAYERS
