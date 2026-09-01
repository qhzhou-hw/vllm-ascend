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

"""DeepSeek-V4 Mooncake first-token tensor consistency validation.

This test intentionally uses a shortened DeepSeek-V4 config and deterministic
dummy weights. It validates the AscendStoreConnector data path, not model
accuracy. The complete first generated-token log-probability vector is compared
for full recompute, non-layerwise Mooncake load, and layerwise Mooncake load.

The file can also be run directly on an Ascend host. See ``--help`` for the
required model and LongBench paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_MODEL_PATH_ENV = "VLLM_ASCEND_DSV4_TINY_MODEL_PATH"
_LONGBENCH_PATH_ENV = "VLLM_ASCEND_LONGBENCH_PATH"
_DATASET = "qasper"
_DEFAULT_NUM_SAMPLES = 3
_DEFAULT_PROMPT_TOKENS = 192
_DEFAULT_ATOL = 1e-3
_MAX_MODEL_LEN = 256
_CACHE_BLOCK_SIZE = 32
_DEFAULT_MODEL_OVERRIDES = {
    "index_topk": 512,
    "compress_ratios": [0, 0, 4, 4],
}
_TRANSFER_BLOCK_SIZE = 128
_DEFAULT_GLOBAL_SEGMENT_SIZE = "1GB"
_METRICS_TIMEOUT_SECONDS = 10
_CASE_FULL_RECOMPUTE = "full_recompute"
_CASE_MOONCAKE = "mooncake"
_CASE_MOONCAKE_LAYERWISE = "mooncake_layerwise"
_CASES = (
    _CASE_FULL_RECOMPUTE,
    _CASE_MOONCAKE,
    _CASE_MOONCAKE_LAYERWISE,
)
_METRIC_NAMES = (
    "master_batch_put_start_requests_total",
    "master_batch_put_end_requests_total",
    "master_batch_get_replica_list_requests_total",
    "master_batch_put_start_failures_total",
    "master_batch_put_end_failures_total",
    "master_batch_get_replica_list_failures_total",
    "master_key_count",
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _middle_truncate(token_ids: list[int], length: int) -> list[int]:
    if len(token_ids) <= length:
        return token_ids
    left_length = length // 2
    return token_ids[:left_length] + token_ids[-(length - left_length) :]


def _deepseek_transfer_granularity(model_overrides: dict[str, Any]) -> int:
    compress_ratios = model_overrides.get("compress_ratios", ())
    max_compress_ratio = max((int(ratio) for ratio in compress_ratios), default=1)
    return max(_TRANSFER_BLOCK_SIZE, _CACHE_BLOCK_SIZE * max_compress_ratio)


def _load_longbench_prompts(
    model_path: Path,
    longbench_path: Path,
    num_samples: int,
    prompt_tokens: int,
    prompt_lengths: list[int] | None = None,
) -> tuple[list[list[int]], list[str]]:
    # AutoTokenizer cannot load deepseek_v4 with older Transformers releases.
    # The tokenizers runtime reads tokenizer.json without changing dependencies.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    prompt_config_path = longbench_path / "official_config" / "dataset2prompt.json"
    dataset_path = longbench_path / "datasets" / f"{_DATASET}.jsonl"
    with prompt_config_path.open(encoding="utf-8") as file:
        prompt_template = json.load(file)[_DATASET]

    prompts: list[list[int]] = []
    sample_ids: list[str] = []
    target_lengths = prompt_lengths or [prompt_tokens] * num_samples
    if len(target_lengths) != num_samples:
        raise ValueError("prompt_lengths must contain one length per sample.")
    with dataset_path.open(encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            prompt = prompt_template.format(**sample)
            token_ids = tokenizer.encode(prompt).ids
            target_length = target_lengths[len(prompts)]
            if len(token_ids) < max(_TRANSFER_BLOCK_SIZE, target_length):
                continue
            token_ids = _middle_truncate(token_ids, target_length)
            prompts.append(token_ids)
            sample_ids.append(str(sample.get("_id", len(sample_ids))))
            if len(prompts) == num_samples:
                break

    if len(prompts) != num_samples:
        raise AssertionError(
            f"Only found {len(prompts)} LongBench prompts satisfying target "
            f"lengths {target_lengths}; expected {num_samples}."
        )
    return prompts, sample_ids


def _full_logprob_tensor(request_output: Any) -> np.ndarray:
    output = request_output.outputs[0]
    if not output.logprobs or not output.logprobs[0]:
        raise AssertionError("vLLM did not return first-token log probabilities.")
    logprobs = output.logprobs[0]
    vocab_size = len(logprobs)
    tensor = np.empty(vocab_size, dtype=np.float32)
    seen = np.zeros(vocab_size, dtype=np.bool_)
    for token_id, logprob in logprobs.items():
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            raise AssertionError(f"Unexpected token id {token_id} for vocab size {vocab_size}.")
        tensor[token_id] = float(logprob.logprob)
        seen[token_id] = True
    if not seen.all():
        raise AssertionError(f"Missing {int((~seen).sum())} entries from the full log-prob tensor.")
    return tensor


def _run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    model_path = Path(args.model_path).resolve()
    longbench_path = Path(args.longbench_path).resolve()
    prompts, sample_ids = _load_longbench_prompts(
        model_path,
        longbench_path,
        args.num_samples,
        args.prompt_tokens,
        json.loads(args.prompt_lengths_json) if args.prompt_lengths_json else None,
    )

    model_overrides = (
        json.loads(args.model_overrides_json) if args.model_overrides_json else dict(_DEFAULT_MODEL_OVERRIDES)
    )

    llm_args: dict[str, Any] = {
        "model": str(model_path),
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "block_size": _CACHE_BLOCK_SIZE,
        "gpu_memory_utilization": 0.2,
        "enforce_eager": True,
        "load_format": "dummy",
        "seed": 0,
        "max_logprobs": -1,
        "hf_overrides": model_overrides,
        # Disabling local prefix caching makes the second request exercise the
        # external store instead of reusing HBM cache blocks.
        "enable_prefix_caching": False,
    }
    if args.max_num_batched_tokens is not None:
        llm_args["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.case != _CASE_FULL_RECOMPUTE:
        llm_args["kv_transfer_config"] = {
            "kv_connector": "AscendStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "backend": "mooncake",
                "lookup_rpc_port": "0",
                "use_layerwise": args.case == _CASE_MOONCAKE_LAYERWISE,
            },
        }

    llm = LLM(**llm_args)
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=-1, seed=123)
    if args.case != _CASE_FULL_RECOMPUTE:
        # Cold pass: compute every prompt token and save transferable prefixes.
        llm.generate(prompts, sampling_params, use_tqdm=False)
    # Ground truth computes the prompt once. Connector cases reuse the cold-pass
    # KV data and capture the hot-pass first-token tensor.
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    tensors = np.stack([_full_logprob_tensor(output) for output in outputs])
    output_token_ids = np.asarray(
        [output.outputs[0].token_ids[0] for output in outputs],
        dtype=np.int64,
    )
    np.savez(
        args.output_path,
        tensors=tensors,
        output_token_ids=output_token_ids,
        prompt_lengths=np.asarray([len(prompt) for prompt in prompts], dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
    )


def _read_metrics(metrics_port: int) -> dict[str, float]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{metrics_port}/metrics",
        timeout=_METRICS_TIMEOUT_SECONDS,
    ) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for name in _METRIC_NAMES:
        prefix = f"{name} "
        for line in text.splitlines():
            if line.startswith(prefix):
                values[name] = float(line.removeprefix(prefix))
                break
        else:
            raise AssertionError(f"Mooncake metric not found: {name}")
    return values


def _wait_for_mooncake(metrics_port: int) -> None:
    deadline = time.monotonic() + _METRICS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            _read_metrics(metrics_port)
            return
        except (OSError, AssertionError):
            time.sleep(0.2)
    raise TimeoutError("Mooncake master metrics endpoint did not become ready.")


@contextmanager
def _mooncake_master(
    output_dir: Path,
    case: str,
    global_segment_size: str,
) -> Iterator[tuple[int, dict[str, str]]]:
    master_path = shutil.which("mooncake_master")
    if master_path is None:
        raise FileNotFoundError("mooncake_master was not found in PATH.")
    master_port = _get_open_port()
    metrics_port = _get_open_port()
    config_path = output_dir / f"{case}-mooncake.json"
    config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "device_name": "",
                "master_server_address": f"127.0.0.1:{master_port}",
                "global_segment_size": global_segment_size,
                "local_buffer_size": "64MB",
                "preferred_segment": False,
                "prefer_alloc_in_same_node": True,
            }
        ),
        encoding="utf-8",
    )
    master_log_path = output_dir / f"{case}-mooncake-master.log"
    master_log = master_log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    mooncake_lib = "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake"
    env["LD_LIBRARY_PATH"] = mooncake_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    process = subprocess.Popen(
        [
            master_path,
            "--port",
            str(master_port),
            "--metrics_port",
            str(metrics_port),
        ],
        env=env,
        stdout=master_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_mooncake(metrics_port)
        yield (
            metrics_port,
            {
                "MOONCAKE_CONFIG_PATH": str(config_path),
                "MOONCAKE_MASTER": f"127.0.0.1:{master_port}",
            },
        )
    finally:
        try:
            process_group = os.getpgid(process.pid)
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            process_group = None
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if process_group is not None:
                os.killpg(process_group, signal.SIGKILL)
            process.wait(timeout=5)
        master_log.close()


def _worker_command(
    case: str,
    model_path: Path,
    longbench_path: Path,
    output_path: Path,
    num_samples: int,
    prompt_tokens: int,
    prompt_lengths: list[int] | None,
    model_overrides: dict[str, Any],
    max_model_len: int,
    max_num_batched_tokens: int | None,
    max_num_seqs: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--case",
        case,
        "--model-path",
        str(model_path),
        "--longbench-path",
        str(longbench_path),
        "--output-path",
        str(output_path),
        "--num-samples",
        str(num_samples),
        "--prompt-tokens",
        str(prompt_tokens),
        "--model-overrides-json",
        json.dumps(model_overrides),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    if prompt_lengths is not None:
        command.extend(("--prompt-lengths-json", json.dumps(prompt_lengths)))
    if max_num_batched_tokens is not None:
        command.extend(("--max-num-batched-tokens", str(max_num_batched_tokens)))
    return command


def _run_case(
    case: str,
    model_path: Path,
    longbench_path: Path,
    output_dir: Path,
    num_samples: int,
    prompt_tokens: int,
    prompt_lengths: list[int] | None,
    model_overrides: dict[str, Any],
    max_model_len: int,
    max_num_batched_tokens: int | None,
    max_num_seqs: int,
    global_segment_size: str,
) -> tuple[Path, dict[str, float] | None]:
    output_path = output_dir / f"{case}.npz"
    log_path = output_dir / f"{case}.log"
    worker_env = os.environ.copy()
    worker_env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_PROC_BIND": "false",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "VLLM_LOGGING_LEVEL": "DEBUG",
        }
    )

    def execute_worker() -> None:
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                _worker_command(
                    case,
                    model_path,
                    longbench_path,
                    output_path,
                    num_samples,
                    prompt_tokens,
                    prompt_lengths,
                    model_overrides,
                    max_model_len,
                    max_num_batched_tokens,
                    max_num_seqs,
                ),
                env=worker_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )

    if case == _CASE_FULL_RECOMPUTE:
        execute_worker()
        return output_path, None

    with _mooncake_master(output_dir, case, global_segment_size) as (metrics_port, mooncake_env):
        worker_env.update(mooncake_env)
        execute_worker()
        metrics = _read_metrics(metrics_port)

    log_text = log_path.read_text(encoding="utf-8")
    assert "MooncakeBackend.put enter" in log_text
    assert "MooncakeBackend.get enter" in log_text
    assert "Failed to put" not in log_text
    assert "Failed to get" not in log_text
    assert metrics["master_batch_put_start_requests_total"] > 0
    assert metrics["master_batch_put_end_requests_total"] > 0
    assert metrics["master_batch_get_replica_list_requests_total"] > 0
    assert metrics["master_batch_put_start_failures_total"] == 0
    assert metrics["master_batch_put_end_failures_total"] == 0
    assert metrics["master_batch_get_replica_list_failures_total"] == 0
    assert metrics["master_key_count"] > 0
    if case == _CASE_MOONCAKE_LAYERWISE:
        with (model_path / "config.json").open(encoding="utf-8") as file:
            num_hidden_layers = int(
                model_overrides.get(
                    "num_hidden_layers",
                    json.load(file)["num_hidden_layers"],
                )
            )
        effective_lengths = prompt_lengths or [prompt_tokens] * num_samples
        transfer_granularity = _deepseek_transfer_granularity(model_overrides)
        transferable_requests = sum(length >= transfer_granularity for length in effective_lengths)
        if transferable_requests:
            # A backend RPC can contain several requests from the same layer
            # when max_num_seqs > 1, so request count cannot be multiplied by
            # layer count. Every physical layer must still issue at least one
            # successful put and get across the pressure batch.
            assert metrics["master_batch_put_start_requests_total"] >= num_hidden_layers
            assert metrics["master_batch_put_end_requests_total"] >= num_hidden_layers
            assert metrics["master_batch_get_replica_list_requests_total"] >= num_hidden_layers
    return output_path, metrics


def _compare_cases(result_paths: dict[str, Path], atol: float) -> dict[str, Any]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for case, path in result_paths.items():
        with np.load(path) as result:
            loaded[case] = {name: result[name] for name in result.files}
    ground = loaded[_CASE_FULL_RECOMPUTE]
    summary: dict[str, Any] = {
        "sample_ids": ground["sample_ids"].tolist(),
        "prompt_lengths": ground["prompt_lengths"].tolist(),
        "vocab_size": int(ground["tensors"].shape[1]),
        "atol": atol,
        "comparisons": {},
    }
    for case in (_CASE_MOONCAKE, _CASE_MOONCAKE_LAYERWISE):
        actual = loaded[case]
        assert np.array_equal(actual["sample_ids"], ground["sample_ids"])
        assert np.array_equal(actual["prompt_lengths"], ground["prompt_lengths"])
        delta = np.abs(actual["tensors"] - ground["tensors"])
        per_sample_max = delta.max(axis=1)
        token_ids_equal = np.equal(actual["output_token_ids"], ground["output_token_ids"])
        allclose = np.allclose(actual["tensors"], ground["tensors"], rtol=0, atol=atol)
        summary["comparisons"][case] = {
            "allclose": bool(allclose),
            "bitwise_equal": bool(np.array_equal(actual["tensors"], ground["tensors"])),
            "max_abs_diff": float(delta.max()),
            "mean_abs_diff": float(delta.mean()),
            "per_sample_max_abs_diff": per_sample_max.tolist(),
            "output_token_ids": actual["output_token_ids"].tolist(),
            "output_token_ids_equal": token_ids_equal.tolist(),
        }
        assert token_ids_equal.all(), f"First output token mismatch for {case}."
        assert allclose, (
            f"First-token log-prob tensor mismatch for {case}: max_abs_diff={float(delta.max())}, atol={atol}."
        )
    summary["ground_truth_output_token_ids"] = ground["output_token_ids"].tolist()
    return summary


def run_validation(
    model_path: Path,
    longbench_path: Path,
    output_dir: Path,
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    prompt_tokens: int = _DEFAULT_PROMPT_TOKENS,
    atol: float = _DEFAULT_ATOL,
    prompt_lengths: list[int] | None = None,
    model_overrides: dict[str, Any] | None = None,
    max_model_len: int = _MAX_MODEL_LEN,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int = 1,
    global_segment_size: str = _DEFAULT_GLOBAL_SEGMENT_SIZE,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_overrides is None:
        model_overrides = dict(_DEFAULT_MODEL_OVERRIDES)
    effective_lengths = prompt_lengths or [prompt_tokens] * num_samples
    if len(effective_lengths) != num_samples:
        raise ValueError("prompt_lengths must contain one length per sample.")
    if any(length < _TRANSFER_BLOCK_SIZE for length in effective_lengths):
        raise ValueError(f"Every prompt length must be at least {_TRANSFER_BLOCK_SIZE} tokens.")
    if any(length >= max_model_len for length in effective_lengths):
        raise ValueError(f"Every prompt length must be less than max_model_len={max_model_len}.")
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be positive.")
    result_paths: dict[str, Path] = {}
    metrics: dict[str, dict[str, float]] = {}
    case_durations: dict[str, float] = {}
    for case in _CASES:
        case_start = time.monotonic()
        result_path, case_metrics = _run_case(
            case,
            model_path,
            longbench_path,
            output_dir,
            num_samples,
            prompt_tokens,
            prompt_lengths,
            model_overrides,
            max_model_len,
            max_num_batched_tokens,
            max_num_seqs,
            global_segment_size,
        )
        case_durations[case] = time.monotonic() - case_start
        result_paths[case] = result_path
        if case_metrics is not None:
            metrics[case] = case_metrics

    summary = _compare_cases(result_paths, atol)
    summary["model_overrides"] = model_overrides
    summary["transfer_granularity"] = _deepseek_transfer_granularity(model_overrides)
    summary["max_model_len"] = max_model_len
    summary["max_num_batched_tokens"] = max_num_batched_tokens
    summary["max_num_seqs"] = max_num_seqs
    summary["num_requests"] = num_samples
    summary["prompt_length_histogram"] = {
        str(length): count for length, count in sorted(Counter(effective_lengths).items())
    }
    summary["case_durations_seconds"] = case_durations
    summary["global_segment_size"] = global_segment_size
    summary["mooncake_metrics"] = metrics
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dsa,logprobs",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_deepseek_v4_mooncake_first_token_tensor_consistency(tmp_path: Path) -> None:
    model_path_value = os.getenv(_MODEL_PATH_ENV)
    longbench_path_value = os.getenv(_LONGBENCH_PATH_ENV)
    if not model_path_value or not longbench_path_value:
        pytest.skip(f"Set {_MODEL_PATH_ENV} and {_LONGBENCH_PATH_ENV} to run this hardware test.")
    run_validation(Path(model_path_value), Path(longbench_path_value), tmp_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run and compare all three validation cases.")
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--longbench-path", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--num-samples", type=int, default=_DEFAULT_NUM_SAMPLES)
    run_parser.add_argument("--prompt-tokens", type=int, default=_DEFAULT_PROMPT_TOKENS)
    run_parser.add_argument("--atol", type=float, default=_DEFAULT_ATOL)
    run_parser.add_argument(
        "--prompt-lengths",
        help="Comma-separated exact LongBench prompt lengths; overrides --num-samples.",
    )
    run_parser.add_argument(
        "--model-overrides-json",
        default=json.dumps(_DEFAULT_MODEL_OVERRIDES),
    )
    run_parser.add_argument("--max-model-len", type=int, default=_MAX_MODEL_LEN)
    run_parser.add_argument("--max-num-batched-tokens", type=int)
    run_parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum concurrent sequences used by each of the three cases.",
    )
    run_parser.add_argument(
        "--request-repetitions",
        type=int,
        default=1,
        help="Repeat the prompt-length profile with additional LongBench samples.",
    )
    run_parser.add_argument(
        "--global-segment-size",
        default=_DEFAULT_GLOBAL_SEGMENT_SIZE,
    )

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--case", choices=_CASES, required=True)
    worker_parser.add_argument("--model-path", required=True)
    worker_parser.add_argument("--longbench-path", required=True)
    worker_parser.add_argument("--output-path", required=True)
    worker_parser.add_argument("--num-samples", type=int, required=True)
    worker_parser.add_argument("--prompt-tokens", type=int, required=True)
    worker_parser.add_argument("--prompt-lengths-json")
    worker_parser.add_argument("--model-overrides-json")
    worker_parser.add_argument("--max-model-len", type=int, required=True)
    worker_parser.add_argument("--max-num-batched-tokens", type=int)
    worker_parser.add_argument("--max-num-seqs", type=int, required=True)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _run_worker(args)
        return
    if args.request_repetitions < 1:
        raise ValueError("request_repetitions must be positive.")
    prompt_lengths = [int(length) for length in args.prompt_lengths.split(",")] if args.prompt_lengths else None
    if prompt_lengths is not None:
        prompt_lengths *= args.request_repetitions
    elif args.request_repetitions > 1:
        prompt_lengths = [args.prompt_tokens] * (args.num_samples * args.request_repetitions)
    num_samples = len(prompt_lengths) if prompt_lengths is not None else args.num_samples
    summary = run_validation(
        args.model_path,
        args.longbench_path,
        args.output_dir,
        num_samples,
        args.prompt_tokens,
        args.atol,
        prompt_lengths,
        json.loads(args.model_overrides_json),
        args.max_model_len,
        args.max_num_batched_tokens,
        args.max_num_seqs,
        args.global_segment_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
