"""V1 adapter contract tests; NPU execution is tested separately.

On an installed vLLM: pytest tests/ut/hypic/test_pic_adapter.py
Without dependencies, exercise the reference dataclass schema and production
hook bodies: python tests/ut/hypic/test_pic_adapter.py --schema-root <vllm repo>
This schema-only mode does not claim to test vLLM execution or its RPC runtime.
"""

from __future__ import annotations

# Schema-only execution bootstraps the vLLM modules before production imports.
# ruff: noqa: E402
import ast
import inspect
import pickle
import sys
import unittest
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def load_definitions(path, module, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), module.__dict__)


SCHEMA_ONLY = "--schema-root" in sys.argv
if SCHEMA_ONLY:
    index = sys.argv.index("--schema-root")
    source_root = Path(sys.argv[index + 1])
    del sys.argv[index : index + 2]
    repo_root = Path(__file__).resolve().parents[3]
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(repo_root / "vllm_ascend")]
    sys.modules["vllm_ascend"] = package
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    for name, relative, definitions in (
        (
            "vllm.v1.core.sched.output",
            "vllm/v1/core/sched/output.py",
            {"SchedulerOutput", "NewRequestData", "CachedRequestData"},
        ),
        ("vllm.v1.outputs", "vllm/v1/outputs.py", {"ModelRunnerOutput"}),
    ):
        module = ModuleType(name)
        module.__dict__.update(dataclass=dataclass, field=field, cached_property=cached_property)
        sys.modules[name] = module
        load_definitions(source_root / relative, module, definitions)

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.hypic.config import HypicConfig, get_hypic_config
from vllm_ascend.hypic.pic_cache import PicCatalog
from vllm_ascend.hypic.planner import build_plan
from vllm_ascend.hypic.protocol import PicAcknowledgement, request_policy
from vllm_ascend.hypic.vllm_adapter import (
    PicSchedulerOutput,
    acknowledge_output,
    logical_scheduler_output,
    pack_scheduler_output,
    worker_input_output,
)

if SCHEMA_ONLY:
    hooks = ModuleType("pic_test_hooks")
    hooks.__dict__.update(
        logger=Mock(),
        PicCatalog=PicCatalog,
        get_hypic_config=get_hypic_config,
        PicSchedulerOutput=PicSchedulerOutput,
        build_plan=build_plan,
        request_policy=request_policy,
        pack_scheduler_output=pack_scheduler_output,
        logical_scheduler_output=logical_scheduler_output,
        PauseState=Enum("PauseState", "UNPAUSED PAUSED_NEW"),
    )
    load_definitions(
        repo_root / "vllm_ascend/patch/platform/patch_hypic.py",
        hooks,
        {
            "_get_computed_blocks",
            "_allocate_slots",
            "_schedule",
            "_update_from_output",
            "_scheduler_init",
            "_mamba_block_aligned_split",
        },
    )
else:
    from vllm_ascend.patch.platform import patch_hypic as hooks


class PicAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = HypicConfig(chunk_size=4, seam_sink_tokens=1)
        self.catalog = PicCatalog(3, epoch="test")
        self.scheduler = SimpleNamespace(
            hypic_catalog=self.catalog,
            hypic_config=self.config,
            hypic_batch_kind=None,
            hypic_batch_units=0,
            max_num_scheduled_tokens=32,
            running=[],
            _pause_state=hooks.PauseState.UNPAUSED,
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2)),
        )
        self.manager = SimpleNamespace(hypic_scheduler=self.scheduler, empty_kv_cache_blocks=())

    def request(self, policy="pic", tokens=None, request_id="r"):
        tokens = tokens or list(range(13))
        return SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=tokens,
            num_tokens=len(tokens),
            num_preemptions=0,
            num_computed_tokens=0,
            num_in_flight_tokens=0,
            sampling_params=SimpleNamespace(extra_args={"hypic_cache_policy": policy}),
            cache_salt="tenant",
            hypic_plan=None,
        )

    def output(self, request):
        data = NewRequestData(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([2, 3, 4, 5],),
            num_computed_tokens=0,
            lora_request=None,
        )
        return SchedulerOutput(
            scheduled_new_reqs=[data],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={request.request_id: request.num_tokens},
            total_num_scheduled_tokens=request.num_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    def warm_plan(self, request):
        cold = build_plan(request.prompt_token_ids, {}, self.config, cache_salt=request.cache_salt)
        _, step = self.catalog.prepare({request.request_id: cold})
        ack = PicAcknowledgement(step.epoch, step.step_id, tuple(f.target for f in step.fills), 2)
        self.catalog.commit(ack, expected_ranks=2)
        return build_plan(request.prompt_token_ids, self.catalog.ready, self.config, cache_salt=request.cache_salt)

    def test_sparse_lookup_does_not_claim_contiguous_prefix(self):
        req = self.request()
        self.warm_plan(req)
        blocks, computed, external = hooks._get_computed_blocks(self.manager, req)
        self.assertEqual((blocks, computed, external), ((), 0, 0))
        self.assertEqual(req.hypic_plan["num_query_tokens"], 3)
        self.assertEqual(req.num_computed_tokens, 0)

    def test_worker_suffix_metadata_view_does_not_change_scheduler(self):
        req = self.request()
        warm = self.warm_plan(req)
        prepared, step = self.catalog.prepare({"r": warm})
        native = self.output(req)
        output = pickle.loads(pickle.dumps(pack_scheduler_output(native, prepared, step)))
        self.assertIsInstance(output, PicSchedulerOutput)
        self.assertEqual(output.num_scheduled_tokens, {"r": 3})
        self.assertEqual(native.num_scheduled_tokens, {"r": 13})
        worker = worker_input_output(output)
        self.assertEqual(worker.scheduled_new_reqs[0].num_computed_tokens, 10)
        self.assertEqual(output.scheduled_new_reqs[0].num_computed_tokens, 0)
        self.assertEqual(logical_scheduler_output(output).num_scheduled_tokens, {"r": 13})
        self.assertEqual(output.scheduled_new_reqs[0].block_ids, ([2, 3, 4, 5],))

    def test_completion_restores_logical_counts_and_requires_ack(self):
        req = self.request()
        prepared, step = self.catalog.prepare({"r": self.warm_plan(req)})
        output = pack_scheduler_output(self.output(req), prepared, step)
        ack = PicAcknowledgement(step.epoch, step.step_id, tuple(f.target for f in step.fills), 2)
        result = acknowledge_output(ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0}), ack)
        result = pickle.loads(pickle.dumps(result))
        original = Mock(return_value="done")
        with patch.object(hooks, "_ORIGINAL_UPDATE_FROM_OUTPUT", original, create=True):
            self.assertEqual(hooks._update_from_output(self.scheduler, output, result), "done")
        self.assertEqual(original.call_args.args[1].num_scheduled_tokens, {"r": 13})
        self.assertIsNone(self.catalog.pending)

    def test_missing_ack_never_publishes(self):
        req = self.request()
        prepared, step = self.catalog.prepare({"r": build_plan(req.prompt_token_ids, {}, self.config)})
        output = pack_scheduler_output(self.output(req), prepared, step)
        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            hooks._update_from_output(self.scheduler, output, SimpleNamespace())
        self.assertEqual(self.catalog.ready, {})

    def test_replay_and_full_recompute_bypass_pic_and_apc(self):
        for policy, preemptions in (("pic", 1), ("full_recompute", 0)):
            req = self.request(policy)
            req.num_preemptions = preemptions
            self.assertEqual(hooks._get_computed_blocks(self.manager, req), ((), 0, 0))
            self.assertIsNone(req.hypic_plan)

    def test_prefix_only_uses_native_lookup(self):
        original = Mock(return_value=("prefix-blocks", 8, 0))
        with patch.object(hooks, "_ORIGINAL_GET_COMPUTED_BLOCKS", original, create=True):
            self.assertEqual(
                hooks._get_computed_blocks(self.manager, self.request("prefix_only")), ("prefix-blocks", 8, 0)
            )

    def test_publication_guard_protects_decode_and_direct_coordinator_calls(self):
        native_publish = Mock()
        coordinator = SimpleNamespace(cache_blocks=native_publish)
        fake = SimpleNamespace(
            vllm_config=SimpleNamespace(additional_config={"hypic_config": {"enabled": True}}),
            kv_cache_manager=SimpleNamespace(coordinator=coordinator),
        )
        with patch.object(hooks, "_ORIGINAL_SCHEDULER_INIT", Mock(), create=True):
            hooks._scheduler_init(fake)
        for policy in ("pic", "full_recompute", "prefix_only"):
            coordinator.cache_blocks(self.request(policy), 100)
        self.assertEqual(native_publish.call_count, 1)
        self.assertEqual(request_policy(native_publish.call_args.args[0]), "prefix_only")

    def test_mixed_prefill_admission_is_deferred_without_allocating(self):
        def allocate(manager, request, num_new_tokens, delay_cache_blocks=False):
            return (num_new_tokens, delay_cache_blocks)

        with (
            patch.object(hooks, "_ORIGINAL_ALLOCATE_SLOTS", allocate, create=True),
            patch.object(hooks, "_ALLOCATE_SIGNATURE", inspect.signature(allocate), create=True),
        ):
            self.assertEqual(hooks._allocate_slots(self.manager, self.request(), 13), (13, True))
            self.assertIsNone(hooks._allocate_slots(self.manager, self.request("prefix_only"), 13))

    def test_workspace_budget_defers_second_request_without_mutating_cache(self):
        req = self.request()
        hooks._get_computed_blocks(self.manager, req)
        self.scheduler.hypic_config = replace(self.config, max_prefill_units=req.hypic_plan["num_prefill_units"])

        def allocate(manager, request, num_new_tokens, delay_cache_blocks=False):
            return ()

        with (
            patch.object(hooks, "_ORIGINAL_ALLOCATE_SLOTS", allocate, create=True),
            patch.object(hooks, "_ALLOCATE_SIGNATURE", inspect.signature(allocate), create=True),
        ):
            self.assertEqual(hooks._allocate_slots(self.manager, req, 13), ())
            self.assertIsNone(hooks._allocate_slots(self.manager, req, 13))
        self.assertEqual(self.catalog.ready, {})

    def test_single_request_over_budget_fails_instead_of_waiting_forever(self):
        self.scheduler.max_num_scheduled_tokens = 10
        with self.assertRaisesRegex(ValueError, "atomic admission"):
            hooks._get_computed_blocks(self.manager, self.request())
        self.scheduler.max_num_scheduled_tokens = 32
        self.scheduler.hypic_config = replace(self.config, max_prefill_units=1)
        with self.assertRaisesRegex(ValueError, "GDN units"):
            hooks._get_computed_blocks(self.manager, self.request())

    def test_schedule_uses_only_admitted_plans(self):
        req = self.request()
        self.warm_plan(req)
        hooks._get_computed_blocks(self.manager, req)
        ignored = self.request(tokens=list(range(100, 113)), request_id="ignored")
        hooks._get_computed_blocks(self.manager, ignored)
        self.scheduler.requests = {"r": req, "ignored": ignored}
        with patch.object(hooks, "_ORIGINAL_SCHEDULE", Mock(return_value=self.output(req)), create=True):
            output = hooks._schedule(self.scheduler)
        self.assertEqual(set(output.pic_plans), {"r"})
        self.assertEqual(output.pic_step.fills, ())
        self.assertEqual(output.num_scheduled_tokens, {"r": 3})

    def test_adapter_preserves_unrelated_plugin_metadata(self):
        req = self.request()
        prepared, step = self.catalog.prepare({"r": self.warm_plan(req)})
        native = self.output(req)
        native.disable_profiling_timing = True
        packed = pack_scheduler_output(native, prepared, step)
        self.assertTrue(packed.disable_profiling_timing)
        self.assertTrue(worker_input_output(packed).disable_profiling_timing)
        self.assertTrue(logical_scheduler_output(packed).disable_profiling_timing)
        output = ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0})
        output.execution_time_ms = 42
        ack = PicAcknowledgement(step.epoch, step.step_id, (), 2)
        self.assertEqual(acknowledge_output(output, ack).execution_time_ms, 42)

    def test_partial_admission_is_rejected_before_sparse_execution(self):
        req = self.request()
        prepared, step = self.catalog.prepare({"r": self.warm_plan(req)})
        output = self.output(req)
        output.num_scheduled_tokens["r"] -= 1
        output.total_num_scheduled_tokens -= 1
        with self.assertRaisesRegex(RuntimeError, "atomic full-prompt"):
            pack_scheduler_output(output, prepared, step)


if __name__ == "__main__":
    unittest.main()
