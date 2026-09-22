"""PIC control-plane regressions; executable without vLLM, torch or an NPU.

Standalone: python tests/ut/hypic/test_pic_control.py
The standalone launcher skips only the package's vLLM logging initializer;
the planner, protocol and cache modules under test are the production modules.
"""

from __future__ import annotations

import pickle
import random
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

if __name__ == "__main__":
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
    sys.modules["vllm_ascend"] = package

from vllm_ascend.hypic.config import HypicConfig
from vllm_ascend.hypic.pic_cache import PicCatalog, PicDeviceCache
from vllm_ascend.hypic.planner import build_plan, validate_plan
from vllm_ascend.hypic.protocol import PicAcknowledgement, request_policy


class PicControlTests(unittest.TestCase):
    def setUp(self):
        self.config = HypicConfig(chunk_size=4, seam_sink_tokens=1)
        self.catalog = PicCatalog(3, epoch="test-engine")
        self.worker = PicDeviceCache(3)

    def plan(self, tokens=None, **kwargs):
        return build_plan(tokens or list(range(13)), self.catalog.ready, self.config, **kwargs)

    def run_step(self, plans):
        prepared, step = self.catalog.prepare(plans)
        # The control messages must survive an actual serialization roundtrip.
        prepared, step = pickle.loads(pickle.dumps((prepared, step)))
        self.worker.prepare(prepared, step)
        self.worker.mark_layer("attention:0")
        self.worker.mark_layer("gdn:1")
        ack = self.worker.acknowledge(completed_ranks=2, expected_layers={"attention:0", "gdn:1"})
        self.catalog.commit(pickle.loads(pickle.dumps(ack)), expected_ranks=2)
        self.assertEqual({key: entry.ref for key, entry in self.catalog.entries.items()}, self.worker.by_key)
        return prepared, step

    def test_cold_warm_and_logical_query_accounting(self):
        self.run_step({"r": self.plan()})
        warm = self.plan()
        self.assertEqual(warm["query_positions"], [4, 8, 12])
        self.assertEqual(warm["logical_advance"], 13)
        self.assertEqual(warm["num_query_tokens"], 3)
        self.assertEqual(warm["num_restore_tokens"], 12)
        self.run_step({"r2": warm})

    def test_middle_tools_reordered_hit_but_head_role_changes_miss(self):
        boundaries = [0, 2, 4, 6, 7]
        self.run_step({"a": self.plan([1, 2, 3, 4, 5, 6, 99], segment_boundaries=boundaries)})
        reordered = self.plan([1, 2, 5, 6, 3, 4, 98], segment_boundaries=boundaries)
        self.assertEqual([s["hit"] for s in reordered["segments"]], [True, True, True, False])
        moved_head = self.plan([3, 4, 1, 2, 5, 6, 98], segment_boundaries=boundaries)
        self.assertEqual([s["hit"] for s in moved_head["segments"]], [False, False, True, False])

    def test_zero_seam_allows_head_role_reuse_and_single_query(self):
        self.config = replace(self.config, seam_sink_tokens=0)
        self.run_step({"a": self.plan([1, 2, 3, 4, 99], segment_boundaries=[0, 2, 4, 5])})
        warm = self.plan([3, 4, 1, 2, 98], segment_boundaries=[0, 2, 4, 5])
        self.assertEqual(warm["query_positions"], [4])
        self.run_step({"b": warm})

    def test_cache_salt_isolation(self):
        self.run_step({"a": self.plan(cache_salt="tenant-a")})
        self.assertTrue(self.plan(cache_salt="tenant-a")["segments"][0]["hit"])
        self.assertFalse(any(s["hit"] for s in self.plan(cache_salt="tenant-b")["segments"]))
        self.assertFalse(any(s["hit"] for s in self.plan()["segments"]))

    def test_short_segments_do_not_create_zero_length_interiors(self):
        plan = self.plan([1, 2, 3, 4], segment_boundaries=[0, 1, 2, 3, 4])
        self.assertEqual([s["cacheable"] for s in plan["segments"]], [True, False, False, False])
        self.assertEqual([s["recompute_seam"] for s in plan["segments"]], [0, 0, 0, 0])
        self.run_step({"r": plan})

    def test_capacity_overflow_uses_compute_only(self):
        prepared, step = self.run_step({"r": self.plan(list(range(29)))})
        self.assertEqual(len(step.fills), 3)
        self.assertEqual(sum(s["store"] for s in prepared["r"]["segments"]), 3)
        self.assertEqual(prepared["r"]["query_positions"], list(range(29)))

    def test_pinned_hits_cannot_be_evicted_for_misses(self):
        self.run_step({"a": self.plan()})
        before = self.catalog.entries.copy()
        prepared, step = self.run_step({"hits": self.plan(), "misses": self.plan(list(range(100, 113)))})
        self.assertEqual(step.fills, ())
        self.assertEqual(self.catalog.entries, before)
        self.assertFalse(any(s["store"] for s in prepared["misses"]["segments"]))

    def test_duplicate_miss_has_one_writer_even_with_worker_reorder(self):
        prepared, step = self.catalog.prepare({"first": self.plan(), "second": self.plan()})
        self.assertEqual(sum(s["store"] for s in prepared["first"]["segments"]), 3)
        self.assertFalse(any(s["store"] for s in prepared["second"]["segments"]))
        self.worker.prepare(dict(reversed(list(prepared.items()))), step)

    def test_shared_input_plan_has_independent_request_reservations(self):
        original = self.plan()
        before = deepcopy(original)
        prepared, step = self.run_step({"first": original, "second": original})
        self.assertEqual(original, before)
        self.assertIsNot(prepared["first"], prepared["second"])
        self.assertEqual(sum(s["store"] for s in prepared["first"]["segments"]), len(step.fills))
        self.assertFalse(any(s["store"] for s in prepared["second"]["segments"]))

    def test_lookup_and_unadmitted_plans_do_not_touch_lru(self):
        self.run_step({"a": self.plan()})
        before = tuple(self.catalog.entries)
        self.plan()
        self.plan(list(range(400, 417)))
        self.assertEqual(tuple(self.catalog.entries), before)

    def test_filling_not_ready_and_previous_step_blocks_schedule(self):
        plans, step = self.catalog.prepare({"r": self.plan()})
        self.assertEqual(self.catalog.ready, {})
        with self.assertRaisesRegex(RuntimeError, "previous acknowledgement"):
            self.catalog.prepare({"r2": self.plan()})
        self.worker.prepare(plans, step)
        with self.assertRaisesRegex(RuntimeError, "incomplete layer"):
            self.worker.acknowledge(completed_ranks=2, expected_layers={"attention:0"})

    def test_wrong_victim_is_rejected_without_mirror_mutation(self):
        self.run_step({"r": self.plan()})
        prepared, step = self.catalog.prepare({"new": self.plan(list(range(100, 113)))})
        before = self.worker.slots.copy()
        bad_fill = replace(step.fills[0], victim=None)
        bad_step = replace(step, fills=(bad_fill, *step.fills[1:]))
        with self.assertRaisesRegex(RuntimeError, "victim divergence"):
            self.worker.prepare(prepared, bad_step)
        self.assertEqual(before, self.worker.slots)
        self.assertIsNone(self.worker.pending)

    def test_invalid_plan_generation_is_rejected_atomically(self):
        prepared, step = self.catalog.prepare({"r": self.plan()})
        prepared["r"]["segments"][0]["slot_generation"] += 1
        with self.assertRaisesRegex(RuntimeError, "invalid slot"):
            self.worker.prepare(prepared, step)
        self.assertEqual(self.worker.slots, {})

    def test_duplicate_writer_is_rejected(self):
        prepared, step = self.catalog.prepare({"a": self.plan(), "b": self.plan()})
        prepared["b"]["segments"][0] = deepcopy(prepared["a"]["segments"][0])
        with self.assertRaisesRegex(RuntimeError, "unique writer"):
            self.worker.prepare(prepared, step)

    def test_partial_and_stale_ack_do_not_publish(self):
        _, step = self.catalog.prepare({"r": self.plan()})
        ack = PicAcknowledgement(step.epoch, step.step_id, tuple(f.target for f in step.fills), 2)
        for bad in (
            replace(ack, epoch="other"),
            replace(ack, step_id=0),
            replace(ack, completed_ranks=1),
            replace(ack, fills=()),
        ):
            with self.assertRaises(RuntimeError):
                self.catalog.commit(bad, expected_ranks=2)
            self.assertEqual(self.catalog.ready, {})
        self.catalog.commit(ack, expected_ranks=2)

    def test_abort_after_dispatch_never_resurrects_victims(self):
        self.run_step({"r": self.plan()})
        _, step = self.catalog.prepare({"new": self.plan(list(range(100, 113)))})
        victims = {fill.victim.key for fill in step.fills}
        self.assertTrue(victims.isdisjoint(self.catalog.ready))
        self.catalog.abort(dispatched=True)
        self.assertTrue(victims.isdisjoint(self.catalog.ready))
        with self.assertRaisesRegex(RuntimeError, "restart"):
            self.catalog.prepare({"retry": self.plan()})

    def test_abort_before_dispatch_preserves_old_payloads(self):
        self.run_step({"r": self.plan()})
        before = self.catalog.entries.copy()
        self.catalog.prepare({"new": self.plan(list(range(100, 113)))})
        self.catalog.abort(dispatched=False)
        self.assertEqual(self.catalog.entries, before)
        self.run_step({"retry": self.plan()})

    def test_engine_epoch_change_requires_worker_restart(self):
        self.run_step({"a": self.plan()})
        other = PicCatalog(3, epoch="new-engine")
        prepared, step = other.prepare({"b": build_plan(list(range(13)), {}, self.config)})
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            self.worker.prepare(prepared, step)

    def test_replayed_step_rejected(self):
        prepared, step = self.run_step({"r": self.plan()})
        with self.assertRaisesRegex(RuntimeError, "stale step"):
            self.worker.prepare(prepared, step)

    def test_hash_collision_does_not_overwrite_cached_payload(self):
        with patch("vllm_ascend.hypic.planner.segment_hash", return_value="collision"):
            self.run_step({"a": self.plan([1, 2, 3, 4, 99])})
            collision = self.plan([5, 6, 7, 8, 98])
            self.assertFalse(collision["segments"][0]["hit"])
            prepared, step = self.run_step({"b": collision})
            self.assertFalse(prepared["b"]["segments"][0]["store"])
            self.assertEqual(step.fills, ())

    def test_invalid_query_map_and_empty_prompt_rejected(self):
        plan = self.plan()
        plan["query_positions"].pop()
        with self.assertRaises(ValueError):
            validate_plan(plan)
        with self.assertRaises(ValueError):
            validate_plan(build_plan([], {}, self.config))

    def test_request_policy_defaults_and_validation(self):
        self.assertEqual(request_policy(SimpleNamespace()), "pic")
        for policy in ("pic", "full_recompute", "prefix_only"):
            request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"hypic_cache_policy": policy}))
            self.assertEqual(request_policy(request), policy)
        with self.assertRaises(ValueError):
            request_policy(SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"hypic_cache_policy": "bad"})))

    def test_random_packed_eviction_and_capacity_stability(self):
        rng = random.Random(20260916)
        for batch in range(500):
            plans = {}
            for row in range(rng.randint(1, 5)):
                tokens = []
                for _ in range(rng.randint(1, 6)):
                    base = rng.randrange(8) * 4
                    tokens.extend(range(base, base + 4))
                tokens.append(1000 + batch)
                plans[str(row)] = self.plan(tokens)
            self.run_step(plans)
            self.assertLessEqual(len(self.catalog.entries), self.catalog.capacity)
            self.assertIsNone(self.worker.pending)


if __name__ == "__main__":
    unittest.main()
