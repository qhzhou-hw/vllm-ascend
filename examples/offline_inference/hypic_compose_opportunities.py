"""Mock-only diagnostics for state-composition overhead, not a new backend.

Compare the validated allocation-inclusive API with a prepared launch and an
optional static NPU graph. Prepared/graph timings are opportunity bounds: they
reuse metadata, output storage and addresses, unlike arbitrary live requests.
"""

import argparse
import json
import statistics
from functools import partial

import hypic_compose_bench as bench
import hypic_cube_batch as candidate
import torch
import torch_npu  # noqa: F401


class CaptureLaunch:
    """Capture one launch; the caller separately retains all pointed-to owners."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.record = None

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.record = (grid, args, kwargs)
            return self.kernel[grid](*args, **kwargs)

        return run


def check(result, expected):
    replay, final = result
    for row, (expected_replay, expected_final) in enumerate(expected):
        torch.testing.assert_close(replay[row], expected_replay, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(final[row], expected_final, atol=2e-5, rtol=2e-4)


def poison(result):
    for tensor in [*result[0], result[1]]:
        tensor.fill_(float("nan"))


def check_updated_packet(graph, packet, capture, owners, inputs, identity):
    """Change S/T addresses, order and ragged lengths without recapturing.

    Keep one request at the original maximum length so packet strides match.
    Avoid empty requests: their privately allocated sentinel needs a separate
    persistent owner, which this isolated capture diagnostic does not provide.
    """
    from vllm_ascend.hypic.compose import ComposeStep, compose_reference

    zeros, transforms, pool, pool_t, steps = inputs
    next_pool = torch.randn_like(pool) * 0.02
    next_pool_t = identity * 0.85 + torch.randn_like(pool_t) * 0.003
    next_s = [torch.randn_like(s) * 0.03 for s in zeros]
    next_t = [identity * 0.85 + torch.randn_like(t) * 0.003 for t in transforms]
    next_steps = [[*reversed(steps[0][:-1]), ComposeStep(0, 0)]] + [[ComposeStep(0, 0)] for _ in steps[1:]]
    next_inputs = (next_s, next_t, next_pool, next_pool_t, next_steps)
    expected = [compose_reference(s, t, next_pool, next_pool_t, seq) for s, t, seq in zip(next_s, next_t, next_steps)]
    next_capture = CaptureLaunch(capture.kernel)
    candidate._output_chain = next_capture
    try:
        next_owners = candidate.batch_compose(*next_inputs, pack_mode="output")
    finally:
        candidate._output_chain = capture.kernel
    next_packet = next_capture.record[1][0]
    assert packet.shape == next_packet.shape
    # final is a captured kernel argument; replay addresses live in the packet.
    graph_outputs = (next_owners[0], owners[1])
    poison(graph_outputs)
    packet.copy_(next_packet)
    graph.replay()
    torch.npu.synchronize()
    check(graph_outputs, expected)


def run_case(args, batch, segments):
    from vllm_ascend.hypic import compose_triton
    from vllm_ascend.hypic.compose import ComposeStep, compose_fused_batch, compose_reference

    torch.manual_seed(args.seed)
    device = "npu:0"
    shape = (segments, args.heads, 128, 128)
    identity = torch.eye(128, device=device)[None, None]
    pool = torch.randn(shape, device=device) * 0.02
    pool_t = identity * 0.9 + torch.randn_like(pool) * (0.03 / 128**0.5)
    zeros = [torch.randn((1, args.heads, 128, 128), device=device) * 0.02 for _ in range(batch)]
    transforms = [identity * 0.9 + torch.randn_like(s) * (0.03 / 128**0.5) for s in zeros]
    steps = [
        [ComposeStep(-((index * 3 + row) % segments) - 1) for index in range(max(1, segments - row))]
        + [ComposeStep(0, 0)]
        for row in range(batch)
    ]
    inputs = (zeros, transforms, pool, pool_t, steps)
    expected = [compose_reference(s, t, pool, pool_t, seq) for s, t, seq in zip(zeros, transforms, steps)]
    original_launch = compose_triton.launch_compose
    compose_triton.launch_compose = partial(candidate.batch_compose, pack_mode="output")
    try:
        full_api = lambda: compose_fused_batch(*inputs)
        raw_api = lambda: candidate.batch_compose(*inputs, pack_mode="output")

        def validate_only():
            launch = compose_triton.launch_compose
            try:
                compose_triton.launch_compose = lambda *_: None
                full_api()
            finally:
                compose_triton.launch_compose = launch

        capture = CaptureLaunch(candidate._output_chain)
        candidate._output_chain = capture
        try:
            owners = full_api()
        finally:
            candidate._output_chain = capture.kernel
        check(owners, expected)
        grid, kernel_args, kernel_kwargs = capture.record
        prepared = lambda: capture.kernel[grid](*kernel_args, **kernel_kwargs)
        prepared()
        check(owners, expected)
        # Packet readback is deliberately OUTSIDE timing. Metadata-upload timing
        # includes CPU tensor construction and H2D, but not building this list.
        packet = kernel_args[0]
        payload = packet.cpu().tolist()
        upload = lambda: torch.tensor(payload, dtype=torch.int64, device=device)
        functions = {
            "full_api": full_api,
            "validate_only": validate_only,
            "raw_api": raw_api,
            "prepared": prepared,
            "metadata_upload": upload,
        }
        graph = None
        if args.graph:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    prepared()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                prepared()
            poison(owners)
            graph.replay()
            torch.npu.synchronize()
            check(owners, expected)
            functions["static_graph"] = graph.replay
        for fn in functions.values():
            for _ in range(3):
                fn()
        timings = bench.timed_variants(functions, args.iterations, args.repeats)
        stream_spans = {}
        for name in ("prepared", "static_graph"):
            if name not in functions:
                continue
            spans = []
            for _ in range(10):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                functions[name]()
                end.record()
                torch.npu.synchronize()
                spans.append(start.elapsed_time(end))
            stream_spans[name] = statistics.median(spans)
        check(full_api(), expected)
        prepared()
        check(owners, expected)
        if graph is not None:
            poison(owners)
            graph.replay()
            torch.npu.synchronize()
            check(owners, expected)
            check_updated_packet(graph, packet, capture, owners, inputs, identity)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "batch": batch,
                    "documents": segments,
                    "heads": args.heads,
                    "case": "warm-ragged",
                    "seed": args.seed,
                    "host_inclusive_ms": timings,
                    "stream_span_ms_includes_dispatch_gaps": stream_spans,
                    "packet_bytes": packet.numel() * packet.element_size(),
                    "prepared_reuses_metadata_and_outputs": True,
                    "graph_reuses_addresses_and_metadata": graph is not None,
                    "graph_updated_packet_verified": graph is not None,
                    "not_production_speedup": True,
                }
            ),
            flush=True,
        )
    finally:
        compose_triton.launch_compose = original_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--segments", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if min(*args.batch_sizes, *args.segments, args.heads, args.iterations, args.repeats) <= 0:
        parser.error("sizes and iteration counts must be positive")
    torch.npu.set_device(0)
    bench.bootstrap()
    for batch in args.batch_sizes:
        for segments in args.segments:
            run_case(args, batch, segments)


if __name__ == "__main__":
    main()
