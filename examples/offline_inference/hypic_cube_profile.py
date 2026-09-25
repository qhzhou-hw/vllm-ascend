"""NPU-event stage timings for the experimental batched Cube implementation."""

import argparse
import json
import statistics

import hypic_compose_bench as bench
import hypic_cube_batch as candidate
import torch
import torch_npu  # noqa: F401


class TimedKernel:
    def __init__(self, kernel, samples):
        self.kernel, self.samples = kernel, samples

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            result = self.kernel[grid](*args, **kwargs)
            end.record()
            self.samples.append((start, end))
            return result

        return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--segments", type=int, default=64)
    args = parser.parse_args()
    bench.bootstrap()
    from vllm_ascend.hypic.compose import ComposeStep

    torch.npu.set_device(0)
    pool = torch.randn((args.segments, 8, 128, 128), device="npu") * 0.02
    transitions = torch.eye(128, device="npu")[None, None].repeat(args.segments, 8, 1, 1)
    s, t = [pool[:1]] * args.batch, [transitions[:1]] * args.batch
    steps = [
        [ComposeStep(-i - 1) for i in range(max(1, args.segments - r))] + [ComposeStep(0, 0)] for r in range(args.batch)
    ]
    call = lambda: candidate.batch_compose(s, t, pool, transitions, steps, pack_mode="gather")
    for _ in range(3):
        call()
    torch.npu.synchronize()
    samples = {}
    for name in ("_gather", "_chain", "_scatter"):
        samples[name] = []
        setattr(candidate, name, TimedKernel(getattr(candidate, name), samples[name]))
    for _ in range(10):
        call()
    torch.npu.synchronize()
    print(
        json.dumps(
            {
                "batch": args.batch,
                "segments": args.segments,
                "device_stage_ms": {
                    name: statistics.median(start.elapsed_time(end) for start, end in values)
                    for name, values in samples.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
