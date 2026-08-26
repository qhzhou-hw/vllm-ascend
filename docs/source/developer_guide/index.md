# Developer Guide

This section is for developers who want to contribute to vLLM Ascend or understand its internal architecture.

## Contribution

- **[Contribution Guide](contribution/index.md)** — How to contribute to vLLM Ascend
- **[Testing](contribution/testing.md)** — Write and run unit, E2E, and nightly tests
- **[Doc Writing](contribution/doc_writing.md)** — Documentation contribution guide
- **[Multi-Node Test](contribution/multi_node_test.md)** — Multi-node testing guide
- **[Nightly CI Test](contribution/nightly_ci_test.md)** — Nightly CI testing
- **[E2E CI Test](contribution/e2e_ci_test.md)** — E2E CI testing

## Design Documents

Explore the design documents covering patch architecture, CPU binding, model runner internals, disaggregated prefill, EPLB, ACL Graph, KV Cache Pool, custom operators, context parallel, quantization, and NPUGraph.

- **[HYPIC Ascend 适配说明](hypic_ascend_porting_zh.md)** — HYPIC 在 vLLM Ascend 上的架构、迁移步骤、验证方法和注意事项
- **[HYPIC Segment Cache 一致性修复](hypic_cache_consistency_fix_zh.md)** — Scheduler/worker LRU 分叉、权威驱逐协议和回归方法

## Evaluation

- **[Using EvalScope](evaluation/using_evalscope.md)** — Model evaluation with EvalScope
- **[Using lm_eval](evaluation/using_lm_eval.md)** — Model evaluation with lm_eval
- **[Using AISBench](evaluation/using_ais_bench.md)** — Model evaluation with AISBench
- **[Using OpenCompass](evaluation/using_opencompass.md)** — Model evaluation with OpenCompass
- **[HYPIC LongBench-E 测试指南](evaluation/hypic_longbench_zh.md)** — Full Recompute、HYPIC-512/1024 准确率测试和 SGLang 对比

## Performance and Debug

- **[Performance Benchmark](performance_and_debug/performance_benchmark.md)** — Benchmarking guide
- **[Optimization and Tuning](performance_and_debug/optimization_and_tuning.md)** — Performance optimization
- **[Service Profiling Guide](performance_and_debug/service_profiling_guide.md)** — Service profiling
- **[msprobe Guide](performance_and_debug/msprobe_guide.md)** — Debugging with msprobe
