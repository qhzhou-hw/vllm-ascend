# HYPIC Scheduler/Worker Segment Cache 一致性修复

本文记录 HYPIC 静态 segment pool 在 packed batch 和 LRU 槽满场景下的
scheduler/worker 状态分叉问题、根因、修复协议和回归方法。HYPIC 的整体 Ascend
适配设计见 [HYPIC 在 vLLM Ascend 上的适配说明](hypic_ascend_porting_zh.md)。

## 1. 问题背景

HYPIC 在两个进程侧维护同一份逻辑 cache 的不同视图：

| 位置 | 状态 | 用途 |
|---|---|---|
| Scheduler | `SegmentCatalog.ready: digest -> token_ids` | 判断 hit/miss，维护逻辑 LRU |
| Worker | `DeviceSegmentCache.segments: digest -> slot` | 将 segment 映射到模型构造期预分配的 NPU buffer |

worker 的 slot 中存放各 Attention/GDN 层对应 digest 的 tensor。只要两侧的成员集合或
LRU 顺序不同，scheduler 就可能把一个 digest 标成 hit，而 worker 已经把它的 slot
复用于其他 segment。旧实现会在下一次 hit 时抛出 `cache divergence`，但错误实际发生在
更早的 `reserve()` 驱逐阶段。

需要长期维持的核心不变量是：

```text
ordered(worker.segments) == ordered(scheduler.ready)
```

仅比较成员集合不够。两边顺序不同但成员相同时，下一次槽满分配仍会选择不同 victim。

## 2. 故障根因

### 2.1 Packed batch 中重复 cold miss 被 worker 去重

同一个 digest 可以在一个 packed forward 的多个请求中同时被判为 miss。这是合法情况，
因为 scheduler 只会在 model forward 成功后 commit，后一个请求规划时看不到本批前一个
请求尚未 commit 的 segment。

以容量 3 为例，两个请求依次包含 miss `[A, B]` 和 `[A, C]`：

```text
Scheduler commit 序列: A, B, A, C  -> LRU [B, A, C]
旧 Worker reserve 序列: A, B, C     -> LRU [A, B, C]
```

旧 worker 使用 `seen_misses` 去掉了第二个 `A`，也丢掉了该操作的 MRU refresh。下一次
加入 `D` 时，scheduler 驱逐 `B`，worker 却驱逐 `A`。之后 scheduler 将 `A` 标为 hit，
worker 查不到对应 slot，于是触发 delayed hard failure。

### 2.2 未真正调度的请求提前 touch 了 scheduler LRU

旧实现从 `KVCacheManager.get_computed_blocks()` 构造 plan 后立即调用
`SegmentCatalog.touch_plan()`。但 cache lookup 发生后，请求仍可能因为以下原因没有进入
当前 model forward：

- 剩余 `max_num_batched_tokens` 不足且 chunked prefill 已禁用；
- KV block 分配失败；
- encoder 或其他 scheduler admission 条件不满足。

这种请求不会出现在 `scheduled_new_reqs` 中，worker 也不会收到它的 plan。scheduler
单方面 touch hit 后，两边 LRU 顺序立即分叉。

### 2.3 `reserve()` 独立选择 victim，缺少 scheduler 授权

旧 `DeviceSegmentCache.reserve()` 在没有 free slot 时直接执行：

```python
_, slot = self.segments.popitem(last=False)
```

worker 没有收到 scheduler 期望驱逐的 digest，也没有在覆盖前验证两边选择是否一致。
因此任何更早的顺序分叉都会在这里转化成错误的 slot 复用。

### 2.4 为什么事后调用 `discard()` 不能修复

`DeviceSegmentCache.discard()` 在旧实现中没有生产调用。即使把 scheduler commit 返回的
evicted digest 传给 worker，事后 discard 也不能恢复已被错误复用的 slot：

```text
Scheduler 期望驱逐 B
Worker 已错误驱逐 A，并把 A 的 slot 分给 D
事后 discard(B) 只会再删除 B，A 和 B 都丢失
```

因此正确方案必须在 worker 覆盖 slot 之前校验 victim，而不是在 forward 后清理差异。

## 3. 修复后的同步协议

### 3.1 Cache lookup 只构造 plan，不修改 LRU

`_get_computed_blocks()` 现在只执行 token 分段和 hit/miss 规划，并把 plan 放到 request。
它不再 touch `SegmentCatalog`。

`Scheduler.schedule()` 返回确定的 `scheduled_new_reqs` 后，才按这些请求的确定顺序调用
`SegmentCatalog.prepare_scheduled_plans()`。未被接纳的 waiting request 不再影响 LRU。

### 3.2 Scheduler 生成 worker 可验证的 cache 操作

对真正进入 packed forward 的 plan，scheduler 执行三步：

1. 在每个 plan touch 之前记录 `cache_order_before`。
2. 按请求和 segment 顺序 touch 所有 hit。
3. 在 shadow LRU 中重放所有 miss，为每个 miss 写入 `expected_eviction`，并在最后一个
   plan 写入 `cache_order_after_commit`。

`expected_eviction` 可以是 digest 或 `None`：

- 有 free slot 或重复 miss 已存在时为 `None`；
- 槽满且需要复用时为 scheduler 选择的 LRU digest。

这使 scheduler 成为驱逐决策的权威来源，worker 不再无条件接受自己的本地选择。

### 3.3 Worker 在覆盖 slot 前逐层校验

`DeviceSegmentCache.prepare()` 按以下顺序工作：

1. 校验当前 worker LRU 等于该 plan 的 `cache_order_before`。
2. 按 plan 顺序 touch hit；scheduler 标记 hit 但 worker 缺少 slot 时立即失败。
3. 收集所有 miss，但不再去重，保留完整的 LRU 操作序列。
4. 调用 `reserve(digest, expected_eviction)`；在 `popitem()` 前比较本地 victim 和
   scheduler victim，不一致则不修改 cache 并抛出异常。
5. 完成 packed miss 后校验 worker 顺序等于 `cache_order_after_commit`。

ModelRunner 还会比较 `scheduled_new_reqs` 和 `input_batch.req_ids` 中的 HYPIC 请求顺序，
防止同一组 plan 被 worker 以不同请求顺序重放。

### 3.4 Model output 后验证 scheduler projection

model forward 完成后，scheduler 按原 plan 顺序执行真实 `SegmentCatalog.commit()`，并
验证：

- 实际 evicted digest 列表等于规划的 `expected_eviction` 列表；
- commit 后的完整 LRU 顺序等于 `cache_order_after_commit`。

该检查用于捕获 scheduler projection 与实际 commit 实现未来发生的语义漂移。

## 4. 状态时序

一次 HYPIC packed prefill 的 cache 生命周期如下：

```text
build plan (read-only)
        |
        v
scheduler confirms admission
        |
        +-- record pre-touch LRU
        +-- touch admitted hits
        +-- project miss victims/final LRU
        |
        v
worker validates request order and pre-touch LRU
        |
        +-- touch hits
        +-- replay every miss
        +-- validate victim before slot reuse
        +-- validate projected final LRU
        |
        v
model forward reads/writes static segment slots
        |
        v
scheduler commits misses and validates projection
```

在 model forward 执行期间，scheduler 暂时还没有 commit miss，而 worker 已经预留了
对应 slot。这是有意设计：scheduler 不会在当前 model output 返回前启动下一次 schedule，
并且 worker 的每个潜在 victim 已由 scheduler projection 明确授权。

为固化该时序，HYPIC 配置检查会强制设置 `async_scheduling=False`。vLLM 0.27 在执行器
支持时可能默认开启 async scheduling；如果不覆盖该默认值，下一轮 schedule 可能在上一轮
commit 前开始，无法满足上述 cache 协议。

## 5. 代码位置

| 文件 | 修改内容 |
|---|---|
| `vllm_ascend/hypic/cache.py` | admission metadata、victim projection、完整 miss replay 和一致性校验 |
| `vllm_ascend/patch/platform/patch_hypic.py` | 将 LRU touch 延迟到 admission 后，并校验 commit projection |
| `vllm_ascend/patch/worker/patch_hypic.py` | 校验 scheduler/worker 的请求顺序 |
| `tests/ut/hypic/test_hypic.py` | cache 顺序、victim 和 packed duplicate-miss 回归测试 |

`discard()` 仍保留为显式 scheduler invalidation 的低层能力，但正常 LRU 驱逐路径不调用
它，也不能用它修补一次已经发生的错误 slot 复用。

## 6. 回归测试

运行 HYPIC 单测：

```bash
pytest -q tests/ut/hypic/test_hypic.py -k device_cache
```

新增用例覆盖：

- scheduler/worker 成员相同但 LRU 顺序不同，在 reserve 前失败；
- scheduler 指定错误 victim 时，worker 在覆盖 slot 前失败且原 cache 不变；
- packed `[A, B] + [A, C]` 保留第二次 `A` 的 MRU refresh；
- 后续加入 `D` 时两侧均驱逐 `B`，再命中 `A` 不发生 divergence。

另外使用固定种子执行了 1000 轮纯 Python 随机状态机验证，覆盖多请求 packed hit、重复
miss、free-slot 分配和槽满驱逐；每轮 commit 后均满足：

```text
tuple(worker.segments) == tuple(scheduler.ready)
```

提交前完成了 Python 语法检查和 `git diff --check`。由于本地环境没有安装 vLLM Ascend
测试依赖，完整 pytest 和真实 NPU cold/warm 回归仍应在 Ascend 构建环境执行。

## 7. 后续维护要求

- 新增 cache 操作时必须同时定义 scheduler projection 和 worker replay 语义。
- 不得在 admission 确认前修改 scheduler LRU。
- 不得对影响 LRU 顺序的重复 hit/miss 操作做无依据去重。
- worker 复用静态 slot 前必须拿到并验证 scheduler-authoritative victim。
- 扩展 PP、DP 或异步 scheduler 前，需要重新评估“model output 前不会启动下一轮
  schedule”的时序假设；当前实现明确关闭 async scheduling。
- NPU 回归至少包含 packed batch、槽满、warm hit 和连续多轮请求，不能只验证 cold
  单请求输出。
