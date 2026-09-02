# DeepSeek-V4 AscendStore 传输放大分析与优化

本文记录 DeepSeek-V4（DSV4）混合 KV cache 在 `AscendStoreConnector` 中的
读写放大问题、已合入的压缩 page 裁剪方案、32K 请求的数据量闭环，以及仍未处理的
compressor state 冗余。

对应实现提交：

```text
2601f3be5 fix(kv_pool): trim partial compressed cache pages
```

## 1. 问题范围

DSV4 同时包含 c4、c128、SWA 和 compressor state。不同 cache entry 的一个物理 page
覆盖的原始 token 数不同，但旧实现统一使用物理分配跨度作为 Mooncake value 长度。

需要区分三个概念：

- `hash_block_size`：前缀 hash 的生成粒度；
- group/store block：AscendStore 一次生成 key 的逻辑 token 粒度；
- physical page：算子实际分配和寻址的 cache page。

三者不相等时，物理 page 中可能只有一段连续前缀属于当前 store block。把完整物理区域
传给 Mooncake 会保存未写入或暂时无效的数据。

## 2. DSV4 Cache 数据分类

非 A5、默认 BF16 KV cache 路径中的主要数据如下：

| 逻辑数据 | 运行时 cache | 格式 | 作用 |
| --- | --- | --- | --- |
| `cmp_kv_c4` | c4 `self_attn.attn` | BF16[512] | c4 压缩完成的 token 状态 |
| `ind_cmp_kv_c4` | `indexer.k_cache` | INT8[128] + FP16 scale | c4 Indexer 压缩状态 |
| `cmp_kv_c128` | c128 `self_attn.attn` | BF16[512] | c128 压缩完成的 token 状态 |
| `chunk_kv` | `swa_cache` | BF16[512] | 最近窗口的原始 token 状态 |
| c4 main state | `compressor.state_cache` | FP32[2048] | c4 KV/score 中间状态 |
| c4 indexer state | `indexer.compressor.state_cache` | FP32[512] | Indexer KV/score 中间状态 |
| c128 main state | `compressor.state_cache` | FP32[1024] | c128 KV/score 中间状态 |

其中 c128 main state 的最后一维由两个等宽部分组成：

```text
FP32[1024]
├── chunk_kv_state:    FP32[512]
└── chunk_score_state: FP32[512]
```

## 3. 旧实现的放大根因

旧实现通过 cache tensor 第一维与统一 block 数的比值计算一个 store block 的长度：

```text
block_size_scale = tensor_num_blocks / num_blocks
physical_block_len = page_content_bytes * block_size_scale
```

这个长度适合作为相邻 block 的物理寻址跨度，但不一定等于当前 store block 的有效数据
长度。

以原部署中的 c128 entry 为例：

```text
physical page content = 128 KiB
physical page coverage = 16384 raw tokens
store block = 2048 raw tokens
```

当前 store block 的有效连续前缀只有：

```text
128 KiB * 2048 / 16384 = 16 KiB
```

旧逻辑还会根据 block 数聚合为 1 MiB 物理跨度，因此单 entry 实际传输了：

```text
1 MiB / 16 KiB = 64x
```

`cache_family=c128` 只影响 key 命名空间，不参与 value 长度计算，不能消除该放大。

## 4. 已实现的修复

修复将每个 entry 的几何信息拆成两个独立概念：

```text
block_stride：完整物理跨度，用于 block_id 寻址
block_len：当前 store block 的有效连续字节数，用于 Mooncake GET/PUT
```

对于 `compress_ratio > 1` 且一个逻辑 page 覆盖范围大于 group block 的 dense compressed
MLA entry，使用 cache spec 计算：

```text
transfer_block_len =
    page_content_bytes * group_block_tokens / entry_logical_page_tokens
```

仅在以下条件全部满足时启用裁剪：

- `compress_ratio` 和逻辑 page token 数有效；
- 逻辑 page 覆盖范围大于 group block；
- group block 与 compression ratio 对齐；
- 结果为整数字节；
- 结果大于 0 且不超过物理 block 长度。

任一条件不满足时保守回退到原始物理长度。

SWA 和 compressor state 不是简单的 dense per-token page，本次修改不对它们套用上述
公式。它们继续使用 cache manager 的 reachability 和原物理几何。

## 5. Layerwise 与 Non-Layerwise

两条路径最终都使用同一组 `group_block_len` 和 `group_block_stride`：

- non-layerwise 在一个 key 中组合某 group 的多层 entry；
- layerwise 按层生成 key，只组合当前层的 entry。

因此修复不会改变总有效字节数，只改变 key 的拆分方式。layerwise 的 key 更多、单 key
更小；non-layerwise 的 key 更少、单 key 更大。

## 6. Key 布局兼容性

修复改变了 Mooncake value 的字节长度。旧实例写入的 value 不能由新实例按新长度安全
加载，因此 key 增加：

```text
value_layout:2
```

PoolKey、LayerPoolKey、scheduler 和直接 key 快速路径使用相同版本。升级后旧对象自然
miss，不需要把旧 value 误解释为新布局。

## 7. 32K 剪裁模型数据量闭环

验证配置使用 43 层 DSV4 cache 结构：

```text
2 dense layers
21 c4 layers
20 c128 layers
4096-token store block
8 store boundaries for a 32768-token prefix
```

各类数据量如下：

| 数据 | 计算 | 大小 |
| --- | --- | ---: |
| `cmp_kv_c4` | 21 * 8 * 32 * 32768 B | 168 MiB |
| c4 Indexer K | 21 * 8 * 32 * 4096 B | 21 MiB |
| c4 Indexer scale | 21 * 8 * 32 * 64 B | 0.328125 MiB |
| 全部 SWA | 43 * 8 * 4 * 32768 B | 43 MiB |
| c4 main state | 21 * 8 * 4 * 16384 B | 10.5 MiB |
| c4 Indexer state | 21 * 8 * 4 * 4096 B | 2.625 MiB |
| `cmp_kv_c128` | 20 * 8 * 32768 B | 5 MiB |
| c128 main state | 20 * 8 * 16 * 32768 B | 80 MiB |
| **合计** |  | **330.453125 MiB** |

字节合计：

```text
346505216 B = 330.453125 MiB
```

Mooncake master 的内存增量与该理论值一致，说明修复后实际传输长度与 cache tensor 的
有效布局闭环。

## 8. 尚未处理：C128 Compressor State

本次提交只修复 dense compressed page 的有效前缀长度，没有消除 c128 compressor
state。

c128 state 每个 slot 为：

```text
chunk_kv_state:    FP32[512] = 2 KiB
chunk_score_state: FP32[512] = 2 KiB
total:                           4 KiB
```

当前通用 SlidingWindow manager 在每个 4096-token store 边界保留完整 128 个 slot：

```text
128 * 4 KiB = 512 KiB per c128 layer per boundary
20 * 512 KiB = 10 MiB per boundary
8 * 10 MiB = 80 MiB per 32K request
```

因为 `4096 % 128 == 0`，这些外部命中边界没有未完成的非重叠 c128 compression group，
理论上不需要恢复 c128 state。但不能在 `AscendStoreConnector` 中硬编码 DSV4/c128 层名
特判，也不能全局禁止该 group 的 KV transfer：PD disaggregation 可能在非 128 对齐的
prompt 尾部交接，此时 partial state 是必要的。

后续合理方向是由 cache spec 描述 compressor state 的通用恢复语义，例如：

```text
compression_period
overlap_tokens
required state range at a restore boundary
```

cache manager 根据实际边界生成 store/load mask，connector 只执行通用 mask，不感知模型
名称。当边界与 period 对齐且 overlap 为 0 时，该 state group 可以自动视为不需要外部
value。

若未来安全消除当前 80 MiB c128 state，32K 理论写入量将变为：

```text
330.453125 MiB - 80 MiB = 250.453125 MiB
```

该数字仅为后续优化目标，不代表当前提交已经实现。

## 9. 验证与回归测试

当前单元测试覆盖：

- c128 partial compressed page 从 1 MiB 物理区域裁剪为 16 KiB 有效 value；
- c4 full page 和 compressor state 保持原长度；
- non-layerwise `prepare_value` 使用裁剪后的长度；
- layerwise `prepare_value_layer` 使用相同长度；
- 所有 key 路径携带 `value_layout:2`。

NPU 验证除检查 Mooncake 内存增量外，还需要比较：

- full recompute；
- non-layerwise Mooncake load；
- layerwise Mooncake load。

相同请求的首 token tensor 应保持一致，并确认第二轮确实发生 Mooncake GET，而不是命中
残留 HBM cache。

## 10. 相关文件

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py`
- `tests/ut/distributed/ascend_store/test_pool_worker.py`
- `tests/ut/distributed/ascend_store/test_metadata.py`

