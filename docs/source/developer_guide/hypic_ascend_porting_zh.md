# HYPIC 在 vLLM Ascend 上的适配说明

本文总结将 HYPIC 从 SGLang Ascend 实现迁移到 vLLM Ascend 的设计、实现步骤、验证方法和容易出错的地方。当前实现以 Qwen3.5 hybrid-attention 模型为首个验证目标，使用 `transition_rope_recompute` 模式。

LongBench-E 的具体实验命令见 [HYPIC LongBench-E 测试指南](evaluation/hypic_longbench_zh.md)。面向使用者的简要配置见 [HYPIC 功能说明](../user_guide/feature_guide/hypic.md)。

## 1. 适配目标和当前边界

迁移目标不是让输出“看起来正常”，而是保持以下算法语义：

1. 按 tokenizer token ID 固定分段，并用完整 token 内容确认 hash 命中。
2. Full Attention 层缓存 segment-local RoPE key 和 value。
3. Gated DeltaNet（GDN）层缓存 causal-conv tail、零初态结束状态 `S` 和 transition `T`。
4. 按 prompt 顺序组合 recurrent state：`H_next = H_previous @ T + S`。
5. 命中 segment 时仅重算 seam，最后一个 segment 始终完整计算以产生 logits。
6. prefill 结束后，普通 vLLM decode 必须看到完整 prompt 的 Attention KV 和最终 GDN state。

当前限制：

- 支持 Qwen3.5 dense/MoE 的文本 hybrid-attention 路径。
- 仅支持 `transition_rope_recompute`。
- 支持 Tensor Parallel；当前要求 `max_num_seqs=1`。
- 不支持 PP、DP、PCP、DCP、speculative decoding 和 KV transfer/disaggregation。
- 要求 eager execution；HYPIC prefill 必须一次调度完整 prompt。
- cache 为进程内状态，engine 退出后清空。

## 2. vLLM Ascend 中的实现分层

| 层次 | 主要职责 | 主要文件 |
|---|---|---|
| 配置 | 解析和校验 `additional_config.hypic_config` | `vllm_ascend/hypic/config.py` |
| Planner | token 分段、hash、hit/miss 和稀疏 query position | `vllm_ascend/hypic/planner.py` |
| Scheduler patch | 构造 plan、传给 worker、维护 host LRU | `vllm_ascend/patch/platform/patch_hypic.py` |
| Model runner patch | 按绝对位置打包稀疏 token，建立 forward context | `vllm_ascend/patch/worker/patch_hypic.py` |
| Full Attention | suffix-causal、RoPE 位置转换、KV 保存与恢复 | `vllm_ascend/hypic/attention.py` |
| GDN | conv seam、`S/T` 生成、状态组合和 decode state 回写 | `vllm_ascend/hypic/gdn.py` |
| Cache | scheduler/device 两侧的 segment LRU | `vllm_ascend/hypic/cache.py` |
| Runtime | 将当前 plan/cache 安装到一次 model forward | `vllm_ascend/hypic/runtime.py` |

该实现通过 vLLM Ascend patch 层接入上游 vLLM，避免复制整个 Scheduler、ModelRunner 或 Qwen 模型文件。HYPIC 未启用时，patch 直接调用原方法。

## 3. 适配流程

### 3.1 先固定算法合同

开始替换 NPU 算子前，先把 SGLang 实现拆成与框架无关的状态合同：

- segment 的 token 范围、hash 和生命周期；
- hit、miss、首段、末段以及 seam 的行为；
- Attention public KV 的位置坐标系；
- GDN `S/T` 的逻辑 layout 和组合顺序；
- prefill 到 decode 的最终状态交接。

如果一开始只对照最终文本，很难区分 planner、Attention、GDN、RoPE、paged cache 或 decode state 中的错误。

### 3.2 用 opt-in 配置隔离新路径

HYPIC 通过 `additional_config` 开启：

```python
from vllm import LLM

llm = LLM(
    model="/path/to/Qwen3.5-35B-A3B",
    tensor_parallel_size=2,
    enforce_eager=True,
    max_num_seqs=1,
    max_model_len=45056,
    max_num_batched_tokens=45056,
    additional_config={
        "hypic_config": {
            "enabled": True,
            "mode": "transition_rope_recompute",
            "chunk_size": 512,
            "seam_sink_tokens": 8,
            "max_cache_segments": 128,
        }
    },
)
```

配置检查同时完成以下约束：

- 强制 eager；
- 禁用 scheduler chunked prefill；
- 将 `max_num_seqs` 设为 1；
- 让 `max_num_scheduled_tokens` 等于 `max_num_batched_tokens`，避免 hybrid 模型默认的 2048-token 调度上限拆开 HYPIC prompt；
- 保持 vLLM prefix caching 开启，以满足 hybrid cache 的 page-size 校验，但 HYPIC 会绕过标准 prefix hit 并独立维护 segment cache。

Full Recompute 基线应显式设置 `enable_prefix_caching=False`，不要与 HYPIC engine 复用。

### 3.3 在 scheduler 侧建立可序列化 plan

planner 对 token ID 执行以下步骤：

1. 按 `chunk_size` 切成半开区间。
2. 对每段计算 128-bit digest。
3. digest 命中后继续比较完整 token tuple，防止截断 hash 冲突。
4. 非末段可缓存；末段永远 miss，保证产生正确的末 token hidden state/logits。
5. 首个 hit segment 可以完全跳过；后续 hit segment 重算 `seam_sink_tokens`。
6. 输出有序且不重复的绝对 `query_positions`。

plan 只包含 Python 标量、list 和 dict，可以安全地从 scheduler 进程传递给 worker。worker 不应重新进行独立 hit/miss 判断，否则 scheduler 和 device cache 很容易分叉。

### 3.4 将 vLLM 的连续前缀语义改造成稀疏绝对位置

这是迁移中最关键的框架差异。

标准 vLLM 的 `num_computed_tokens` 表示从位置 0 开始的一段连续前缀；HYPIC 跳过的是多个不连续 segment。Scheduler 仍使用 `num_computed_tokens` 减少本次调度 token 数，但 ModelRunner 必须用 plan 中的绝对 `query_positions` 重新打包：

- 从原 prompt token row 选择本次真正执行的 token；
- `positions` 写入原始绝对位置，而不是 `0..N-1`；
- 用绝对位置重新计算 paged cache `slot_mapping`。

只替换 `input_ids` 而沿用标准 slot mapping 会把稀疏 token 写到错误 KV slot。这个问题在 prefill 阶段可能不报错，但 decode 会读取错位 KV，常表现为 cold 输出正常、warm 输出异常或摘要逐渐漂移。

### 3.5 Full Attention：位置无关缓存与 suffix-causal

同一 segment 可能出现在 prompt 的不同绝对位置，所以缓存的 key 不能永久绑定第一次请求的 RoPE 位置。

写 cache 时：

```text
absolute-position key -> rerotate -> segment-local-position key
```

读 cache 时：

```text
segment-local-position key -> rerotate -> current absolute-position key
```

当前实现要求 NeoX-style RoPE，并从 vLLM 的 `cos_sin_cache` 计算位置差旋转。Qwen 的 cache 每个 rotary pair 只存一份频率，扩展到 NeoX 两个半向量时不能遗漏重复。

每个 query segment 访问“前序 KV + 当前 segment KV”，因此经常有 `Q_len < KV_len`。正确的 causal 原点位于矩形 mask 的右下角：

```text
absolute_query_position(i) = KV_len - Q_len + i
```

Ascend `npu_prompt_flash_attention` 必须使用 right-down causal，即 `sparse_mode=3`。upper-left causal 会静默屏蔽本应可见的历史 KV。

### 3.6 命中后仍要恢复标准 paged KV

HYPIC custom prefill 可以直接读取自己的 segment cache，但紧接着的 decode 仍走普通 vLLM Attention backend。被跳过的 segment 如果只存在于 HYPIC cache、没有写回标准 paged KV，首个 decode token 就会缺少完整上下文。

因此每个 hit segment 都必须：

1. 恢复到当前绝对 RoPE 位置；
2. 根据 block table 和绝对 token position 计算物理 slot；
3. 通过 `reshape_and_cache` 写入标准 key/value cache。

这一步与稀疏 token 的 slot mapping 修复是两个不同问题，缺一不可。

### 3.7 GDN：生成并组合 `S/T`

每个可缓存 GDN segment 保存：

- `conv_tail`：causal conv 所需的最后 `K-1` 个输入；
- `zero_state`：零初态运行该 segment 后得到的 `S`；
- `transition`：该 segment 对任意输入状态的线性变换 `T`。

组合公式为：

```text
H_next = H_previous @ T + S
```

实现用零初态运行一次得到 `S`，再用单位状态和零 value 运行一次得到 `T`。组合使用 FP32，最终再转换为 vLLM recurrent cache 的 dtype。

`sgl_kernel_npu` 的公开 GDN kernel 使用 `[N,H,K,V]` state，而 HYPIC 逻辑状态是 `[N,H,V,K]`。进入和离开 kernel 必须显式 `transpose(-1, -2)`。Qwen3.5 中 `K == V == 128`，漏转置不会触发 shape error，只会产生错误数值，因此必须用非对称 reference case 验证。

### 3.8 Cache 和 LRU

Scheduler 保存 digest 到完整 token tuple 的 host catalog；每个 worker 保存逐层 NPU tensor。二者使用同一容量和访问顺序维护 LRU。

注意：vLLM Ascend 的 `max_cache_segments` 同时限制 Full Attention public KV 和 GDN state 的 segment 数。它和 SGLang 的 `max_mamba_cache_size` 都以 segment/state slot 为主要概念，但占用的物理内存不等价，不能直接根据同一个数值推断显存一定可用。

一旦 scheduler 判定 hit 而 worker 找不到对应 layer state，必须立即报 cache divergence，不能静默回退；静默回退会让当前 plan 的 token 数和 worker 实际需要的 token 数不一致。

## 4. Ascend 环境搭建

以下为本次实机验证组合，不代表唯一兼容组合：

| 组件 | 验证值 |
|---|---|
| CANN | 9.0.0 |
| Python | 3.12 |
| vLLM | 0.26.1 开发版本 |
| sgl-kernel-npu | 2026.5.1 |
| NPU | 2 卡，TP=2 |

推荐在 Ascend 机器上直接构建，不要用本地 CPU 环境代替 NPU 验证：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3.12 -m venv /data/venvs/vllm-hypic
source /data/venvs/vllm-hypic/bin/activate
python -m pip install -U pip wheel setuptools

python -m pip install -e /data/vllm
python -m pip install -e /data/vllm-ascend
python -m pip install sgl-kernel-npu==2026.5.1

export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH=/data/vllm:/data/vllm-ascend:${PYTHONPATH:-}
```

版本必须遵循驱动、CANN、PyTorch、torch_npu 和 Triton-Ascend 的兼容矩阵。不要只升级其中一个包。

安装后至少检查：

```bash
python - <<'PY'
import torch
import torch_npu
import sgl_kernel_npu
import vllm
import vllm_ascend

print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
print("vLLM:", vllm.__version__)
PY
```

## 5. 验证顺序

不要直接从完整 LongBench 开始。推荐按以下层级定位问题：

### 5.1 纯逻辑单测

```bash
pytest -q tests/ut/hypic/test_hypic.py
```

覆盖配置校验、cold/warm plan、末段规则、hash collision guard、LRU、right-down causal、RoPE rerotation 和 transition compose 顺序。

### 5.2 单请求 full baseline

先禁用 HYPIC，确认模型、chat template、temperature 0 和 max output length 正常。Full 输出都异常时，不应继续排查 HYPIC cache。

### 5.3 cold/warm 重复请求

对同一个长 prompt 连续请求两次，至少检查：

- cold 输出与 Full Recompute 接近；
- warm 确实走 hit plan；
- QA 短答案通常应稳定；
- 摘要可能因 BF16 和近似状态重建出现措辞差异，但不能空输出、乱码或明显失去上下文；
- prefill 后 decode 不得在第一个 token 立刻分叉成无关文本。

### 5.4 多条固定样本

用 Qasper、GovReport、HotpotQA、MultiNews 各抽至少一条，同时覆盖短答案和 512-token 摘要。先比较单样本官方 metric，再比较文本；跨运行时 greedy decoding 不保证 bitwise 相同。

### 5.5 完整 LongBench-E

采样闸门通过后再运行 Full Recompute、HYPIC-512 和 HYPIC-1024。完整流程见配套 LongBench 文档。

## 6. 值得注意的坑

### 6.1 稀疏位置不能冒充连续前缀

`num_computed_tokens` 只是让 scheduler 少调度 token，不足以表达 HYPIC。worker 必须重新选择绝对 token、position 和 slot mapping。

### 6.2 shape 正确不代表 GDN state 正确

`K`、`V` 都是 128 时，错误 layout 可以运行完整模型。必须验证 transition compose 数值和最终 decode state。

### 6.3 right-down causal 与普通 causal 不同

只在 `Q_len == KV_len` 上测试无法暴露该错误。单测必须包含 `Q_len < KV_len`。

### 6.4 HYPIC cache 和 vLLM paged cache 都要维护

前者服务于后续 prefill reuse，后者服务于当前请求 decode。只更新其中一个会产生阶段性正确、随后错误的现象。

### 6.5 最后一段不能命中

最后一段负责产生请求末尾 logits。把它当作普通 hit 全跳过，会导致没有可用于采样的最后 hidden state。

### 6.6 prefix caching 的开关语义不同

HYPIC 模式内部保留 vLLM prefix caching 配置以通过 hybrid-cache 校验，但标准 computed-block hit 已被 HYPIC planner 接管。Full Recompute 基线则必须关闭 prefix caching。

### 6.7 显存要为按需 segment cache 留余量

提高 `gpu_memory_utilization` 会增大 vLLM KV cache，却压缩 HYPIC public KV/GDN tensor 的动态空间。长测使用 256 segments 时，本次验证把 HYPIC 的 `gpu_memory_utilization` 降到 0.70；默认 128 segments 可从 0.85 起测。具体值应通过最长请求和 cache 填满后的 NPU 峰值确定。

### 6.8 当前只能单序列 HYPIC

`max_num_seqs=1` 是正确性约束，不只是保守性能参数。Full Recompute 可以 batch=4；HYPIC 不能直接把 runner batch 调到 4。若参考 SGLang 使用 rolling B4，必须在报告中注明 cache admission 顺序不同。

### 6.9 断点续跑会改变 HYPIC cache 历史

结果 JSONL 可以按 index 跳过已完成样本，但 engine 重启后 process-local HYPIC cache 为空。继续运行的预测仍有效，却不再复现原来从 index 0 连续运行的命中历史。严格准确率对比应从该 HYPIC mode 的第一条重新运行，或在恢复时重放已完成 prompt 以重建 cache。

### 6.10 `cached_tokens` 当前不是可靠统计项

现有 offline LongBench runner 的 `cached_tokens` 字段填 0，并未从 HYPIC plan 汇总真实命中 token。准确率可以使用该 runner；若要报告命中率或性能收益，应另加 scheduler plan 统计，不能直接使用该字段。

## 7. Review 和回归检查表

- [ ] HYPIC 关闭时所有 patch 调回原实现。
- [ ] planner 使用 token ID，不使用字符长度分段。
- [ ] hash 命中后比较完整 token tuple。
- [ ] 末段不缓存、不命中。
- [ ] 稀疏 input、absolute position 和 slot mapping 一致。
- [ ] public key 在 local/absolute RoPE 坐标之间正确转换。
- [ ] suffix attention 使用 right-down causal。
- [ ] hit segment 写回标准 paged KV。
- [ ] GDN kernel 边界执行 state layout 转换。
- [ ] transition 按 prompt 顺序用 FP32 组合。
- [ ] 最终 conv/recurrent state 写入 decode slot。
- [ ] scheduler 和 device LRU 不发生 divergence。
- [ ] 真实 NPU 上完成 cold/warm、四数据集采样和 LongBench-E。
