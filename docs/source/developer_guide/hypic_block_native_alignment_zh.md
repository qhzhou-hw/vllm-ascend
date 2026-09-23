# HYPIC 与 MindSpeed-MM block-native PIC 的推理语义对齐

## 1. 两种模式的范围

默认 `mode="transition_rope_recompute"` 保留既有 HYPIC 算法。新增
`mode="block_native_pic"` 对齐 MindSpeed-MM 的
`mindspeed_mm/fsdp/models/qwen3_5/block_native_pic.py`，不将旧实验结果视为新模式的验证。

| 行为 | 默认 HYPIC | Block-native PIC |
| --- | --- | --- |
| Conv history | 默认跨段传递，可用 `reset_conv_history=True` 单独关闭 | 每个 Document 和 Query 均从零开始 |
| Document GDN 输出 | 从前段累计状态 replay | 从零状态计算，不注入前段状态 |
| GDN S/T | 按请求顺序组合 | 同样组合，但只用于 Query 初始状态和最终 decode 状态 |
| Document attention | 可以访问前面的段 | 仅访问自身段内的因果前缀 |
| Query attention | 完整因果前缀 | 所有 Document，加上 Query 自身的因果前缀 |
| Hit Document | 可重算 seam | 完整复用，不重算 seam |
| 超长 Document | 可继续按 chunk 切分 | 报错，禁止改变训练段边界 |
| Full recompute | 普通全序列因果推理 | 相同 block-native 图，只禁用缓存读写 |

`reset_conv_history` 本身只控制卷积，不会改变 GDN replay 或 attention。
在 block-native 模式中卷积重置是强制语义，无论该布尔配置是否填写。

### 如何选择模式

以下配置均位于 `additional_config["hypic_config"]`；首先设置 `enabled=True`。

| 目的 | `mode` | `reset_conv_history` | `seam_sink_tokens` |
| --- | --- | --- | --- |
| 保持原有 HYPIC 行为 | `transition_rope_recompute`（默认） | `False`（默认） | 保持原实验值，默认 8 |
| 仅关闭 HYPIC 跨段卷积历史 | `transition_rope_recompute` | `True` | 独立配置；0 表示不重算 seam |
| 对齐当前 MindSpeed-MM block-native 训练图 | `block_native_pic` | 有效值强制为 `True` | 必须显式为 0 |

第三种配置不是第二种配置的别名：它还改变 Document 的 GDN 初始状态和 attention 可见性。
默认 HYPIC 的 Document miss 输出仍使用累计状态 replay，并非训练图中完全独立的 Document 输出。
“保留 HYPIC”指本次修改没有改变默认模式的既有算法路径，不意味着原有近似缓存复用与
任意新前缀下的 full recompute 严格等价。已有旧模式准确率也不能用来证明新模式准确率。

## 2. 配置与输入契约

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/block-native-trained-Qwen3.5",
    tensor_parallel_size=2,
    enforce_eager=True,
    max_num_seqs=4,
    max_num_batched_tokens=45056,
    additional_config={
        "hypic_config": {
            "enabled": True,
            "mode": "block_native_pic",
            "chunk_size": 512,
            "seam_sink_tokens": 0,
            "max_cache_segments": 96,
        }
    },
)

# token_ids 必须是与训练一致的、已经移除 PIC 分隔符的最终 token 序列。
# 边界只是示例：Document 0=[0,128)，Document 1=[128,384)，Query=[384,600)。
boundaries = [0, 128, 384, 600]
params = SamplingParams(
    temperature=0,
    max_tokens=128,
    extra_args={
        "hypic_cache_policy": "pic",
        "hypic_segment_boundaries": boundaries,
    },
)
# outputs = llm.generate([{"prompt_token_ids": token_ids}], params)
```

输入必须满足：

- 边界是递增、不重复的整数 token offsets，包含 0 和 prompt 总长度。
- 至少一个非空 Document 和一个非空 Query，最后一段是 Query。
- System prompt、工具描述等应按训练时的归属划入 Document 或 Query，不能仅按字符串名称猜测。
- Query 包含当前用户问题及 assistant generation prefix；生成的回答延续同一 Query，而不是新 segment。
- 使用与训练相同的 tokenizer/chat template；PIC separator token IDs 先移除再计算 offsets。
  不能将训练中的结构性分隔符作为普通文本留在推理输入中。
- `chunk_size` 在此模式仅约束每个缓存 Document 的最大长度，不触发自动切段。
  超限需增大 slot 大小并重新评估显存，或使用真正按新段边界训练的模型。
- 最后的 Query 不写持久化 segment pool，因此可以长于 `chunk_size`，但仍受 prompt/token budget 限制。
- 非零 seam 明确报错；`prefix_only` 和抢占后的 native replay 当前不支持。

旧 HYPIC 模式的调用方式、默认 seam=8 和自动 chunk 切分保留。

## 3. 实现要点

### Conv

启用 history 重置（或使用 block-native 模式）时，在每个 segment 进入 `_causal_conv`
前选择零 history buffer。未启用时保留上一段 tail 传递。不能直接对上一轮的
`history` 调用 `zero_()`，因为它可能引用 `hypic_conv_pool[slot]`，会破坏仍被引用的缓存。
两种模式都保留 tail 保存逻辑；block-native 不把前段 tail 注入后段。
最后 Query 计算出的 tail 正常写入 native conv cache，decode 时继续使用，而不是每 token 清零。

tail 是最后 `K-1` 个卷积前的原始 QKV 向量，不是 GDN recurrent state，也不是卷积输出。
旧模式默认会将 hit 段保存的 tail 交给下一个 miss 段、hit seam 或 Query。
重置模式则在下一段开始时忽略该 tail；它不会清空缓存槽本身，也不会禁用 tail 保存。
有 seam 时只在 segment 起点重置一次，seam 与该段 interior 之间不额外重置。
旧模式自动切出的子段也各自构成重置边界；block-native 禁止自动拆分训练 Document。

### GDN

本节以下描述 block-native 计算图；旧 HYPIC 的累计状态 replay 保持不变。

Document 的 zero state `S` 和 transition `T` 仍逐段计算并缓存。请求累计状态满足：

```text
H = 0
for document in request_order:
    H = H @ T_document + S_document
Query_initial_state = H
```

Document 的输出 replay 使用零 initial state；Query 的输出 replay 才使用 `H`。
计算中仍会把 Query 自身的 S/T 作用于 `H`，得到写入 native decode cache 的最终状态。
训练使用 `[H,K,V]` 布局和左乘，推理使用 `[H,V,K]` 和右乘；两者通过转置对应，不能机械照抄乘法顺序。

### Attention / KV

未命中的 Document 只用自身 KV 做 causal attention；Query 使用全部 Document KV 和自身 KV。
命中的 Document 跳过全部 query tokens，将其 key 从段内 RoPE 位置转换到当前请求位置，并恢复
native paged KV，以供 Query 和后续 decode 使用。RoPE 重定位不改变原有绝对位置编号策略。

### 调度与缓存

plan 明确携带 `mode` 和有效 `reset_conv_history`。cache key 纳入二者，普通 HYPIC、
只清零 conv 的 HYPIC、block-native PIC 不能互相命中。
控制协议升级为 v3，旧 plan/旧 worker 不兼容，升级时必须停止并重启所有 engine worker，重新预热。

block-native 下 `full_recompute` 仍建立 block-native plan，但传入空 ready 集合，且将全部
segment 标记为不可缓存，确保既不读也不发布 PIC；native APC 同样不参与。
这条路径与 `pic` 采用同一模型计算图，允许公平比较复用前后的输出。
旧 HYPIC 下的 `full_recompute` 和 `prefix_only` 行为不变。

因此，对比训练对齐模式的无缓存基线时，应保持 `mode="block_native_pic"` 和相同边界，
只将请求的 `hypic_cache_policy` 改为 `"full_recompute"`。不要通过关闭 `enabled`
或切回旧模式获得基线，否则 Document attention 和 GDN 计算图都会改变。
旧模式的原生 `full_recompute` 不应用 `reset_conv_history`；这个开关作用于 HYPIC 分段执行路径。

## 4. 验证与当前状态

新增/扩展的测试：

- `tests/ut/hypic/test_hypic.py`：conv 单独开关、缓存键隔离、cold/mixed/warm、seam 0/1、
  短段 tail 补零、compute-only、多请求 native state 槽。
- `tests/ut/hypic/test_block_native_pic.py`：显式边界校验、超长 Query 不被切段、模式缓存隔离、
  GDN 和 attention 的 cold/warm、段重排、替换前缀、KV 恢复、Query 的最终状态。
- 同一测试文件还直接加载相邻 `MindSpeed-MM` checkout 的训练原语，与推理路径比较 conv/GDN，
  并核对 attention mask。没有训练源码时该项明确 skip，不冒充通过。
- `tests/ut/hypic/test_pic_adapter.py`：对齐图的 full recompute、无缓存读写、原子 admission、
  拒绝 native prefix/preemption fallback。

在服务器安装好配套依赖后执行：

```bash
cd /data/vllm-ascend
python -m pytest -q tests/ut/hypic
```

直接训练原语对比要求训练仓库位于同级 `/data/MindSpeed-MM`。张量测试使用真实 PyTorch
卷积和 attention，但以小型递归参考实现替代 NPU GDN kernel；它们验证计算图，不替代算子或模型验证。

实机仍需依次检查：

1. 相同训练 checkpoint、token IDs、边界、dtype、位置编码和确定性生成参数。
2. 对比训练原语与 NPU GDN kernel 的逐层输出、S/T 及 Query 最终状态。
3. 同一个 engine 中先跑 `full_recompute`，再跑 `pic` cold/warm；对比 logits、首 token 和生成结果。
4. 重排 Document、替换前缀、重复工具、batch size 1/4，确认 warm 与当前请求的 full recompute 对齐。
5. 显存不足导致 compute-only 时保持同一计算语义；decode 检查完整 paged KV 和 Query state 延续。

### 2026-09-23：Ascend mock 权重验证

使用用户提供的新 Ascend 环境完成服务器验证，没有下载模型，也没有在本地运行测试。
环境为 Ascend 910B2、Python 3.11、PyTorch/torch-npu 2.10.0、CANN 9.0.0，
隔离 venv 使用 `sgl-kernel-npu` 2026.9.0（官方 2026.9.0.post5 发布包）和
`triton-ascend` 3.2.2。Ruff 使用仓库配置指定的 0.14.0。

| 验证 | 结果 | 范围 |
| --- | --- | --- |
| 张量与训练原语单测 | 53 passed | 包括旧模式、conv reset、block-native；训练原语对比未跳过 |
| PIC 控制面 | 23 passed | 生产 planner/protocol/cache |
| 调度适配 schema 测试 | 16 passed | vLLM 源码 dataclass + 生产 hook，不等于完整 engine 集成 |
| 真实 NPU 算子，Document 长度 64 | 10/10 passed | 随机 BF16 权重与输入 |
| 真实 NPU 算子，Document 长度 129 | 10/10 passed | 覆盖非对齐段长 |
| 真实 NPU 算子，Document 长度 512 | 10/10 passed | 目标 chunk 大小 |

每组 NPU 验证包含：GDN batch size 1/4 的 cold、warm、Document 重排、替换前缀
（新前缀 compute-only）八项，以及真实 CANN attention cold/warm 两项。
GDN 直接执行生产 `forward_hypic_gdn`，与 MindSpeed-MM 的 segmented conv 和
block-native GDN 原语比较；两侧调用同一个真实 NPU GDN kernel。
另以不经过 S/T 组合的逐段 recurrence 检查最终 SSM state，并检查 Query conv tail、
attention 段内/全局可见性和 hit KV 恢复。

跨三种段长汇总的最大绝对差为：GDN 输出不超过 `9.537e-6`，最终 SSM state 不超过
`1.206e-4`，CANN attention 对 FP32-score 参考不超过 `9.766e-4`。
这些是固定随机种子、当前随机权重尺度下的观测值，不是对所有模型的误差保证。
NPU attention 用 identity RoPE；非零旋转及段重排由张量单测覆盖。
收紧断言后已复跑三组：GDN 输出 `atol=1e-5, rtol=0.02`，最终 SSM state
`atol=2e-4, rtol=0.02`，attention `atol=0.002, rtol=0.02`，均通过。

复现入口：

```bash
# 两个源码 checkout 应同级：vllm-ascend/ 与 MindSpeed-MM/。
cd /data/hypic-mock-validation-20260923/vllm-ascend
source /data/hypic-mock-venv/bin/activate
python examples/offline_inference/hypic_mock_validate.py --unit-tests
python tests/ut/hypic/test_pic_control.py
python tests/ut/hypic/test_pic_adapter.py --schema-root ../vllm
python examples/offline_inference/hypic_mock_validate.py --npu --doc-len 64
python examples/offline_inference/hypic_mock_validate.py --npu --doc-len 129
python examples/offline_inference/hypic_mock_validate.py --npu --doc-len 512
```

服务器日志保存在上述工作目录的父目录：`unit-tests.log`、`control-tests.log`、
`adapter-tests.log`、`npu64.log`、`npu129.log`、`npu512.log`。
测试入口只绕过顶层 vLLM 包初始化，并提供 eager gating 与真实 NPU KV scatter 的接口桥接；
它没有替换被测 HYPIC GDN/attention 算法，也不会在缺少 NPU GDN kernel 时偷偷退回参考实现。

**验证边界：算子/计算语义与缓存数据路径已验证；未启动完整 vLLM engine，未加载完整
Qwen3.5 随机模型，未验证完整模型 logits、生成结果、真实 checkpoint 准确率或吞吐。**
训练原语对比复用了同一 NPU GDN kernel，因此不单独证明该 kernel 与训练所用 GPU kernel
逐位一致。上面的完整 engine / 模型验证清单仍需后续执行。

## 5. 确认结论与上线边界

- 当前确认的是与同级 MindSpeed-MM checkout 的 block-native 训练原语在核心计算语义上对齐，
  不是与普通 causal 训练、任意训练分支或任意 checkpoint 自动兼容。
- 默认 HYPIC 计算路径保留；可选 history 重置仅改变卷积边界，不替代 block-native 模式。
- 对齐依赖相同的最终 token IDs、Document/Query 边界、模型参数和位置编码配置；
  serving 不会自动识别或删除训练分隔符。多模态位置编码、量化和完整模型多卡执行未在本次验证中覆盖。
- 将 mode/reset 视为 engine 启动配置，不能在带有旧缓存的运行中直接改值；协议 v3 升级必须重启所有 worker。
- 完整 engine 的 prefill→decode、logits/生成对比和模型级准确率仍是后续验收项，不能用上述单算子结果替代。
