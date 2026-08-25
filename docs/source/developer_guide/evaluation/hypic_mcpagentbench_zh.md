# HYPIC MCPAgentBench 工具选择评测与语义分段

本文说明如何把 OpenAI `tools[]` 中的每个工具 schema 作为 HYPIC 独立语义段，并记录 Qwen3.5-35B-A3B 在 Ascend 上的 MCPAgentBench 全量工具选择结果。

## 1. 评测范围

本实验只评估首轮工具选择：模型返回 `message.tool_calls` 后立即结束，不执行工具，也不返回工具结果。参数不计分，只比较工具名。主指标是 multiset exact match，它忽略并行工具调用顺序，但保留重复工具次数。

Prompt 使用去重版本：工具定义只通过 `tools[]` 输入，删除 system prompt 中重复的 `Available Tools List`。每条请求固定 40 个候选工具，并明确要求模型一次性选择标注数量的全部工具。

## 2. 每个工具一个 HYPIC 语义段

Qwen chat template 会把 `tools[]` 序列化到 `<tools>...</tools>`，每个工具通常对应一行 JSON。OpenAI 请求本身不直接包含 token 边界，因此评测代理先调用 vLLM render 接口获得模板渲染后的 token IDs，在工具 schema 的起止位置建立 hard boundaries，再把边界通过 `vllm_xargs.hypic_segment_boundaries` 传给 scheduler。

Planner 对 hard boundary 之间的区域独立分段；当单个工具 schema 超过 `chunk_size` 时，仍按固定 chunk size 继续子分段。因此语义边界不会让 segment 超过显存规划上限，也不会把两个工具 schema 合并到同一个 segment。未提供语义边界时，行为保持原来的纯固定长度分段。

需要注意：

- 边界必须基于最终 chat template 的 token IDs，而不是原始 JSON 字符偏移。
- 候选工具顺序必须确定，不能用 `set` 去重，否则 Full 与 HYPIC 的输入不可比。
- render payload 必须纳入代理缓存 key；只按 messages 缓存会错误复用另一组 tools。
- 客户端取消请求时要同步取消上游生成，并禁用失效 keep-alive 复用，避免残留连接拖住后续请求。
- 最后一个 segment 仍按 HYPIC 规则完整计算，以产生采样所需 logits。

对应实现位于：

- `vllm_ascend/hypic/planner.py`
- `vllm_ascend/patch/platform/patch_hypic.py`
- `tests/ut/hypic/test_hypic.py`

## 3. 固定实验配置

| 配置项 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B |
| 请求数 | 178（全量） |
| 候选工具 | 每条 40 个，顺序固定 |
| 随机种子 | 20260824 |
| Prompt | `deduplicated-selection-only` |
| HYPIC chunk size | 512 |
| 解码 | temperature 0，top_p 1，max_tokens 512 |
| 工具执行 | 禁止 |

两组逐条校验了 `task_id`、标注工具和候选工具顺序，均完全一致；prompt token 总数均为 2,277,988。两份日志中的 `CallToolRequest` 都为 0。

## 4. 全量结果

| 指标 | Full Recompute | HYPIC-512 | HYPIC - Full |
|---|---:|---:|---:|
| Ordered exact | 113/178（63.48%） | 110/178（61.80%） | -1.69 pp |
| Multiset exact（主指标） | 118/178（66.29%） | 113/178（63.48%） | -2.81 pp |
| 工具级 recall | 237/334（70.96%） | 207/334（61.98%） | -8.98 pp |
| 工具级 precision | 237/239（99.16%） | 207/210（98.57%） | -0.59 pp |

按工具数量分组：

| 分组 | 指标 | Full Recompute | HYPIC-512 | HYPIC - Full |
|---|---|---:|---:|---:|
| 单工具（60 条） | Multiset exact | 58/60（96.67%） | 60/60（100.00%） | +3.33 pp |
| 多工具（118 条） | Multiset exact | 60/118（50.85%） | 53/118（44.92%） | -5.93 pp |
| 多工具（118 条） | Recall | 179/274（65.33%） | 147/274（53.65%） | -11.68 pp |

配对统计为：两组都正确 100 条、两组都错误 47 条、仅 Full 正确 18 条、仅 HYPIC 正确 13 条。47 条请求的预测不同，其中 Full 的正确工具数更多 29 条，HYPIC 更多 16 条，二者相同 2 条。

## 5. 结论

HYPIC-512 的请求级 multiset exact 比 Full 低 2.81 个百分点，但多工具请求 recall 低 11.68 个百分点。单工具请求没有退化，差异集中在需要同时选择多个工具的请求。因此当前结果不能表述为完全等价；后续应重点检查多个工具 schema 分段后的跨段组合信息，以及一次性生成多个 tool calls 时的漏选。

该实验不执行工具，也不评估参数质量和多轮依赖，不能替代 MCPAgentBench 的端到端任务成功率。
