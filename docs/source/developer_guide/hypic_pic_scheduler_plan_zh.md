# PIC 前缀无关 KV 匹配与复用：实现方案

本文记录总体设计，分阶段能力不能视为全部已经可用。
首版落地范围、兼容折衷和待实机验证项见
[实现记录](hypic_pic_implementation_zh.md)。方案基于 `hypic` 分支的现有实现，
调度接口核对使用本地 vLLM `58d3918e3` 参考源码；实施时必须锁定实际配套的 vLLM
提交，并验证接口和跨进程序列化，不能直接假设其他版本兼容。

## 1. 目标和语义

PIC 在本文指按 segment 内容匹配、允许前缀及绝对位置变化的缓存复用能力；
HYPIC 是当前 Qwen3.5 混合注意力模型上的实现。首版继续使用现有 HYPIC 数值算法，
把匹配、资源准入和缓存生命周期接入明确的调度协议。

例如已有请求 `[system A][tool X][tool Y][question A]`，新请求为
`[system B][tool Y][tool X][question B]`。即使连续前缀缓存没有命中，新请求也可以
匹配 X、Y，重算未命中部分、必要的 seam 和末段，并为 decode 建立完整请求状态。

必须区分以下语义：

- 普通 APC（Automatic Prefix Caching）按照完整前缀身份复用已有计算。
- PIC 内容匹配不包含前缀身份，RoPE 重定位只处理位置变换，不能消除深层 hidden
  state、KV 和卷积输入对原上下文的依赖。因此当前 HYPIC 是需测量准确率的近似复用。
- 第一阶段复用的是计算结果；命中 KV 仍物化到请求自己的 paged cache，不能据此宣称
  请求 KV 显存减少或实现了零拷贝共享。

## 2. 当前实现与需要调整的位置

| 组件 | 现状 | 计划 |
| --- | --- | --- |
| `hypic/planner.py` | token hash + 完整 token 比较；固定最大 chunk 和语义边界 | 保留分段规则，引入表示身份、显式查询区间和 version 2 plan |
| `hypic/cache.py` | scheduler/worker 双 LRU，授权 victim 并检查顺序 | scheduler 唯一分配 slot；worker 验证并执行命令 |
| platform `patch_hypic.py` | 返回空 computed blocks，以跳过 token 数模拟 computed prefix | 单独传输稀疏复用计划和逻辑推进量，恢复原字段语义 |
| worker `patch_hypic.py` | prefill-only packed batch，重写 input positions 和 slot mapping | 消费结构化计划，增加请求/step/slot 版本验证和完成反馈 |
| `hypic/attention.py` | 本地位置 K 缓存、RoPE 重定位、写回 paged KV | 保留数值路径；适配 slot lease 和选择性缓存写入 |
| `hypic/gdn.py` | 缓存 S/T、conv tail；forward 内组合请求状态 | 保留组合算法；明确 payload 覆盖区间和唯一写入者 |
| 模型构造期静态 pool | 在普通 KV cache 定容前分配 | 保留，增加容量、峰值工作区与 admission 预算核对 |

现有一致性修复已将查找变为只读、驱逐推迟到 admission 后，并禁止异步调度。
新协议应继承这些约束；迁移期间保留已有 divergence 回归测试。

## 3. 匹配身份和分段规则

缓存匹配分两级：

```text
namespace = model/weights epoch + adapter identity + cache isolation domain
            + dtype/quantization + RoPE configuration + algorithm/layout version
content_key = hash(segment token IDs) + token length
representation = payload kind + effective seam/coverage + supported state layout
entry_key = (namespace, content_key, representation)
```

namespace 固定在 engine 生命周期时可保存为内部 ID，不需要每次重复传输完整配置。
TP worker 保存对应 rank 的 shard；TP topology 和 shard layout 必须在命名空间中一致。
已有 request cache salt/租户隔离信息应传入 PIC，禁止跨隔离域命中。
hash 只用于查表，完整 token IDs 仍用于碰撞确认。首版只允许完整 segment 精确匹配，
不做子串、编辑距离或语义相似匹配。

继续接收 `hypic_segment_boundaries`：边界基于最终 chat template 渲染并 tokenize
后的 token 序列；不能通过工具 JSON 原始字符偏移直接推导。每个工具单独成段，超出
chunk_size 的工具在自己的边界内继续切分。没有显式边界时保留现有固定长度规则。

当前首段 cold fill 保存 full S/T，中间段通常保存去除 seam 后的 interior S/T，
但两者使用相同 token hash。新 key 必须区分有效 coverage：首段产生的 full payload
不能直接当作中间段 interior payload，反向也不行。配置 seam=0 时两者可有相同
coverage，但仍属于 PIC 的近似语义。短于或等于 seam 的 segment 应全量计算，
不调用零长度 GDN unit，也不为没有复用收益的 interior 分配槽。

首版保留末段必须执行的规则以生成 logits。不同请求中的相同 segment 允许匹配，
但同一批次的 cold miss 不视为已完成 hit；它们分别计算，由 scheduler 确定一个
写入者保存规范副本，其余 occurrence 不覆盖该 slot。这保证 batch 重排不会导致
同一 slot 在不同层混合不同来源的 payload。该策略可能改变旧实现的输出，需单独消融。

## 4. 调度与 worker 协议

建议新增 `hypic/protocol.py` 和 `hypic/scheduler.py`，数据字段示意如下；
实际定义采用配套 vLLM 支持的显式可序列化类型，不依赖运行时随意追加对象属性。

```text
PicRequestPlan:
  version, engine_epoch, step_id, request_id, request_generation
  logical_length, exact_prefix_length, query_ranges, logical_advance
  occurrences[]:
    start, end, entry_key, action, recompute_ranges
    slot_id?, slot_generation?, write_owner?

PicStepPlan:
  request_plans[], reservations[], evictions[], leases[]

PicStepResult:
  engine_epoch, step_id, rank, success
  completed_fills[(slot_id, generation)], completed_requests[]
```

`action` 至少包括 REUSE、COMPUTE_AND_STORE、COMPUTE_ONLY。调度器的预算和输入
打包使用 query token 数；请求的实际位置、注意力可见长度、KV 分配和 decode 起点
使用 logical length。成功后逻辑完成度推进 logical_advance，而非简单加 query 数。
提交前使用 in-flight 状态防止重复调度，不能提前宣告 skipped KV 已有效。

接入范围包括 scheduler admission、输出 schema、worker 输入构造、完成度更新和
失败/抢占回滚。不得只修改 `get_computed_blocks()` 返回值：上游 scheduler 的
running/waiting、prefill 判定、block 分配和更新路径均使用连续前缀计数。
通用协议宜作为小范围 upstream 扩展，Ascend 侧负责策略和执行；若先在插件落地，
把版本适配集中在独立 adapter，并验证签名，避免复制整份 scheduler。

每个调度 step 按以下顺序执行：

1. 只读匹配候选请求，构造 query/restore/fill 需求。
2. 在 shadow transaction 中同时规划 paged blocks、segment 槽及预算；失败不改变 LRU。
3. admission 成功后固定命中条目的 lease、确定 slot/victim、登记 FILLING 写入者。
4. worker 先验证整个计划，再执行覆盖、恢复 KV、稀疏 forward 和状态写回。
5. 收齐所有相关 TP rank 的完成证据后，发布 READY、推进请求逻辑状态并释放 lease。
6. 正常 decode 使用已有 paged KV 和 request-local GDN state。

## 5. 槽位管理、失败恢复与 admission

scheduler 保存 `entry_key -> slot_id, generation, state, pin_count`；worker
保存执行镜像。worker 不独立选择 LRU victim。缓存生命周期为：

```text
FREE -> RESERVED -> FILLING -> READY -> EVICTING -> FREE
                       \-> INVALID -> FREE
```

每次槽位复用增加 generation，engine 重启增加 epoch。任何读写都验证
`(epoch, slot_id, generation, entry_key)`；过期结果不得将新槽标记 READY。
只有 READY 且 pin_count=0 的条目可被驱逐。FILLING 不能作为跨请求 hit。
唯一写入者和所有层完成是 READY 的必要条件，不能仅凭某层写完或请求出现在
`scheduled_new_reqs` 就发布命中资格。

首版保持同步 schedule/execute/update 时序；完成证据可以在现有 TP 执行成功语义上
聚合，但必须核查所有 rank 和所有目标层已完成，不能只采纳采样 rank 的结果。
命中的 lease 持有到本次 forward 不再读该 pool；物化完且请求状态写回后，decode
通常不再需要持有 segment lease。

预算必须同时覆盖四类资源：

| 预算 | 用途 |
| --- | --- |
| compute tokens | 未命中 token、seam、末段；决定真实模型计算输入 |
| restore tokens/bytes | 所有层的 KV 拷贝和 RoPE 重定位；避免大量 hit 淹没单 step |
| request KV blocks | 完整逻辑上下文和 decode 增量，按各 hybrid cache group 规则分配 |
| segment slots / workspace | 已 pin 的 hit、唯一 fill、GDN 和 attention 临时工作区 |

slot 需求按实际分段后的 distinct entry key 联集计算。现有
`ceil(max_num_batched_tokens/chunk_size)-1` 不能替代检查：语义短段、多个请求和大量
skipped tokens 都会改变实际需求。静态 pool 与请求 KV、模型和工作区合计必须进入
显存预算，不能在 profile 完成后再额外扩容。

容量不足时，先降低该 step 接纳的请求数，或将低收益填充改为 COMPUTE_ONLY。
若一个请求本身超过 pool 容量，保留能 pin 的 hit，仅缓存选中的 miss，其余正常计算；
相应 attention/GDN 路径要支持无持久槽的 miss。重规划后重新核算 query 和工作区预算。
单请求全量计算也超限时明确报告容量限制，不能无限等待或在 worker 临时改变计划。

失败处理区分以下情况：

- 未下发：撤销预留和 lease，不改变有效缓存。
- 已覆盖 victim：原 victim 不得“回滚为 READY”，因为数据已丢失；使目标 slot 失效，
  等待所有使用者退出后回收，并重规划请求。
- 取消/抢占：只释放该请求持有的引用；in-flight 写入先结束或确认取消，避免提前复用。
- 请求恢复：增加 request generation，从有效的连续状态或重新规划的 PIC prefill
  恢复；不能继续使用原 step 的槽引用。首版不支持未验证的跨步稀疏续跑。
- worker/rank 重启：全 engine epoch 失效，清理 catalog 的 READY 资格并重新预热。

## 6. 与普通前缀缓存及现有功能的兼容

PIC 产生的 KV 可能近似，不能以普通 token-prefix key 发布为精确 APC 结果。
首版对 PIC 请求禁用普通 APC 的查询与发布，但保持普通 KV block 分配、释放及 decode；
普通请求继续走原有 APC。隔离必须覆盖 allocate-time cache、后续 decode/cache_blocks、
请求结束等所有发布路径，不能只跳过一次 `get_computed_blocks()`。

保留 `additional_config.hypic_config` 和现有 chunk/seam/pool 配置，默认关闭功能。
新增请求级策略（接口名待实施确定）：full_recompute、prefix_only、pic；前者禁止
两类复用以作为实验基线。已有 HYPIC enabled 用法默认映射到 pic，旧边界参数继续有效。
旧配置可兼容，但新 plan/payload 格式需要重建内存缓存，不能混用旧 slot。

| 功能 | 第一阶段承诺 | 后续条件 |
| --- | --- | --- |
| Qwen3.5 text-only、TP、batch > 1 | 沿用现有支持范围并回归 | 不扩大模型支持声明 |
| 普通请求 APC | 同 engine 路由隔离；分别调度 prefill 批次 | PIC 结果不能污染 exact namespace |
| 单请求 APC + PIC | 暂不启用组合 | 完整恢复精确前缀 KV、GDN state、conv state 后启用 |
| eager、同步调度、prefill/decode 分批 | 保留当前约束 | lease/ACK 验证后再开放异步与混合 batch |
| chunked prefill | 保持关闭 | 需要跨步 query/restore 状态、GDN 边界状态和 lease 生命周期 |
| PP/DP/CP、KV transfer、speculative、prompt logprobs | 继续明确拒绝未支持组合 | 各自单独设计和验证 |
| graph、量化、多模态、LoRA | 不因本方案自动增加支持 | payload、隔离身份和对应 kernel 均需适配 |

第二阶段组合 APC 与 PIC 时，先取所有 hybrid group 共同有效的精确前缀 P，
恢复各层 GDN recurrent/conv 状态，再在 `[P,N)` 匹配 segment；复用前缀的共享 block
保持只读，边界 block 的新增写入需要按原有 COW 规则分配。P 若落在 segment 内，
首版组合策略将该 segment 剩余部分重新计算，不强行复用不匹配的 payload。
只有确认支持相应混合状态 checkpoint 的位置可作为 P，不能只看 full-attention KV。

以后若允许 PIC 结果再次作为连续前缀缓存，需要独立 PIC namespace 和可验证的
执行来源身份；仅添加一个固定“PIC”盐值仍不足以区分不同缓存历史产生的近似状态。

## 7. 分阶段实施与验收

| 阶段 | 可交付内容 | 验收重点 |
| --- | --- | --- |
| P0：冻结基线 | 配套 vLLM commit、现有冷/热输出、指标与缓存语义测试 | 明确 legacy 行为及计划中的变化 |
| P1：匹配与协议 | namespace/coverage key、version 2 plan、唯一写入者、序列化测试 | chunk 换位命中；错误 payload、盐值及旧 generation 不命中 |
| P2：缓存事务 | scheduler slot allocator、pin、完成反馈、失败/取消处理 | 槽满长跑无失配、无悬挂引用、无未完成 hit |
| P3：调度接入 | 分离 query/逻辑计数、预算、COMPUTE_ONLY、APC 查询发布隔离 | 多 batch 长短请求、decode、资源不足回退正确 |
| P4：实机验证 | 固定样本、完整准确率、吞吐及显存报告，更新用户文档 | 正确性门槛通过后才默认启用新路径 |
| P5：独立优化 | 同请求 APC+PIC、混合 batch、异步、分页读取优化 | 每项单独协议/性能回归，不并入首版验收 |

代码组织优先在 `vllm_ascend/hypic/` 内增加协议、调度策略及状态机，保持 platform
patch 为版本 adapter；worker patch 只传递计划和执行反馈。修改已有 attention/GDN
函数以接收显式 slot 与 write_owner，不在 forward 中决定准入或驱逐。

测试覆盖以下情况：

- 不同前缀、位置和工具顺序；相同内容重复出现；token hash 碰撞；不同 cache salt。
- 首段/中间段互换，seam=0/8，长度 1、seam、seam+1、chunk_size 及非 block 对齐边界。
- packed 重复 miss、请求重排、槽满驱逐、被 pin 的 victim、未接纳请求不改变 LRU。
- TP 部分失败、过期 ACK、取消、抢占恢复、worker 重启、写后失败和长时间池容量稳定。
- PIC 开/关交错：普通 APC 命中结果不受 PIC 请求历史影响；返回 full_recompute 的
  请求不读取任何复用结果；每次 decode 前验证逻辑 KV 与 GDN state 有效范围。
- 请求计划及完成反馈必须走真实 engine/worker 通道测试，不能只有进程内对象测试。

数值验证区分三组：full recompute、legacy HYPIC、new PIC。协议/管理重构在相同
payload 来源和相同输入上应保持结果一致；coverage key、唯一写入者等语义修正单独
验证，不将它们带来的差异笼统归因于浮点误差。

所有模型执行和 NPU 性能验证在 Ascend 服务器完成。先固定种子采样，再运行原有
LongBench 和 MCPAgentBench。MCPAgentBench 固定 178 请求、每条 40 候选工具、新去重
prompt、seed 20260824，只评首轮 tool_calls，不执行工具；逐条校验候选工具顺序和
token 边界。沿用已有 ordered/multiset exact、micro precision/recall 评分。
LongBench 沿用原模型、数据和评分，并保留各长度区间结果。

准确率不预设与 full recompute 完全相等；报告相对 legacy 和 full 的逐样本差异及
统计不确定性，在正式运行前固定允许的回退门槛。无静默错槽、无未完成数据读取、
无 exact APC 污染、无资源泄漏是必须满足的确定性门槛。

性能报告分别记录冷/热请求、TTFT、吞吐、p95 延迟、有效跳过 token、restore bytes、
填充/驱逐次数、等待 slot 次数、静态池/请求 KV/峰值工作区显存。命中率高但恢复成本
抵消收益时，调整准入和复用收益阈值，不能只以命中率判定优化成功。

## 8. 后续零拷贝方向

保留 segment-local K 并在 attention kernel 读取时做位置修正，理论上可以避免
整段物化。但需要新的 segment/page 混合寻址、seam 私有覆盖、跨请求只读引用管理、
decode kernel 支持及完整性能验证。它改变现有 paged attention 数据访问契约，
应在第一阶段可用后单独立项；不能直接把 segment pool 的地址塞进普通 block table。

相关已有文档：[Ascend 适配说明](hypic_ascend_porting_zh.md)、
[缓存一致性修复](hypic_cache_consistency_fix_zh.md)。
