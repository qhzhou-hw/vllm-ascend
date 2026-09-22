# PIC 调度适配首版：实现记录与验证清单

状态：代码已实现，已在新 Ascend 服务器执行控制层、实际 vLLM 类型和 tensor 单测。
完整 engine、TP、模型准确率及性能验证尚未完成，不能据此宣称 PIC 已通过端到端验证。

## 1. 已落地的行为

- 匹配：plan v2 使用 token 内容、完整 token 比较、cache salt 和有效 GDN coverage。
  pool 为 engine 私有，engine epoch 隔离模型生命周期；首段 full 与中间段 interior
  payload 不再仅凭相同 token hash 互相命中。
- 槽位：`PicCatalog` 是 slot/generation/victim 的唯一决策者，worker
  `PicDeviceCache` 只验证并执行。一次 step 的 read 集合和 fill 集合都持有隐式 lease，
  同步模式下只允许一个 step 在途，不能驱逐这些集合中的条目。
- 重复 miss：各 occurrence 正常计算，只有 scheduler 指定的首个 occurrence 写入
  持久 pool。worker 请求重排不改变写入者；不再依赖两边 LRU 操作顺序相同。
- 容量：所有 hit 先固定槽位；不足以容纳新 fill 时，对剩余 miss 使用 COMPUTE_ONLY。
  短于或等于 seam 的中间段不生成零长度 interior unit。
- ACK：worker 核对全部 pool-bearing attention/GDN 层完成，等待 NPU 写入，TP
  all-reduce 确认所有 rank 完成，再通过显式 `PicModelRunnerOutput` 返回确认。
  scheduler 校验 epoch、step、fill 列表和 rank 数后才发布 READY。
- APC 隔离：PIC 和 full_recompute 请求既不读普通前缀缓存，也不发布普通前缀缓存。
  publication guard 位于 coordinator 的最终 cache_blocks 入口，覆盖分配和 decode。
  prefix_only 请求仍使用 native APC，并与 PIC prefill 分批调度。

新文件为 `hypic/protocol.py`、`pic_cache.py` 和 `vllm_adapter.py`；platform/worker
patch 负责接线，attention/GDN 数值路径保留既有算法，增加选择性写入。
旧 `SegmentCatalog`/`DeviceSegmentCache` 仅保留作历史单测和独立 helper 的兼容接口，
生产 PIC 路径不再使用它们。

## 2. 上游兼容方式与首版边界

没有复制上游 scheduler，也没有修改本地 vLLM 参考仓库。
`get_computed_blocks()` 对 PIC 返回零连续前缀命中，原 scheduler 按完整逻辑长度
分配 blocks 并执行原有 optimistic/in-flight 记账。成功 admission 后，显式
`PicSchedulerOutput` 将 worker 计算量替换为真实 query 数，并保留逻辑 token 数。
处理输出时向原 scheduler 还原逻辑计数，避免 inflight 只扣 query 数导致状态漂移。

V1 worker 原 metadata builder 需要 `seq_len = base + query_len`。adapter 仅在
worker-local `NewRequestData` 副本中设置 base=N-Q，随后用 plan 中的绝对位置覆盖
token gather 和 slot mapping。这个 base 不进入 scheduler，也不表示真的存在连续
前缀命中。该兼容层仍需通过真实 metadata 构造、decode 衔接和 RPC 验证。

单 query 的 sparse prefill 仍使用 PIC 路径；GDN 写回使用覆盖所有 request rows
的 non-spec state indices，避免 native metadata 将短 query 归为 decode 后丢掉行。

首版刻意保留以下限制：

| 项目 | 当前行为 |
| --- | --- |
| token admission | 保守按完整逻辑长度预算；query 和 restore 量均不超过此预算 |
| GDN workspace | `max_prefill_units=256` 默认上限，按实际 segment/unit 数准入 |
| 单请求超预算 | 明确报错；不无限等待，也不自动更改 chunk/seam/model |
| 调度 | eager、同步、PIC prefill 与 native prefill/decode 分批 |
| 抢占恢复 | native 重算完整已知序列；继续禁止该 PIC 请求发布普通 APC |
| 取消 | 等待已下发 forward/ACK；有效 fill 可保留，之后释放 step lease |
| 执行失败 | fail closed，需重启 engine；不自动修复部分 TP rank 的 pool |
| APC + PIC 同请求 | 尚未实现，需要混合前缀状态 checkpoint 恢复 |
| 零拷贝/异步/chunked prefill | 后续阶段，未包含在本次实现 |

逻辑预算可能降低 warm cache 下的有效 batch size，新增 ACK 同步也有成本。
本次不宣称吞吐提升；应先验证正确性，再评估 query/restore 独立预算和异步 ACK。
`max_prefill_units` 控制临时状态数量，不能代替特定模型的真实显存 profile。
继续使用构造期预分配 pool，未引入运行期持久缓存扩容。

## 3. 本地已完成的检查

在没有 torch/vLLM/NPU 的本地 Python 环境执行了：

```powershell
python tests/ut/hypic/test_pic_control.py
python tests/ut/hypic/test_pic_adapter.py --schema-root <vllm-source-root>
```

控制层 22 项测试通过，其中包括 500 轮固定种子的随机 packed 请求、槽满驱逐和
容量稳定性测试。adapter 契约 13 项测试通过，schema 来自 vLLM `58d3918e3` 参考源码。

schema-only 模式只抽取真实 dataclass 定义及生产 hook 函数体，验证消息 pickle
往返、逻辑/query 计数分离、APC guard 和 admission；没有导入或执行完整 vLLM。
这不是实际 multiprocess executor、CANN kernel 或 Ascend 推理验证。

另已编写 attention 唯一写入者、GDN COMPUTE_ONLY/state-index 的 tensor 单测，
需要 torch 测试环境后执行。现有历史 HYPIC 单测继续保留。

变更的 Python 文件使用仓库固定版本 Ruff 0.14.0 做格式和 lint 检查。
完整 `bash format.sh ci` 仍受本机 WSL 启动权限限制，应在测试服务器补跑。

## 4. 测试环境到位后的执行顺序

先记录 vLLM/vLLM-Ascend commits、Python、torch/torch_npu、CANN、sgl-kernel-npu
版本和模型配置。当前契约测试只覆盖参考接口，不能直接推断 v0.27.1 或其他提交兼容。

### 4.1 单测和真实消息类型

```bash
pytest -q tests/ut/hypic/test_pic_control.py \
  tests/ut/hypic/test_pic_adapter.py tests/ut/hypic/test_hypic.py
```

安装环境中的 adapter 测试应不带 `--schema-root`，使用实际 vLLM 类型。
确认所有已有与新增测试通过，再启动模型。

### 4.2 小池和 APC 隔离 smoke

```bash
VLLM_LOGGING_LEVEL=DEBUG python examples/hypic_pic_smoke.py \
  --model /path/to/Qwen3.5-35B-A3B --tp 2 --batch-size 4 \
  --cache-segments 2 --chunk-size 512 --seam 8 \
  --output /path/to/new-pic-smoke-b4-seam8.json
```

脚本运行 full_before、PIC cold/warm、工具重排、prefix_only cold/warm 和 full_after，
保存逐条 token 输出。输出文件采用排他创建，不覆盖已有实验结果。
它使用直接构造的 token chunks，验证边界与控制路径，不代表 MCPAgentBench 的 prompt。

分别用 bs=1/4、TP=1/2、seam=0/8、大/小 pool 运行。原生 full 与普通 APC 的输出应
保持一致；PIC 差异单独报告。验证 full_after 不受先前 PIC 污染，并检查热命中 step
确实存在 query 减少，不能仅凭生成文本正常判定命中生效。

还需在真实 server/executor 上补充：生成中取消、decode 抢占后恢复、worker
重启、TP rank 故障注入；错误后不得出现 READY 假命中。验证错误明确信息后重启，
不要复用部分覆盖的旧 pool。

### 4.3 准确率与性能

保持原模型、数据、候选工具顺序、seed、prompt 和评分方法，先固定样本对比
full recompute / legacy HYPIC / new PIC，再运行完整 MCPAgentBench 与 LongBench。
MCPAgentBench 只评首轮 tool_calls，不执行工具。coverage key 和唯一写入者改变了
旧缓存的填充语义，输出差异必须逐条分析。

收集冷/热 TTFT、吞吐、p95、实际 query/restore token 数、fill/eviction 数、
静态 pool 和峰值 workspace 显存。TP 每步同步的影响需要独立统计。

## 5. 新服务器验证记录（2026-09-16）

环境为两张 Ascend 910B2C（每卡 64 GiB）、CANN 9.0.0、Python 3.12.13、
torch 2.10.0+cpu / torch_npu 2.10.0。两卡基础 NPU 张量运算正常。
使用 `/data/venvs/pic` 独立虚拟环境，保留镜像原有 vLLM 0.21 安装。
测试导入路径已核对为 `/data/vllm` 和 `/data/vllm-ascend`，前者采用
`58d3918e3` 参考源码，版本标识为 `0.26.1rc1.dev517+g58d3918e3`。

三个测试文件不带 `--schema-root` 执行，最终 **60 passed**，日志为服务器上的
`/data/pic-validation/unit-tests3.log`。其中 tensor 单测仍使用 CPU 张量及部分
mock，不等价于真实 NPU attention/GDN 或 multiprocess executor 测试。

首轮 tensor 回归暴露了共享输入 plan 的别名问题：对整个请求字典执行一次
`deepcopy` 会保留不同请求之间的对象共享，首个 occurrence 的 `store=True`
因此也出现在第二个请求中。`PicCatalog.prepare` 已改为逐请求独立深拷贝，
增加原始 plan 不被修改、每个请求独立持有 reservation 的回归测试。

sgl-kernel-npu 从 `2026.05.01` tag 在服务器构建，安装版本为 `2026.5.1`。
其独立 GDN NPU 测试首轮未通过：镜像预装 Triton 3.2.1 的公开
`triton.language` 缺少 `insert_slice`，30 项测试均在编译前报错。
从华为 Ascend pip 源安装 Triton-Ascend 3.2.2 后该旧接口仍不存在，
因此改用此前 HYPIC 文档固定的 `3.2.1.dev20260530` 版本复测，但该版本也需要
修正旧内核接口路径：在 sgl-kernel-npu 的 `fla/solve_tril.py` 导入
`triton.language.extra.cann.extension as tle`，将 `tl.insert_slice` 和
`tl.extract_slice` 改为 `tle` 下的同名函数。未改变内核计算公式。
原文件及 diff 分别保存在服务器 `/data/pic-validation/solve_tril.py.original`
和 `sgl-triton-api.patch`。依赖重新构建后，其 `test_gated_delta.py` 的 30 项
真实 NPU 变长 GDN 测试通过，日志为 `/data/pic-validation/gdn-tests3.log`。
这覆盖底层算子，不等于完整 HYPIC 状态组合或模型准确率验证。
通过正常 plugin 初始化顺序导入 NPUModelRunner 和 PIC worker patch 也已成功，
日志为 `/data/pic-validation/worker-plugin-import.log`。
Qwen3.5-35B-A3B BF16 权重正在下载；Hugging Face 直连超时后改用同名
ModelScope 仓库。后续需记录模型 revision，并在权重及 Ascend 自定义算子
准备完成后执行第 4 节的 engine smoke，再进行准确率对比。
