# HYPIC 有序状态融合与全命中 GDN 快路径

## 状态

2026-09-24：前两轮 kernel 实验记录见第 1–6 节；第三轮 GDN 优化与最终保留范围见第 7 节。
第二轮实现已重新通过 78 项张量回归、23 项控制面回归、16 项 adapter
schema 回归，以及融合后端 group=1/group=4 各 10 项真实 NPU mock 检查。状态融合
前后对照 66 组全部通过数值检查。未下载模型，未在本地运行推理或测试。
这验证的是算子和缓存流程，不代表完整 vLLM engine 或模型级准确率验收。

默认融合后端仍为 `torch`；第三轮全命中快路径适用于两个后端。Triton 必须显式选择，
不能把下面的配置视为已经验收的生产配置。
第一轮的负结果保留在第 5 节。第二轮融合核相对第一轮快
2.03–5.13 倍；较长序列及多请求组合也快于 `torch`。短序列仍有固定开销，不自动启用。
同进程、同输入的三方延迟对照见第 6 节；这不是完整模型 TTFT 的加速倍数。

## 1. 本次优化范围

本节至第 6 节记录前两轮实现；第 7 节覆盖后续 GDN 调用层面的变化。
对于每层、每个请求，按原顺序执行：

```text
H_0 = 0
H_i = H_(i-1) @ T_i + S_i
```

仅把逐段 Python `bmm + add` 调度替换为一个持久执行的 Triton kernel：

- 每个 program 负责一个请求、一个 head、最多 8 条 value 行，保留完整 key 维度。
- program 内按 segment 顺序更新，不做树形归并，不交换段顺序。
- 独立请求、head、value 行块之间可以并行。
- 初始状态为零，因此第一步直接复制 S，省略 `0 @ T`。这要求输入状态和 transition 有限；
  不把异常 NaN/Inf 缓存作为等价性目标。
- hit 直接读取原静态池；miss/seam/Query 读取本轮计算的 S/T。索引/指针表合为一次小型上传，
  不 gather 全部 hit 张量到另一份大缓存。
- 按需写出 replay 的 initial state，同时产出每个请求的最终状态。不会修改公共 S/T。

该路径使用 FP32 S/T、向量乘法/归约和累计状态，不降成 BF16/TF32。
矩阵乘内部归约顺序可能与 `torch.bmm` 不同，不能承诺逐位一致。
当前接口要求同一 NPU 上连续的 FP32 张量、`1 <= key_dim <= 128`；没有经过验证的
形状不应据此推断性能。

实机调试发现并避开了两类问题：

- Ascend Triton 拒绝合并不同来源的指针，改为在 fresh/pool 分支内分别 load，再合并值。
- 最初的 FP32 `tl.dot` 实现未通过单位矩阵检查，不能用于正式结果。替换为 FP32 向量
  归约后通过检查；当前归约维每次处理 16 个元素，避免整张矩阵归约产生过量 UB 临时空间。

这些是当前实现/编译器组合的观察，不应推广为所有 Ascend FP32 矩阵乘均不正确。

**不在前两轮范围内**：减少 Query 的 GDN pass、复用 miss Document 第一遍输出、树形 scan、
缓存多段组合结果、改变 attention/conv、改变 scheduler 的 PIC 命中规则。

## 2. 两种语义模式如何保留

`build_compose_steps` 将现有 plan 转成有序的组成单元：

- hit 且有 seam：先处理本轮 seam，再处理缓存 interior。
- 无 seam 的 hit：处理缓存整段。
- miss：按原计划处理整段，或依次处理 seam/interior。
- 旧 HYPIC：每个需要 replay 的单元获得进入该单元前的累计 H。
- block-native：Document 的 replay initial state 保持零，仅 Query 获得 Document 的组合状态。

conv history 的传递/重置开关不变。最终 recurrent state 和 Query conv tail 仍写入当前
请求的 native decode 槽，槽号不假设等于 batch row。
前两轮保留旧 `torch` 执行分支及三遍 GDN 计算。第三轮共用有序参考组合函数，并为
无 fresh Document/seam 的请求增加单遍 Query 路径，见第 7 节。

后端和分组大小是执行策略，不改变公共 S/T 的数学定义，因此不加入 PIC segment hash，
也不改变协议版本。切换后端仍应重启 engine；不要在正在执行的 layer 上热改属性。

## 3. 配置和显存边界

```python
additional_config = {
    "hypic_config": {
        "enabled": True,
        "mode": "block_native_pic",  # 也支持 transition_rope_recompute
        "chunk_size": 512,
        "seam_sink_tokens": 0,
        "state_compose_backend": "triton",  # 默认 torch
        "state_compose_batch_size": 1,      # 默认 1，验证后可以尝试 4
    }
}
```

`state_compose_batch_size` 是最多保留多少个请求的待融合 workspace，不是 scheduler 的
`max_num_seqs`。设为 4 时，先为最多 4 个请求计算新 S/T，然后一次融合、分别 replay。
最后不足 4 个的分组也必须执行。设为 1 仍可获得 head/行块并行和段内 kernel 融合。

多请求分组会同时保留这些请求的 Q/K/V/g/beta 和 S/T，峰值显存可能高于原逐请求路径。
默认 1 避免默认增加跨请求 workspace 保留；增大时必须同时测显存峰值。完成组合后先
释放这组的新 S/T，再执行 replay。不额外永久分配一份 segment pool。

显式选择 triton 后，非 NPU、非法形状、缺少 Triton 或编译错误直接报错，**不会悄悄回退**
并把旧路径的耗时报告成优化结果。默认 torch 不导入 Triton kernel 模块。

## 4. 服务器验证入口

以下均使用 mock 权重，不下载模型；在与旧验证相同的依赖环境中执行。

```bash
# 张量回归；包含新 compose 元数据和分组流程测试。
python examples/offline_inference/hypic_mock_validate.py --unit-tests

# 对比真实 GDN 输出、最终状态及训练原语。先 group=1，再 group=4。
python examples/offline_inference/hypic_mock_validate.py --npu --doc-len 512
python examples/offline_inference/hypic_mock_validate.py --npu --doc-len 512 \
  --compose-backend triton --compose-batch-size 1
python examples/offline_inference/hypic_mock_validate.py --npu --doc-len 512 \
  --compose-backend triton --compose-batch-size 4

# 仅状态融合的数值/耗时 A/B：独立请求，cold/mixed/warm，非交换 transition。
python examples/offline_inference/hypic_compose_bench.py \
  --batch-sizes 1 4 --segments 4 16 64 96 --mode block-native --iterations 10 --repeats 3
python examples/offline_inference/hypic_compose_bench.py \
  --batch-sizes 1 4 --segments 4 16 64 96 --mode legacy --iterations 10 --repeats 3

# 边界形状，不用于推断 Qwen 128×128 状态的性能。
python examples/offline_inference/hypic_compose_bench.py \
  --dim 65 --value-dim 57 --heads 3 --segments 1 4 --iterations 3 --repeats 2
python examples/offline_inference/hypic_compose_bench.py \
  --dim 1 --value-dim 3 --heads 1 --segments 1 --iterations 3 --repeats 2
```

基准排除首次 JIT，但**包含**每次 metadata 上传、输出分配和同步计时范围内的计算，
报告多轮 median。必须先通过 final/replay 状态数值比较，再输出 speedup；重复启动后再次
校验结果与缓存未被修改。它不是 TTFT/完整模型 benchmark。

待完成验收：

- [x] 本次验证入口的 78 项张量单测通过；两个模式、reset 开关、seam、cold/mixed/warm 均回归。
- [x] 当前向量归约核 NPU JIT 成功，源码不包含 BF16/TF32 转换，单位矩阵检查严格零误差。
- [x] 非交换随机矩阵、不同 head/请求、ragged 分组、重复缓存 slot 的输出及所有 replay 状态对齐。
- [x] NPU 覆盖 `(key_dim,value_dim)=(1,3)/(65,57)/(128,128)`；张量回归覆盖最后不足一组。
- [ ] NPU 专项覆盖所有非法输入拒绝分支；接口已有校验，不代表各分支均已测试。
- [x] batch 1/4、段数 4/16/64/96、两种模式的中位耗时和融合调用新增峰值内存已记录。
- [ ] 全 GDN/engine 的峰值显存专项；算子 workspace 测量不覆盖多请求延迟 replay 的全部内存。
- [x] block-native 实际 GDN cold/warm/reorder/changed-prefix 回归，包括最终 decode state；
  torch/group=1/group=4 各 10 项检查通过。旧 HYPIC 的真实 GDN 端到端专项仍待补充。
- [x] 记录性能负结果，不为目标形状默认启用。

## 5. 第一轮实测结论（优化前，历史结果）

环境：Ascend 910B2，torch 2.10.0+cpu + torch-npu 2.10.0，triton-ascend 3.2.2
（导入 `triton.__version__` 为 3.2.0），sgl-kernel-npu 2026.9.0。

验证目录：服务器 `/data/hypic-compose-validation-20260924/`。新目录从之前的 mock
验证快照复制，只覆盖本次变更文件，旧验证结果保留。

- `unit-tests.log`：78 passed；`control-tests.log`：23 passed；`adapter-tests.log`：16 passed。
- `npu-torch.log`、`npu-triton-g1.log`、`npu-triton-g4.log`：每份 10 checks PASS。
  与训练原语对比的 GDN output 最大误差 `9.54e-6`，最终状态最大误差 `1.21e-4`；
  三份日志逐用例报告的误差一致。这不是完整模型准确率报告。
- `bench-block-native.log`、`bench-legacy.log`：各 24 组，128×128、8 heads。
- `bench-ragged.log`、`bench-small.log`：分别 12/6 组边界形状；四份基准均含严格零误差的
  单位矩阵检查。66 组随机状态检查最大绝对误差为 `2.24e-7`，阈值未放宽。
- `lint-final.log`：Ruff 通过；9 个 Python 文件 format check 通过。由于服务器是最小
  源码快照，lint 显式指定 `lint.isort.known-first-party=["vllm_ascend"]`，避免缺失的
  无关源码子目录被误判成第三方包。

以下为 block-native 的 warm 命中情况，仅测状态融合（ms，越小越好）：

| batch | Document 段数上限 | 原 torch | 融合核 | 原耗时/新耗时 |
| --- | --- | --- | --- | --- |
| 1 | 4 | 0.408 | 1.044 | 0.390× |
| 1 | 16 | 1.023 | 2.838 | 0.360× |
| 1 | 64 | 3.515 | 9.779 | 0.359× |
| 1 | 96 | 4.421 | 14.594 | 0.303× |
| 4 | 4 | 1.028 | 1.912 | 0.538× |
| 4 | 16 | 3.784 | 8.776 | 0.431× |
| 4 | 64 | 12.455 | 36.030 | 0.346× |
| 4 | 96 | 17.933 | 54.181 | 0.331× |

每请求另有一个 fresh Query；batch 中第 r 个请求的 Document 数为 `max(1, 上限-r)`，
因此 batch=4 是不等长请求。cold/mixed/warm 与 legacy 的全部 48 组吞吐比范围
`0.298–0.624×`，没有目标形状上的加速。边界形状部分 batch=4 用例有小幅收益
（最高约 1.64×），不能外推为 Qwen 的收益。

融合调用的新增 allocator peak 包含输出/metadata，不含已存在的输入 S/T 池，也不含
NPU 片上 UB。例如 warm、batch=4、96 段：原调用 5,248,000 bytes，新调用
4,201,472 bytes。不能据此声称完整 GDN 分组降低了峰值显存。

从实现上看，向量核按 value 行反复读取 transition，并用多轮向量归约代替矩阵单元计算；
这很可能抵消了调用融合收益，但尚未用硬件 profiler 分解各项成本。后续若继续这条路线，
应先解决矩阵单元路径的精度/编译兼容性，再比较成本；不要通过切换 BF16 或放宽阈值
把数值退化包装成加速。本次没有修改 Query pass 数、segment 顺序或训练语义。

## 6. 第二轮优化与同配置延迟对照

### 实现变化

本轮没有引入新的数值近似或改变 segment 组合次序，仅优化执行方式：

1. **连续加载 T**：按 `[reduction, key]` 读取，避免第一轮按转置方向跨行读取。
2. **跨行复用 T**：每个 program 同时维护 8 条 H 行，同一 T tile 由这些行共享。
   相较逐行 program，逻辑上的重复 T tile 读取减少，实际 HBM 流量仍取决于硬件缓存。
3. **归约 tile 调整为 16**：128 维的内部归约循环从 16 次减到 8 次。较大 tile
   候选触发编译错误，最终只保留通过测试的 8 行 × 16 归约元素组合。
4. **metadata 一次上传**：将指针、长度和 source/replay 索引打包为一个 int64 张量，
   替代三次独立 H2D 上传。索引在 kernel 中转回 int32；没有复制大型 S/T 池。
5. **避免冗余清零**：如果每个 replay 单元均有唯一写入目标，使用 `empty_like`，由
   kernel 完整覆盖；block-native 中未写入的 Document replay 仍保持零初始化。
   前置校验继续拒绝重复 replay 目标，不允许通过省略初始化读取未定义状态。

独立 `hypic_dot_probe.py` 的 12 个 FP32 矩阵乘用例与 NPU torch 对齐，说明不能把
第一轮失败归因于所有 FP32 `tl.dot`。但持久循环内的 Cube 候选即使禁用若干优化、
增加同步或经 GM 中转，仍未通过单位矩阵检查，根因尚未完全定位。因此未把它合入
运行路径，也没有通过 BF16 转换或放宽误差阈值规避问题。

### 测量口径与快照

服务器目录 `/data/hypic-compose-validation-20260924/optimization/`：

- `before/`：本轮开始前完整快照，未修改，保留上一轮逐行融合核。
- `after/`：本轮实现和同进程对照脚本。
- `compare-block-native.log`、`compare-legacy.log`：正式对照，各 24 组。
- `dot-probe.log`、`cube-*.log`、`tile-*.log`：筛选过程，包括失败候选，不混入正式结果。

同一进程内使用完全相同的 FP32 输入，分别运行原 `torch`、旧融合核、新融合核。
旧融合核仍经相同的输入校验入口，仅在 benchmark 内临时切换 launcher。排除首次 JIT，
预热后执行 5 轮 × 每轮 10 次，并轮换三种实现的计时顺序，取轮次中位数。
延迟包含 Python 调度、输入校验、metadata 上传、输出分配及 NPU 计算；不是纯设备事件耗时。
三者都必须先通过 final/replay 数值检查，且结果池在重复执行后不可被修改。

以下为 **block-native、warm、8 heads、128×128 状态**，单位 ms：

| 融合组大小 | Document 段数上限 | 原 torch | 优化前融合核 | 优化后融合核 | 前/后加速比 | torch/后加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 0.383 | 1.058 | 0.488 | 2.17× | 0.79× |
| 1 | 16 | 0.950 | 2.788 | 0.789 | 3.53× | 1.20× |
| 1 | 64 | 3.351 | 9.818 | 2.295 | 4.28× | 1.46× |
| 1 | 96 | 4.833 | 14.462 | 3.278 | 4.41× | 1.47× |
| 4 | 4 | 1.081 | 1.912 | 0.591 | 3.24× | 1.83× |
| 4 | 16 | 3.459 | 8.782 | 1.993 | 4.41× | 1.74× |
| 4 | 64 | 12.201 | 35.949 | 7.446 | 4.83× | 1.64× |
| 4 | 96 | 18.571 | 54.204 | 11.088 | 4.89× | 1.67× |

组大小 4 对应 `state_compose_batch_size=4` 的融合调用，不是仅把 scheduler batch
设为 4 而仍逐请求融合。与第一轮相同，每个请求另有 Query；第 r 个请求的 Document
数为 `max(1, 上限-r)`。表格中的前后值均为本轮重测，并非直接拼接两个时间点的日志。

全部 48 组（两模式 × cold/mixed/warm × 组大小 1/4 × 段数 4/16/64/96）：

- 相对旧融合核加速 **2.03–5.13×**。
- 相对 torch：block-native 为 **0.74–1.83×**，legacy 为 **0.84–2.84×**。
  不应只报告最大加速比；组大小 1 的 4 段短请求仍有回退。
- 数值最大绝对误差 `2.24e-7`，仍使用 `atol=2e-5, rtol=2e-4`，没有放宽。
- 不额外创建永久 segment 池；多请求 workspace 延迟释放的显存权衡仍存在。

最终代码另通过以下回归（对应 optimization 目录的日志）：

- `unit-final.log` 78 项、`control-final.log` 23 项、`adapter-final.log` 16 项全部通过。
- `npu-final-g1.log`、`npu-final-g4.log` 各 10 项真实 NPU mock 检查通过，包含 block-native
  cold/warm/reorder/changed-prefix 输出、训练原语对比和最终 decode state。
- `compare-ragged.log` 12 组、`compare-small.log` 6 组通过，覆盖非整 tile 和最小 key 维。
  加上两组主实验共 66 组，包含按序组合、replay 状态、重复 slot 和缓存不可变性检查。
- `lint-final.log` 与 `format-final.log`：Ruff 通过，10 个 Python 文件格式通过。

真实 GDN 的旧 HYPIC 全流程专项和完整 engine 验证仍不在本轮结果内；不能将上述
block-native mock 检查扩大解释为所有模型、所有模式的端到端验收。

复现（在 `after/` 目录运行，before 必须是第一轮快照）：

```bash
python examples/offline_inference/hypic_compose_bench.py --before-root ../before \
  --segments 4 16 64 96 --iterations 10 --repeats 5
python examples/offline_inference/hypic_compose_bench.py --before-root ../before \
  --segments 4 16 64 96 --mode legacy --iterations 10 --repeats 5
```

默认后端继续为 torch。上表测量对象仅是状态组合；真实模型 prefill/TTFT 还包含
GDN、attention、投影和调度，不能把上表加速比直接视为整个请求的加速比。

## 7. 第三轮：GDN 调用优化（2026-09-24）

### 7.1 最终保留的实现

判定依据是实际 fresh GDN unit 数，而不是单纯的 cache hit 比例：

- 仅有最终 Query 需要执行（所有 Document 命中，且没有 fresh seam）：按请求顺序
  组合缓存 Document 的 S/T 得到 `H_doc`，直接执行一次 `GDN(Query, H_doc)`。
  不计算 Query 的 S/T，直接将这次 GDN 返回的最终状态写入 native decode 槽。
- legacy 的 Query-only 请求也走该路径，以零状态初始化，无需访问 segment 池。
- 任何 Document miss 或 fresh seam：保留原 packed 三遍 S/T/replay。block-native
  Document 仍以零状态 replay；legacy 仍以对应前缀状态 replay。未启用拆分 miss
  Document/Query 的实现。
- conv history 是否重置、命中后的 tail 恢复、seam 顺序、缓存哈希与槽归属不变。
  `max_prefill_units` 沿用保守的既有统计，不因这个优化放宽调度准入。
- 组合接口支持 0 个 fresh S/T，包括 cache-only 和空 Document 序列；Triton 对
  空序列返回零状态，不解引用空 fresh 张量。分组中可混合普通请求和快路径请求。
- 每层增加最多 8 项的 `cu_seqlens` LRU，key 为 device 和 CPU 可知的 unit lengths。
  只保存不可变 int32 边界，不保存模型输出；不读取 NPU `.item()` 来查找缓存。
  底层 `sgl-kernel-npu` 的 chunk metadata 按 Tensor 对象身份缓存，因此复用同一个
  Tensor 可避免一部分重复 metadata 构造和设备同步。不同布局必须使用不同条目。
- 及时释放 S/T 构造期间无用的 output，保留原有分组 workspace 生命周期约束。

这不是通用的“GDN 三遍全部变一遍”：只有无 fresh Document/seam 时跳过前两遍。
默认 torch 融合后端不变，不新增精度降级，不需要新增用户 flag。

### 7.2 尝试后未保留的方案

以下日志保存在服务器 `/data/hypic-gdn-pass-validation-20260924/`，避免只记录正结果：

1. 拆开 Document 和 Query，复用 block-native Document 的零状态输出：数值通过，
   但初版 cold/mixed 延迟变慢约 20–30%。复用 metadata 后仍存在部分回退。
   见 `bench-block-torch.log`、`bench-block-torch-v3.log`，故未对 miss 启用。
2. 单 Query 使用 `cu_seqlens=None`：当前依赖在 JIT 中报 `i_t is not defined`，
   见 `bench-block-torch-v2.log`。没有修改外部依赖，也没有启用这条路径。
3. 将 value 扩为 `[v, 0]`、初始状态扩为 `[0, I]`，一次 GDN 求 S/T：数学上可以
   沿 value 维独立计算，但本机实现未通过检查，Query 输出最大差异约 `2.20e-3`。
   见 `bench-block-torch-v4.log` 和 `gdn-combined-st-rejected.py`。根因尚未完成定位；
   不据此断言算法本身不成立。不放宽阈值，不在最终实现保留该代码。

### 7.3 测量口径与复现

使用 Ascend910B2，BF16 mock q/k/v、FP32 S/T，未下载模型。基线是第二轮优化完成后的
代码快照，不是最初的一行一 program 慢 kernel。当前轮目录：

```text
/data/hypic-gdn-pass-validation-20260924/
  before/     第二轮已验证代码
  after/      当前代码
  final-*.log 最终数值检查与延迟结果
```

`hypic_gdn_pass_bench.py` 同进程、同输入比较两版完整单层 `forward_hypic_gdn`，包含
conv、gating、S/T 构造、有序组合、replay 和 decode state 写回；不含模型投影、
attention、调度和生成。预热/JIT 不计时，5 轮各 3 次，交替两版计时顺序取中位数。
“cold”表示 PIC Document miss，不表示编译器或元数据冷启动；重复 layout 缓存已预热。
显存指标是输入/缓存预分配后、一次 GDN 调用的额外 allocator 峰值，不是模型总显存。

验证阈值未随优化放宽：输出 `atol=1e-5, rtol=0.02`；最终状态
`atol=2e-4, rtol=0.02`；缓存池 `atol=2e-5, rtol=0.002`。
benchmark 使用更慢的衰减 `A_log=-5`，并覆盖 ragged Query 与非顺序 native state 槽。

在 `after/` 内运行：

```bash
python examples/offline_inference/hypic_mock_validate.py --unit-tests
python examples/offline_inference/hypic_mock_validate.py --npu --compose-backend torch
python examples/offline_inference/hypic_mock_validate.py --npu \
  --compose-backend triton --compose-batch-size 4
python examples/offline_inference/hypic_gdn_pass_bench.py --before-root ../before \
  --backend torch --documents 4 16 --heterogeneous
python examples/offline_inference/hypic_gdn_pass_bench.py --before-root ../before \
  --backend triton --documents 4 16 --heterogeneous
python examples/offline_inference/hypic_gdn_pass_bench.py --before-root ../before \
  --backend torch --mode legacy --documents 4
python examples/offline_inference/hypic_gdn_pass_bench.py --before-root ../before \
  --backend triton --mode legacy-reset --documents 4 --heterogeneous
```

`--heterogeneous` 额外将 cache-only 与 fresh 请求混在一个 batch/group 中；`mixed`
则表示每个请求都含一个 compute-only 的新 Document。两者不能混为同一测试。

### 7.4 最终版本的实测延迟

以下为 block-native、每个 Document 512 tokens、Query 最大 9 tokens（batch 内
Query 长度依次缩短）、2 heads、K/V=128 的全命中结果。Triton 分组大小等于 batch。
这是完整单层 GDN 的毫秒数，不能换算为模型 TTFT 的同等加速。

| 后端 | Batch | Document 数 | 优化前 ms | 优化后 ms | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| torch | 1 | 4 | 13.618 | 5.011 | 2.72× |
| torch | 1 | 16 | 14.011 | 5.716 | 2.45× |
| torch | 4 | 4 | 55.275 | 25.884 | 2.14× |
| torch | 4 | 16 | 59.032 | 29.687 | 1.99× |
| Triton | 1 | 4 | 14.507 | 5.473 | 2.65× |
| Triton | 1 | 16 | 13.212 | 5.121 | 2.58× |
| Triton | 4 | 4 | 54.303 | 22.760 | 2.39× |
| Triton | 4 | 16 | 56.511 | 24.482 | 2.31× |

同一主实验中 cold 的加速比为 **1.04–1.17×**，主要来自元数据复用和 workspace
生命周期调整，并非减少了 miss 的 GDN pass。全命中每请求调用数从 3 降为 1；
batch=4 从 12 降为 4；cold/mixed 调用数不变。

全命中额外 allocator 峰值：torch 从约 **1.343 MiB 降为 1.212 MiB**；Triton
group=4 从约 **2.293 MiB 降为 1.663 MiB**。这不包含已有 segment pool、模型
权重或其他层的内存，不能当成整个 engine 的显存节省。

### 7.5 最终验证记录与边界

- `final-unit-v2.log`：**85 passed**；`final-control.log`：**23 passed**；
  `final-adapter.log`：**16 passed**，合计 **124 项回归**。
- `final-training-torch.log`、`final-training-triton.log`：各 **10 项**真实 NPU
  mock 检查通过，包含 bs=1/4 的 cold/warm/reorder/changed-prefix GDN 以及
  CANN attention。与训练原语对比，GDN output 最大差异 `9.54e-6`，最终状态最大
  差异 `1.04e-4`，使用既有容差。
- `final-block-torch.log` / `final-block-triton.log`：20 + 20 组；
  `final-legacy-torch.log` / `final-legacy-reset-triton.log`：8 + 10 组；
  `final-long-query-torch.log` / `final-short-query-triton.log`：8 + 8 组；
  `final-query-only-triton.log` / `final-legacy-seam0-torch.log`：8 + 8 组。
  共 **90 组**前后对照通过，output 最大差异 **0**，最终状态最大差异 **1.92e-4**。
  覆盖 2/8 heads、Document 129/512/1024 tokens、Query 最大 1/9/129 tokens、
  legacy seam=0/8、history reset 开/关、bs=1/4、混合请求与 compute-only miss。
- 主测试之外的复现参数分别为 `--documents 4 --doc-len 1024 --query-len 129 --heads 8`
  （torch）、`--documents 4 --doc-len 129 --query-len 1 --heads 8`（Triton）、
  `--mode legacy --documents 0`（Triton）、`--mode legacy --seam 0 --documents 4`（torch）。
- `final-unit.log` 的两项失败来自新增测试夹具：它强制 miss 后错误地允许覆盖 ready
  槽，与 `PicCacheCoordinator.prepare` 的“不覆盖已有 key”规则相违背。已改为
  compute-only miss，保留混合批次场景和原断言；修正后 85 项全部通过。
- `final-lint.log`：本轮 6 个 Python 文件 Ruff 通过；运行时文件与本地 SHA256
  一致。未在本地执行数值测试，未下载真实模型，未执行完整 engine 或生成质量评测。

快路径的最终状态来自 Query 的实际 kernel，而不是 Query 的 S/T 重构；数学表达
等价，BF16 内部计算可能带来不同舍入，因此并不承诺最终状态逐位一致。90 组实测
output 一致，也不能推广为所有输入、所有层和所有后续 decode 步骤逐位一致。
真实模型 TTFT、变长布局持续抖动时的 metadata 缓存收益，以及训练后模型的任务
准确率，仍需要后续模型级验证。当前仅交付上述验证范围内的实现。
