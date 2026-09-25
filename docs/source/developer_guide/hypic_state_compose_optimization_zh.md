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

## 8. FP32 Cube 实验（2026-09-25）

### 8.1 结论与交付范围

**长 segment 链存在收益，但目前不适合替换生产 Vector 路径。** 本轮只新增
`examples/offline_inference/hypic_compose_cube.py` 和 `hypic_cube_probe.py` 实验入口，
不新增生产 backend，不改变 `HypicConfig` 默认值，不修改缓存池、调度器、seam 或
conv history 语义。实验脚本仅在自己的进程内临时替换 composition launcher。

服务器为 Ascend 910B2；torch 2.10.0、torch-npu 2.10.0、triton-ascend 包 3.2.2、
sgl-kernel-npu 2026.9.0。所有数值测试均在服务器使用 mock 张量；未下载模型，
未在本地运行数值测试。Vector 基线来自提交 `89bc2e750`，前后快照的
`compose_triton.py` SHA256 均为
`9008eee2a32ede3a8b6221e72693943494c5a1f1e1af58e1975bcd7bb693fed8`。

### 8.2 尝试了哪些实现

所有候选仍按原顺序计算 `H_i = H_(i-1) @ T_i + S_i`，首段直接取 `S_0`；
输入、状态与累加器保持 FP32，`tl.dot(input_precision="ieee")`，未采用
BF16/TF32/HF32 近似或改变段的组合顺序。编译产物含 `linalg.matmul`、
`mix_mode="mix"`，诊断 kernel 的二进制符号同时含 `_mix_aic` 和 `_mix_aiv`，
确实涉及 Cube/Vector 混合执行，不是给 Vector reduction 改名。

| 候选 | 本轮观察 | 处理 |
| --- | --- | --- |
| 原缓存池直接读取、单 kernel 融合 | 16 行 tile 的单位矩阵检查有 75% 元素错误，最大绝对误差 1 | 拒绝进入生产 |
| 缓存指针表 + 每段独立 kernel | 同样未通过单位矩阵检查 | 拒绝进入生产 |
| 每段直接传 S/T 张量参数 | 精度检查通过，但 batch=4、64 段为 53.02 ms，Vector 为 7.63 ms | 仅作正确性对照 |
| 连续打包 S/T + 128 行 tile 融合 | 已测范围内正确，64 段有收益；短链较慢，临时显存较大 | 保留实验入口 |

最小复现进一步区分了“直接张量参数”“同一 allocation 内按段索引”、动态循环
和静态展开。两段输入的部分 tile 会失败，较多段或 128 行 tile 可以通过。
关闭 tightly-coupled buffer reuse、关闭 auto-bind sub-block、启用 Vector ND2NZ
三种独立选项均未修复该最小复现。直接缓存池版本增大到 128 行 tile 又遇到
UB 编译溢出：需要 3,145,728 bits，可用 1,572,864 bits。

这些是当前编译/执行栈的可复现现象，尚未定位到具体编译 pass；不能解释为
普通舍入误差，也不能宣称已修复通用的 FP32 Cube 支持。失败候选保留用于诊断，
不会在精度失败后静默回退并继续发布性能结果。

### 8.3 打包融合候选与延迟

`--cube-variant packed --cube-rows 128` 先按请求的实际 segment 顺序，分别
`torch.stack` 对应 S/T（命中时取 pool，miss 时取 fresh）。每请求启动一个
持久 kernel，每个 head 独立计算；请求之间仍由 host 逐一提交。最终状态与
replay 按原位置写回。不要求各请求有相同段数。

block-native 仅末段需要 replay 时，历史缓冲只保留两个循环槽；legacy 多段
需要 replay 时保留完整前缀状态。显式释放上一请求的 packed 输入和历史缓冲，
避免它们跨越下一请求的打包分配。**这仍不是现有静态 segment pool 的零拷贝路径**，
临时打包内存尚未纳入生产池预算，因此不自动接入 engine。

下表为同进程、相同输入、轮换计时顺序的最终实测：8 heads、K=V=128、
block-native warm，batch 内第 r 个请求的 Document 数为 `max(1, N-r)`，
另有一个 fresh Query。每组 5 次计时、每次 10 次调用，取中位数，排除 JIT。
延迟包含校验、Python 调度、S/T 打包、分配、状态组合和 replay 写回；不是纯 Cube
指令耗时，也不是完整模型 TTFT。对照列是本轮测量，不混用上一日的结果。

| Batch | Document 数 N | Vector ms | Packed Cube ms | Vector / Cube | Cube 额外峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.469 | 0.572 | 0.82× | 7.50 |
| 1 | 16 | 0.796 | 0.736 | 1.08× | 20.50 |
| 1 | 64 | 2.283 | 1.279 | 1.79× | 98.00 |
| 4 | 4 | 0.593 | 1.840 | 0.32× | 7.50 |
| 4 | 16 | 1.963 | 2.420 | 0.81× | 20.50 |
| 4 | 64 | 7.427 | 4.837 | 1.54× | 98.00 |

峰值为输入池分配之后单次调用的额外 allocator 峰值，包含打包与内部 workspace，
不是模型总显存。仅凭上述数据不能制定通用的 16/64 段 backend 切换阈值；
不同 head 数、形状、请求分组和内存预算仍需验证。

### 8.4 复现与后续方向

最终候选验证结果：

- 主组合对照共 **26 组**：6 组 warm、8 组 cold/mixed、12 组 legacy；最终状态
  和 replay 在这些输入上相对 FP32 参考的最大绝对差均为 **0**。覆盖 bs=1/4、
  非交换 T、不同顺序、变长段数、重复调用和缓存池不被修改。
- `cube-packed-edges.log`：4 组空序列/cache-only 边界、单位矩阵检查、
  1 Document + Query 对照通过。零 fresh S/T 的 GDN 快路径仍可使用该实验 launcher。
- `cube-packed-training.log`：**10 项 NPU mock 检查通过**，其中 8 项 GDN 覆盖
  bs=1/4 的 cold/warm/reordered/changed-prefix，另有 2 项 attention 检查。
  GDN 对训练原语的 output 最大绝对差为 **9.54e-6**，最终状态为 **1.04e-4**，
  使用既有容差，未放宽精度要求。它不是模型权重/完整 engine 的端到端验证。
- `cube-existing-unit.log`：服务器上的原有 tensor/metadata 回归 **85 passed**；
  两个新脚本 Ruff 检查、格式检查以及 `git diff --check` 通过。

远端原始记录保留在 `/data/hypic-cube-validation-20260925/`，其中 `before/`
和 `after/` 保存对照快照；最终性能日志为 `cube-packed-final.log`。
`cube-packed-mixed.log`、`cube-packed-legacy.log`、`cube-packed-training.log`
分别记录混合命中、legacy replay 和训练原语对照。

在服务器 `after/` 中运行：

```bash
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant packed --cube-rows 128 --before-root ../before \
  --batch-sizes 1 4 --segments 4 16 64 --cases warm --iterations 10 --repeats 5
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant packed --cube-rows 128 --before-root ../before \
  --batch-sizes 1 4 --segments 4 16 --cases cold mixed
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant packed --cube-rows 128 --before-root ../before --mode legacy \
  --batch-sizes 1 4 --segments 4 16 --cases cold mixed warm
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant packed --cube-rows 128 --cube-target mock \
  --npu --compose-backend triton --compose-batch-size 4
python examples/offline_inference/hypic_cube_probe.py --segments 2 3 4 --rows 16 32 64 128
```

最后一条是诊断矩阵，预期包含已知失败并输出 `failures`，不是全通过测试；
composition benchmark 则遇到精度失败立即退出，不给失败配置计时。误差阈值沿用
原 benchmark：单位矩阵要求精确一致，随机最终状态/replay 为 `atol=2e-5, rtol=2e-4`。

下一步值得做的是修复直接 pool 访问的混合 kernel 布局问题，或实现有界 tile
打包及跨请求提交，减少打包、workspace 与 host launch 成本，而不是降低输入精度
掩盖问题。在这些条件满足并完成模型级测试前，继续使用现有 Vector 实现。

### 8.5 短 segment 链补测（2026-09-25）

固定同一 packed Cube 候选、128 行 tile、FP32、8 heads、K=V=128，在同一台
910B2 上补测 N=1/2/4/8/16、batch=1/4、warm/cold/mixed，共 30 组。
每组 7 次计时、每次 20 次调用，轮换后端计时顺序并取中位数；不含 JIT，
包含打包、分配、组合与 replay 写回。没有改变生产实现或降低精度。

下表为 warm 结果，单位 ms：

| Document 数 N | Batch=1 Vector | Batch=1 Cube | Batch=4 Vector | Batch=4 Cube |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452 | 0.537 | 0.538 | 1.780 |
| 2 | 0.443 | 0.510 | 0.593 | 1.972 |
| 4 | 0.448 | 0.547 | 0.591 | 1.925 |
| 8 | 0.520 | 0.598 | 1.033 | 2.051 |
| 16 | 0.768 | 0.701 | 1.948 | 2.556 |

沿用上一节的 ragged 输入口径：第 r 个请求实际 Document 数为 `max(1, N-r)`，
各请求再加一个 fresh Query；因此 N 是批内最大 Document 数，不是统一的总段数。
这里的“短”指状态组合链的段数，不是完整 prompt token 长度，也不是完整模型
prefill 延迟。每段在本测试中已表示为 S/T，没有执行对应的 token prefill。

初轮 30 组全部通过，最终状态及 replay 对 FP32 参考的最大绝对差均为 0。
单位矩阵和空序列/cache-only 边界检查也通过。warm 下，batch=1 的 N<=8
Cube 慢约 15%–22%；N=16 约快 1.10×。batch=4 的所有这些短链都比 Vector 慢，
N<=4 约慢至 3.3×，N=8 约 2.0×，N=16 约 1.31×；cold/mixed 趋势相同。

这支持短链继续使用 Vector，而不是把所有请求无条件改成 Cube。当前 packed
版本仍有每请求的打包与 host 提交，短链的矩乘工作量不足以抵消这些成本；
这只是结合实现结构的解释，尚未通过 profiler 将各项开销单独定量拆分。

原始日志：`/data/hypic-cube-validation-20260925/cube-short-20260925-a.log`。
另以 `--seed 20260925 --cases warm` 复测 10 组，记录于同目录
`cube-short-20260925-b.log`；同样全部通过、最大绝对差 0，合计 40 组。
复测的 batch=1、N=16 加速比为 1.085×；batch=4、N=16 仍慢 1.37×，
短链回退趋势没有改变。上表保留初轮独立计时，不把两轮中较好的数值混在一起。

```bash
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant packed --cube-rows 128 --before-root ../before \
  --batch-sizes 1 4 --segments 1 2 4 8 16 --cases warm cold mixed \
  --iterations 20 --repeats 7
```

## 9. 跨请求批量 Cube 优化（2026-09-25）

### 9.1 实现与语义

本轮新增实验模块 `examples/offline_inference/hypic_cube_batch.py`，仍未注册
生产 backend、修改默认配置或接入正式 engine。`hypic_compose_cube.py` 新增
`batch`、`native`、`window` 三个候选；原有 `packed` 保持逐请求基线。

- `batch`：batch 内所有 S/T 一次性 stack，然后一次 Cube kernel 和一次
  Vector scatter。减少逐请求提交，但大 stack 的临时内存仍较大。
- `native`：用 CANN `index_select` 从原缓存池批量取 S/T，再补入 fresh 段。
  多个 fresh 段按请求合并 `index_copy_`，避免逐段复制；单 fresh 段仍直接 copy。
  空缓存池不执行非法 index_select，而是分配后完整填入 fresh/padding。
- `window`：按固定段数窗口打包，以前一窗口的最终状态作为下一窗口的合成首段 S。
  已测窗口 16 可以限制工作集，但增加提交次数，性能不如原 packed，保留作诊断。

Cube grid 是 request × head，单 program 处理完整 128×128 FP32 状态。
不同长度的请求采用**前置** `(S=0, T=I)` 补齐，使实际末段在 batch 内对齐。
补齐不改变 affine 组合的数学语义；所有实际段保持原顺序，replay 仍对应其入段状态。
仅末段需 replay 时保留两个循环历史槽；存在更早 replay 目标时保留完整前缀状态。
实现目前显式要求 K=V=128，不推断它对其他形状也成立。

没有使用 BF16/TF32/HF32 近似，没有树形重排，没有让 miss 改用不同的 GDN 语义。
动态 segment pool、history reset、seam 和调度器实现均未修改。

### 9.2 先定位打包瓶颈，再选择搬运实现

最初的自写 Triton gather 精度正确，但非常慢。NPU event 分阶段计时，在
BS=4、64 Documents、8 heads、K/V=128 下测得：

| 阶段 | 设备计时 ms |
| --- | ---: |
| Triton gather | 32.882 |
| Cube chain | 0.838 |
| 结果/replay scatter | 0.0044 |

增加搬运 tile 到 8192、将 source 分支改为直接地址并提示对齐后，gather 仍为
32.879 ms。这说明本候选的主要瓶颈在搬运而不是 Cube 矩乘，不能以更高 Cube
峰值算力解释或解决。尚未定位该 gather 低效的具体编译 pass，不声称已修复它。
最终 `native` 候选使用 CANN 搬运，不使用这条慢 gather。

复现诊断用 `hypic_cube_profile.py`；它刻意调用 `pack_mode="gather"`，测的是
被拒绝的搬运候选，不是最终 native 的耗时分解。event 计时排除 JIT，不能代替
含 Python、元数据上传和分配的整次调用延迟。

### 9.3 测试口径

benchmark 新增 `packed_ms`、`packed_workspace_bytes`、`before_workspace_bytes`，
在相同输入、同一进程里轮换测量 Torch、现有 Vector、原逐请求 packed 和新候选；
各后端都先通过最终状态和 replay 精度检查。新候选的结果不能只与更慢的失败
gather 比较，必须与原 packed 和现有 Vector 分别比较。

新增 `tests/ut/hypic/test_cube_batch_npu.py`，覆盖空缓存池、空请求、变长 batch、
重复来源、非顺序 replay 目标、1/2 heads、三次连续调用、输入和缓存池不被修改。
随机 T 非交换；沿用 `atol=2e-5, rtol=2e-4`，未放宽容差。

本轮所有数值测试在同一 Ascend 服务器运行，没有下载模型或本地推理。原始日志
仍位于 `/data/hypic-cube-validation-20260925/`；`cube-batch-first.log`、
`cube-batch-tile8192.log`、`cube-batch-stage8192.log`、
`cube-batch-stage-pointers.log` 保留失败/低效尝试。
`cube-native-v2-*.log` 记录包含 fresh 合并写入的最终 native 候选。

```bash
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant native --cube-rows 128 --before-root ../before \
  --batch-sizes 1 4 --segments 1 2 4 8 16 64 --cases warm --iterations 10 --repeats 5
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant native --cube-rows 128 --before-root ../before \
  --batch-sizes 1 4 --segments 4 16 --cases cold mixed --iterations 5 --repeats 5
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant native --cube-rows 128 --before-root ../before --mode legacy \
  --batch-sizes 1 4 --segments 4 16 --cases cold mixed warm
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant native --cube-rows 128 --cube-target mock \
  --npu --compose-backend triton --compose-batch-size 4
```

### 9.4 最终 native 候选的收益与边界

`cube-native-v2-warm.log`：同进程、相同输入，5 轮、每轮 10 次，取中位数，
排除 JIT，包含打包、元数据上传、分配及写回。8 heads，K=V=128，沿用前述
`max(1, N-r)` 个 Documents 加一个 Query 的 ragged 配置。
以下是 batch=4、warm 的结果，单位 ms：

| 最大 Document 数 N | 现有 Vector | 原逐请求 packed Cube | 新 native 批量 Cube | 相对原 Cube 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.563 | 1.856 | 1.168 | 1.59× |
| 2 | 0.568 | 1.838 | 1.476 | 1.25× |
| 4 | 0.610 | 1.923 | 1.511 | 1.27× |
| 8 | 1.057 | 2.106 | 1.598 | 1.32× |
| 16 | 1.971 | 2.483 | 1.809 | 1.37× |
| 64 | 7.446 | 4.959 | 3.096 | 1.60× |

因此本次确实减少了多请求 Cube 的开销，但 **N<=8 仍不如 Vector**；N=16 仅比
Vector 快约 1.09×，N=64 快约 2.41×，不能把这些倍数当作模型 TTFT 加速比。
batch=1、N=64 从原 packed 的 1.385 ms 变为 1.400 ms，基本持平；batch=1
短链则有回退，例如 N=4 从 0.593 ms 变为 0.917 ms。

miss 较多时并非通用优化：batch=4、N=16，cold 的原 packed/新 native 为
2.643/4.060 ms，mixed 为 2.692/3.290 ms。合并 fresh 写入保证了相同语义，
但索引上传、池 gather、补写成本仍在，不能仅因 warm 变快就启用所有请求。

额外 allocator 峰值也有代价：

| 配置 | Vector MiB | 原 packed MiB | 新 native MiB |
| --- | ---: | ---: | ---: |
| Batch=1、N=64、warm | 约 1.0 | 98.0 | 69.0 |
| Batch=4、N=64、warm | 约 4.0 | 98.0 | 280.0 |

新方案同时打包整个 batch；原 packed 逐请求执行。不能忽略这一区别或把 280 MiB
算作输入缓存已占的内存。上述峰值是输入池分配后的单次调用额外峰值，生产
segment pool 显存预算尚未为它预留空间。

结论：保留原 Vector/packed 路径和显式实验入口，不默认切换后端，不增加未经验证的
自动段数阈值。下一步应优先减少/消除 batch 打包内存，并建立带显存上限的请求分组，
再评估完整 engine 与模型级收益。本轮代码没有下载模型、运行 LongBench/MCPAgentBench
或改动它们的配置。

### 9.5 最终验证记录

- `cube-native-v2-warm.log` / `mixed.log` / `legacy.log`：12 + 8 + 12，共
  **32 组**前后对照通过，最终状态与 replay 相对 FP32 参考的最大绝对差均为 **0**。
- `cube-native-v2-training.log`：**10 项**真实 NPU mock 检查通过，包含 bs=1/4
  的 cold/warm/reordered/changed-prefix GDN 和 attention；GDN output 最大绝对差
  **9.54e-6**，最终状态 **1.04e-4**，沿用既有阈值。
- `cube-native-v2-unit-scoped.log`：原有 tensor/metadata 85 项与新增 NPU 回归
  8 项，合计 **93 passed**。新增回归覆盖 `batch` 的 torch pack 和 `native`
  两种搬运路径，检查重复调用和所有输入张量不变。
- `cube-native-v2-unit.log` 是一次扩大到整个测试目录的收集失败记录：
  `test_pic_adapter.py` 需要完整 `vllm` 包，而当前 mock venv 未安装该包。
  后续只运行明确可在该环境执行的上述 4 个测试文件，没有把收集失败标作通过；
  未重新构建完整 vLLM engine。
- 5 个本轮 Python 文件 Ruff 检查、格式检查通过，`git diff --check` 通过。
  服务器与本地实验实现/测试的 SHA256 已核对一致。

因此本轮交付是经过 NPU 算子/缓存 plumbing 验证的实验实现与性能记录，
不是所有场景更快、显存无额外成本或完整模型准确率已验证的生产 backend。

## 10. 直接索引 Cube：省去 S/T 打包

本节是第 9 节之后的进一步实验，入口是
`hypic_compose_cube.py --cube-variant indexed --cube-rows 128`。
代码仍位于 examples，**没有替换生产 Vector 内核或改变默认后端**。

### 10.1 实现与语义

- Host 把 hit 的 pool slot 和 miss 的 fresh unit 转为 S/T 原始张量地址，
  上传一个包含地址、请求长度和 replay 目标的 int64 packet。
  每请求的基础地址只查询一次，不再按段重复调用 `data_ptr()`。
- Cube grid 仍为 request × head，每个 program 处理完整 128×128 状态；
  直接读取 packet 指向的 S/T，**没有 S/T gather、stack、index_copy 或 batch 打包**。
  kernel 内不再区分 hit/miss，仅保留有序递推 `H_i = H_(i-1) @ T_i + S_i`。
- 按每个请求的实际段数循环，不添加 identity padding，不把段改成树形归并，
  不改变非交换 T 的顺序。首段仍直接取 S，空序列结果为零。
- 只有末段需要 replay 时保留两个 history 槽；中间段需要 replay 时保留全部前缀。
  单独的 Vector scatter 按实际长度取出 final 和各段的**输入状态**，支持非顺序目标。
- 当且仅当所有 fresh replay 单元都会被写入时，使用 `empty_like` 省去清零；
  未被 replay 指定的单元依旧初始化为零。没有改变 block-native 的 fresh Document 语义。
- FP32 输入、累加与 IEEE dot 不变，没有 TF32/BF16 降精度，也没有放宽误差阈值。

空请求的第一条地址指向本次调用持有的零张量，避免读取空 pool/fresh tensor。
这不是有意义的额外 segment：循环仍为零段，final 读取 H0。
直接在 kernel 中以 `if count > 0` 初始化整块 state 的版本触发了 UB overflow：
编译器报告需要 2097152 bits、可用 1572864 bits。失败日志为
`cube-indexed-ragged.log`，保留在服务器；最终版本去掉该初始化分支后通过验证。
这是对当前编译器行为的实测规避，不声称修复了编译器本身。

### 10.2 同进程前后延迟与显存

服务器目录仍为 `/data/hypic-cube-validation-20260925/after`。
优化前模块在修改前保存为 `examples/offline_inference/hypic_cube_native_v2.py`，
SHA256 为 `b647d9be36a2a00cd6960e54b7a7576335f1cc7ef03ad7cc9abe703b14c5277d`。
新增 `--cube-baseline-module` 可以导入这个冻结副本，结果中的 `native_ms` /
`native_workspace_bytes` 专门记录上一轮 native V2，而不是与历史日志跨进程比速度。

`cube-indexed-final.log`：固定 seed=20260924，8 heads、K=V=128；
同进程、同输入，轮换后端顺序，每轮 10 次、5 轮中位数，排除 JIT，
包含验证包装、Host 地址处理、上传、分配、Cube 与 scatter。
N 为最大 Document 数；请求 r 包含 `max(1, N-r)` 个 Document 和一个 Query，
不是所有请求等长，也不是 prompt token 数。

BS=4、warm（Documents 全命中），单位 ms：

| N | 现有 Vector | 上轮 native V2 | 本轮 indexed | 相对上轮加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.566 | 1.167 | 0.713 | 1.64× |
| 2 | 0.558 | 1.459 | 0.698 | 2.09× |
| 4 | 0.655 | 1.640 | 0.798 | 2.06× |
| 8 | 1.065 | 1.733 | 0.855 | 2.03× |
| 16 | 1.978 | 1.862 | 1.024 | 1.82× |
| 64 | 7.435 | 3.041 | 2.045 | 1.49× |

BS=4、N=64、warm 的额外 allocator 峰值从 **280.01 MiB 降到 12.01 MiB**，
减少约 95.7%；现有 Vector 仍约 4 MiB。indexed 不是零 workspace：
仍包含 replay/final、history、元数据及 allocator 计入的内部工作空间。
BS=1、N=64、warm 从 1.294 ms / 69 MiB 变为 0.957 ms / 5 MiB。

cold/mixed 不再需要先 gather 缓存再补写 fresh，因此也有收益：

| BS | N | 场景 | Vector ms | native V2 ms | indexed ms | native / indexed MiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 16 | cold | 1.999 | 3.731 | 1.095 | 118.01 / 43.00 |
| 4 | 16 | mixed | 2.020 | 3.214 | 1.159 | 102.01 / 26.01 |
| 4 | 64 | cold | 7.493 | 11.022 | 2.164 | 405.51 / 139.50 |
| 4 | 64 | mixed | 7.470 | 7.259 | 2.099 | 342.51 / 75.01 |

cold 的 replay 输出随 fresh 单元数增加，legacy 中间 replay 还要求保存前缀，
不能把 warm 的 12 MiB 推广到所有场景。短链仍有边界：BS=4、N<=4 和
BS=1、N<=8 的 indexed 尚不如现有 Vector；例如 BS=1、N=4 为 0.587 对 0.465 ms。
没有据此引入未经引擎验证的自动阈值。

复现最终 block-native 对照：

```bash
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant indexed --cube-rows 128 \
  --cube-baseline-module hypic_cube_native_v2 --before-root ../before \
  --batch-sizes 1 4 --segments 1 2 4 8 16 64 --cases warm cold mixed \
  --iterations 10 --repeats 5
```

没有冻结模块时可省略 `--cube-baseline-module`，仍会与原逐请求 packed 和
`--before-root` 指定的生产 Vector 作同输入比较。不要把 indexed 自身作为 native 基线。

### 10.3 验证范围与剩余限制

- `cube-indexed-final.log`：36 组 block-native 对照，覆盖 BS=1/4，
  N=1/2/4/8/16/64，cold/mixed/warm，final/replay 最大绝对差均为 0。
- `cube-indexed-final-legacy.log`：24 组 legacy 对照，覆盖 BS=1/4，
  N=1/4/16/64，三种命中模式，最大绝对差均为 0。
- `cube-indexed-final-training.log`：10 项真实 NPU mock GDN/attention 检查通过；
  GDN output 最大绝对差 9.54e-6，最终状态 1.04e-4，与上轮相同阈值。
- 新增 indexed 回归覆盖空请求、空 pool、重复命中、非顺序 replay、
  1/2 heads、1/2/3/4/16/65 段、两槽 ring 奇偶边界、全部/末段 replay、
  三次重复调用、非交换随机 T 和全部输入不可变。
- `cube-indexed-final-unit.log`：4 个指定测试文件合计 **109 passed**，
  包含原有 85 项与本实验 24 项；没有把缺少完整 vLLM 包的整目录收集标作通过。
  本轮修改的 4 个 Python 文件通过 Ruff 检查/格式检查，`git diff --check` 通过。

当前没有 full engine 构建/并发请求、跨 stream、图捕获或模型权重准确率测试。
地址 packet 指向的输入张量必须在异步 kernel 使用期间有效，不能在其他 stream
复用或驱逐对应 pool slot；生产接入仍需结合现有调度器 pin/reserve 生命周期，
并为 history/replay 预算显存。例：BS=4、N=64、legacy cold 的额外峰值仍为
266.51 MiB（native V2 为 526.51 MiB），不能把它视为已解决所有显存预算问题。

所有数值验证均在 Ascend 上使用 mock tensors/weights，无模型下载，无本地推理。
以上是状态组合调用延迟，不是整模型 TTFT；保留原 Vector、packed、native 对照，
后续生产接入需单独验证，不能把实验入口等同于默认推理路径已加速。

## 11. 状态组合直接输出：去掉 history/scatter

先将第 8–10 节的实验、测试和文档提交为 `ec6ef1966`
（`perf(hypic): add indexed FP32 Cube composition experiments`，带 sign-off）。
本节以该提交的 indexed 路径为优化前基线，不再以上一轮较慢的 native 打包版本为基线。
仍只修改实验入口，生产后端、模型和调度策略保持不变。

### 11.1 实现

新增 `--cube-variant output --cube-rows 128`：

- 继续通过地址 packet 直接读取原 S/T，保持原顺序和 FP32 IEEE 计算。
- packet 的第三项由 replay 索引改为目标地址；地址 0 表示不需要 replay。
  kernel 不解引用 0，只在存在目标时写入当前的输入状态。
- 初始 replay 写 H0=0；后续在第 i 段计算前写 H_(i-1)，最终直接写 H_N。
  非顺序 replay 目标与空请求语义不变，输入和缓存池只读。
- 删除该路径的 history 张量与单独 scatter launch；即使需要中间 replay，
  也不会保存一整份 N+1 前缀 history。返回接口仍为原来的 replay 列表和 final。
- 未指定 replay 的单元继续清零；仅当每个单元都会被完整覆盖时才省略清零。

先验证了无条件写入废弃缓冲区的版本（`cube-output-first.log`），再验证条件写回
版本（`cube-output-sparse-first.log`）。后者通过编译和基础正确性检查，因此最终
候选**没有废弃缓冲区，也不写未请求的中间状态**。
此处的条件分支仅控制 store，不产生两个整矩阵初始化值之间的合流；
不能把它与第 10 节触发 UB 溢出的 state 初始化分支混为一谈。

### 11.2 与已提交 indexed 版本同进程比较

优化前服务器副本：`examples/offline_inference/hypic_cube_indexed_ec6ef1966.py`，
内容来自修改前保存的已提交文件，SHA256 为
`b618a504fe97827fcd5e9f67574188b0278fbf94c6a76331ac059c5080eeec06`。
新增 `--cube-indexed-baseline-module`，结果使用 `indexed_ms` 和
`indexed_workspace_bytes` 明确标识该基线，不覆盖原有 `native_ms` 含义。

`cube-output-final.log` 的配置沿用第 10 节：8 heads，K=V=128，seed=20260924，
BS=1/4、N=1/2/4/8/16/64、三种命中模式，5 轮 × 10 次中位数；
同输入、同进程轮换计时，排除 JIT，包含包装检查、元数据、分配和全部 kernel。
N 仍指最大 Document 数，每请求另有 Query，batch 仍为 ragged。

BS=4、warm，单位 ms：

| N | Vector | 已提交 indexed | 直接输出 output | 相对 indexed 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.576 | 0.735 | 0.554 | 1.33× |
| 2 | 0.596 | 0.759 | 0.574 | 1.32× |
| 4 | 0.637 | 0.783 | 0.597 | 1.31× |
| 8 | 1.061 | 0.816 | 0.650 | 1.26× |
| 16 | 1.981 | 0.993 | 0.833 | 1.19× |
| 64 | 7.439 | 2.031 | 1.870 | 1.09× |

BS=1、warm 的 N=1/4/16/64，indexed → output 分别为
0.623→0.451、0.581→0.421、0.717→0.538、0.975→0.810 ms。
收益主要是每次调用约 0.16–0.19 ms 的固定开销降低，不应据此声称矩阵乘法本身
获得相同比例加速。短链在本机这组形状下与 Vector 接近或略快，差距很小，
尚不足以支持跨设备、跨形状的自动切换策略。

BS=4、block-native 的额外 allocator 峰值（包含返回值及内部 workspace）：

| N | 场景 | indexed MiB | output MiB |
| ---: | --- | ---: | ---: |
| 16 | warm | 12.01 | 8.00 |
| 16 | cold | 43.00 | 39.00 |
| 64 | warm | 12.01 | 8.01 |
| 64 | cold | 139.01 | 135.01 |
| 64 | mixed | 74.51 | 70.51 |

warm 相比已提交版本进一步减少约三分之一。cold/mixed 的大量 replay 返回值仍然
需要存在；不能把所有额外内存都当作可删除的临时 workspace。

legacy 的中间 replay 原先要求完整 history，显存收益更明显。
`cube-output-final-legacy.log`，BS=4、N=64：

| 场景 | indexed ms | output ms | indexed MiB | output MiB |
| --- | ---: | ---: | ---: | ---: |
| cold | 2.141 | 1.987 | 266.01 | 134.01 |
| mixed | 2.150 | 1.999 | 202.51 | 70.51 |

两者都减少 132 MiB 的前缀 history；保留了所有请求指定的 replay，没有改变
评分、模型、命中策略或 history reset 语义。

复现命令（在原服务器目录执行）：

```bash
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant output --cube-rows 128 \
  --cube-indexed-baseline-module hypic_cube_indexed_ec6ef1966 \
  --before-root ../before --batch-sizes 1 4 --segments 1 2 4 8 16 64 \
  --cases warm cold mixed --iterations 10 --repeats 5
```

冻结模块可以从 `ec6ef1966:examples/offline_inference/hypic_cube_batch.py` 提取；
不要用修改后的 output 路径冒充优化前基线。

### 11.3 验证

- `cube-output-final.log`：36 组 block-native；`cube-output-final-legacy.log`：
  24 组 legacy。共 60 组同进程对照的 final/replay 最大绝对差均为 0。
- `cube-output-final-training.log`：10 项 NPU mock GDN/attention 检查通过，
  GDN output 最大绝对差 9.54e-6、最终状态 1.04e-4，阈值不变。
- NPU 回归将 output 加入空池/空请求、重复命中、非顺序 replay、长短链、
  稀疏/全部 replay、重复调用及输入不变检查；使用非交换随机 T。
- `cube-output-final-unit.log`：4 个指定测试文件共 **125 passed**，
  包含原有 85 项及本实验 40 项；不是缺少完整 vLLM 包的整个测试目录结果。
  本轮 4 个 Python 文件 Ruff 检查/格式检查及 `git diff --check` 均通过。
- `cube-output-second-seed.log`：seed=20260929，BS=4、warm、N=1/64 的
  两组复验均通过，最大绝对差 0；indexed → output 分别为
  0.703→0.536 ms、2.028→1.861 ms，收益方向与主实验一致。

保持 K=V=128 的实验限制；尚未验证完整 engine、跨 stream、图捕获和模型级准确率。
此次优化只涉及状态组合，不改 GDN 计算次数或请求调度，也不自动启用到生产路径。

## 12. 全缓存状态组合热路径

本节按“所有输入 segment 的 S/T 都已缓存，只需要最终组合状态”优化。
这对应无 fresh/seam/replay 单元的组合阶段，不包含后续 Query 的 GDN 计算。
它与第 11 节仍含一个 fresh Query 摘要的 warm 基准不同，**不能直接跨表比较绝对延迟**。

### 12.1 两项独立改动

1. `vllm_ascend/hypic/compose.py`：保留全部校验，但将 pool/fresh 单元数从逐段
   `len(tensor)` 改为每次调用/每请求读取一次 `shape[0]`。旧循环每个 segment
   重复调用三次 tensor 长度接口，包含 PyTorch dispatch、维度与 tracing 检查。
   新值只在当前调用中存活，不是跨请求缓存，不会复用旧的 pool 尺寸。
   这是本轮唯一的生产模块修改；原 Vector 内核和默认后端不变。
2. examples 中新增 `--cube-variant cached`：仅当 fresh 单元全部为 0、
   每个 source 都是缓存 slot、所有 replay 都为 -1 时使用专用 Cube kernel。
   直接传入原 S/T pool，上传 `[输出请求号, 实际长度, slot IDs...]` 的 int32 packet，
   不构造逐段 S/T/replay 地址，不读取 fresh，不保存 history，不判断 replay。
   slot 在设备端转 int64 后参与地址乘法，Host 校验 int32 元数据容量。
   含 miss、fresh 或 replay 的计划仍走通用 output 路径，不能强制视为命中。

只为非空请求启动 program；空请求的 final 初始化为零，全空 batch 不启动 Cube。
不存在用 slot 0 假装空段或解引用空 pool 的行为。各请求保持原有顺序，首段取 S，
后续仍为 FP32 IEEE `H = H @ T + S`，输入池保持只读。

BS=4、最大 64 段时，元数据由通用 output 的 6272 bytes 降到 1056 bytes。
没有引入跨请求持久化 workspace、缓存组合结果或 Graph 依赖；每次调用仍重新
构造元数据和返回值，计时包含这些开销。

### 12.2 同进程消融

服务器在修改前保存了两个对照：

- `examples/offline_inference/hypic_cube_output_before_cacheonly.py`：第 11 节的 output，
  SHA256 `d4df85cd43da7317d21c681df6d79e5c34bf0e2cb224fadad43028284e53519f`。
- `examples/offline_inference/hypic_compose_api_before_cacheonly.py`：旧校验包装，
  SHA256 `1e08771399310db356d23291ce7b69bd2284fe99da037fef923c1d8425b7178c`；
  对应 `ec6ef1966` 中的 `compose.py`。

`cube-cached-final.log`：同输入、同进程、轮换顺序，seed=20260924，8 heads、
K=V=128，5 轮 × 10 次中位数，排除 JIT。N 为最大缓存 segment 数，
请求 r 使用 `max(1, N-r)` 段，**不再追加 Query 段**。
下表是完整包装调用耗时，不是固定地址反复 launch 的设备下界，单位 ms：
每轮连续调用后同步，统计平均每次调用耗时并取各轮中位数；不是逐次同步的
孤立请求 RTT，也不是整模型 TTFT。

| BS | N | 旧校验 + output | 新校验 + output | 新校验 + cached |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.430 | 0.408 | 0.396 |
| 1 | 16 | 0.507 | 0.433 | 0.414 |
| 1 | 64 | 0.801 | 0.573 | 0.575 |
| 4 | 4 | 0.536 | 0.486 | 0.448 |
| 4 | 16 | 0.815 | 0.580 | 0.497 |
| 4 | 64 | 1.859 | 0.944 | 0.896 |

BS=4、64 段相对旧调用约 **2.08×**，但其中约 0.915 ms 来自校验热路径，
专用 kernel/元数据再减少约 0.048 ms，不能将全部收益归因于 Cube 计算变快。
BS=1、64 段的专用路径与“新校验 + 通用 output”基本持平，不声称所有形状都更快。
BS=4 的额外 allocator 峰值约 6 MiB，与同输入通用 output 基本一致；主要减少的是
Host/元数据开销，不是再省去一份大张量。

结果中的 `output_ms` 是旧校验+旧 kernel，`output_current_api_ms` 是新校验+旧 kernel，
`triton_ms` 是新校验+cached。`before_ms` 仍表示使用当前校验包装的原 Vector，
不能将它误认为修改前的完整 output 延迟。

```bash
python examples/offline_inference/hypic_compose_cube.py \
  --cube-variant cached --cube-rows 128 \
  --cube-output-baseline-module hypic_cube_output_before_cacheonly \
  --cube-output-api-module hypic_compose_api_before_cacheonly \
  --cache-only --cases warm --batch-sizes 1 4 --segments 1 2 4 8 16 64 \
  --iterations 10 --repeats 5 --before-root ../before
```

### 12.3 验证范围

- `cube-cached-final.log`：12 组全缓存对照的最终状态最大绝对差均为 0。
- `cube-cached-training.log`：10 项 NPU mock GDN/attention 检查通过，
  GDN output/state 最大绝对差仍为 9.54e-6 / 1.04e-4。
- 新增全缓存专用回归覆盖 0/1/2/4/16/65 段、1/8 heads、变长 batch、
  重复/重排 slot、全空和部分空请求、空 pool、三次调用及缓存只读；
  测试显式拒绝通用 output kernel，确保确实走了专用路径。
- 对 0/1 fresh 单元新增 source/replay 越界和重复 replay 拒绝检查，
  断言非法元数据不能进入 kernel；mixed/fresh/replay 回归继续验证通用分支。
- `cube-cached-unit.log`：4 个指定文件共 **163 passed**（原有 85 项、
  本实验 78 项），不是完整 vLLM engine 测试集。
- `cube-cached-second-seed.log`：seed=20261002、BS=4、N=4/16/64 的 3 组复验
  均通过，最终状态最大绝对差 0；64 段旧调用 1.877 ms、新调用 0.879 ms。
- 修改的 Python 文件通过 Ruff 检查/格式检查，`git diff --check` 通过。

所有数值验证均在 Ascend 上使用 mock 数据，没有下载模型、运行本地数值测试或
更改调度器的 ready/reserve 生命周期。Cube 专用路径仍是显式实验入口，不是默认
生产后端；将来接入实际请求仍需保证 pool slot 在异步计算结束前不能被复用。

### 12.4 诊断工具与后续空间

新增 `hypic_compose_opportunities.py`，分别测量完整调用、只校验、无包装调用、
预先持有输出/packet 的 launch，以及可选的静态 NPU Graph。
它用的是通用 output 的“缓存 Documents + 一个 fresh Query 摘要”诊断输入，
**不是本节全缓存测试的性能结论**。

`cube-opportunities-final.log` 中 4 组诊断均通过。Graph 验证会先将返回值填为 NaN，
防止沿用之前正确结果而误报成功；还验证了不重新捕获时修改 packet、S/T 地址、
segment 顺序及部分请求实际长度，结果仍正确。它不验证空请求 sentinel 生命周期，
也不等于完整 engine 的图捕获支持。

固定 packet/输出的结果排除了每请求准备成本；各计时项不能简单相加减，
NPU event span 也含 Host dispatch 间隙，不能将其称为纯 Cube 执行时间。
后续值得继续验证的是有明确所有者与容量上限的 metadata/workspace 复用；
不能跳过 ready/generation/pin 校验或持久保存可能已失效的裸 slot 地址。
