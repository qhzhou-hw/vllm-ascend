# Ascend C cached-only 状态组合

## 范围

本实现是 `hypic` 分支上的独立、显式启用算子，不修改模型 forward、缓存调度器、
conv history 或现有 `state_compose_backend` 默认值。

假设每个 segment 的状态摘要已经在缓存池中，按请求给定的 slot 顺序计算：

```text
H = 0
for slot in request_slots:
    H = H @ T_pool[slot] + S_pool[slot]
```

输入是有限值 FP32 张量。首段直接取 S，省略零矩阵乘法；空请求输出零。
输出仅为最终 H，不计算 S/T，不执行 seam、fresh token、replay 或 Query forward。
各段不重排；默认顺序入口不做树形归并，另有显式树形/混合实验入口，见第三至五轮。
所有入口均只读缓存池。

当前支持 Ascend910B、K=V=128、连续 FP32 的 `[slots, heads, 128, 128]` 缓存池。
batch、head 数和各请求 segment 数可变，支持重复 slot、任意合法 slot 顺序和空请求。
不支持的 dtype/shape/device 明确报错，不静默转换精度或回退到 Triton。

## 实现

- `csrc/hypic_compose_cached/kernel.cpp`：Ascend C Cube/Vector 混合 kernel。
  每个计算任务负责一个请求、一个 head 的 32、64 或 128 行，始终保留完整 128 列。
  行块之间独立，各行块在 kernel 内顺序遍历全部 segment。
- `csrc/hypic_compose_cached/binding.cpp`：PyTorch 自定义算子注册、Host tiling、
  输入检查、当前 NPU stream 上的启动。
- `tree_kernel.cpp` / `tree_single_kernel.cpp`：分层启动和单次启动的有序树形实验。
- `vllm_ascend/hypic/compose_ascendc.py`：非负 slot ID 接口及 prepared plan。

状态行块保存在 Vector 侧 UB；Matmul 高阶 API 使用 FP32 Cube 计算 `H @ T`，
输出到该 AIV 独享的 GM scratch，再搬回 UB，由 Vector Add 加入本段 S。
Cube 消费完旧状态后，复用其 UB 加载 S，只需 state/product 两份行块。
每次批量调用只启动一个 kernel，各请求链不逐段返回 Host。
Matmul 内部的 Cube/Vector 通信和暂存由 CANN 管理；不能将其描述为零 GM 搬运。
显式选择 `MatrixMadType::NORMAL`，不使用 HF32/BF16 近似。

Prepared plan 只保存经过检查的元数据和 tiling，**不缓存组合结果或池地址**。
每次调用读取当前传入的 S/T，每次分配独立输出与 CANN system workspace，
避免返回值被下一次调用覆盖及 workspace 并发复用问题。
plan 限制在创建它的 stream 上使用。回调只捕获启动地址和标量参数，
张量在调用线程释放；通过 `recordStream` 和 allocator 的 stream 顺序规则保护异步访问，
避免让任务队列中保留的回调持续持有整份工作区。

`tile_rows=0`（默认）按实际 Cube 核数选择：从 32 行开始，若
`batch * heads * (128 / tile_rows)` 超过 `2 * Cube核数`，则增大到 64，再到 128。
因此小 batch 保留更多并行行块，大 batch 减少任务轮数及 T 的重复读取。
这是减少任务轮数的确定性策略，不是运行时自动调优；可显式传入 32/64/128 做对照。

调用方仍负责 ready/generation/pin 检查，确保 slot 在异步计算完成前不被覆盖或驱逐。
本算子不会绕过或代替调度器的 slot 生命周期管理。

## 单独编译

仅在 Ascend 服务器编译；不需要模型权重、完整 vLLM 安装或安装整个自定义算子包。
先加载对应 CANN 环境，在已安装 torch/torch-npu 的 Python 环境中执行：

```bash
cmake -S csrc/hypic_compose_cached -B build/optimized_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0
cmake --build build/optimized_release -j 4
```

`SOC_VERSION` 和 CANN 路径必须匹配实际环境。使用干净的新 build 目录；
CANN 9.0 的旧版 CMake helper 从未指定 build type 切换到 Release 时，
可能收集两份 host object，导致重复符号。

## 调用

```python
import torch
import torch_npu

from vllm_ascend.hypic.compose_ascendc import CachedComposeAscendC, compose_cached_ascendc

torch.npu.set_device(0)
torch.ops.load_library("/absolute/path/build/optimized_release/libhypic_compose_ascendc.so")

# S_pool/T_pool 均为 [slots, heads, 128, 128]、NPU、连续 FP32。
slots = [[4, 1, 4], [], [0, 2]]
final = compose_cached_ascendc(S_pool, T_pool, slots)

# 相同形状和 slot 顺序重复调用时，可单独保存元数据准备结果。
plan = CachedComposeAscendC(S_pool, T_pool, slots)
final = plan(S_pool, T_pool)

# 仅用于调优对照；默认自动选择通常无需指定。
plan_64 = CachedComposeAscendC(S_pool, T_pool, slots, tile_rows=64)
```

这里使用真正的非负 pool slot ID，不使用 `ComposeStep.source` 的 `-(slot+1)` 编码。
修改原始 `slots` 列表不会修改已创建 plan；改变顺序需要新建 plan。
`torch.ops.hypic_ascendc.cached_compose` 是内部低层接口，调用者应使用上面的受检查入口。

## Mock 验证和计时

```bash
python examples/offline_inference/hypic_compose_ascendc_bench.py \
  --library build/optimized_release/libhypic_compose_ascendc.so \
  --tests --triton --batch-sizes 1 4 --segments 0 1 2 4 16 64 \
  --heads 8 --iterations 10 --repeats 5
```

无需下载任何模型。脚本直接加载实现文件，避免依赖完整 vLLM 初始化。
测试覆盖空 pool/请求、不同长度、重复/重排 slot、非交换矩阵、输入只读、
旧返回值不被覆盖、prepared plan 读取更新后的池内容及非法输入拒绝。
还覆盖 0/32/64/128 分块、超过工作核数的任务、重复调用的显存上限、
临时 plan 的异步生命周期和跨 stream 创建的 pool。
参考为关闭 HF32 的 FP32 `torch.bmm` 顺序组合，阈值 `atol=2e-5, rtol=2e-4`。

计时区分：

- `ascendc_auto_full_ms`：包括 plan 构建、检查、Host tiling、元数据传输、分配和执行。
- `ascendc_auto_prepared_ms`：排除 plan 构建，保留池检查、输出/workspace 分配和执行。
- `--tile-rows 32 64 128` 对应 `ascendc_r32_*` 等显式分块测量。
- `triton_vector_ms` / `triton_cube_cached_ms`：原有 launcher，包含其元数据构造与分配，
  不包含外层 `compose_fused_batch` 校验。

各项在相同输入上轮换顺序，预热后每轮连续调用、同步再取平均和多轮中位数。
它们不是纯 kernel 时间、不是孤立请求 RTT、不是模型 TTFT；prepared 数字不能直接
宣称为动态请求的完整加速比。

## 第一版验证记录（2026-11-03，历史对照）

实机：Ascend910B2C，CANN 9.0.0，torch 2.10.0+cpu / torch-npu 2.10.0，
Triton 导入版本 3.2.0。构建使用 GCC 12.3.1 和 CANN 自带 bisheng。
所有数值测试均在服务器使用 mock 张量，未下载模型，未在本地做数值验证。

服务器工作目录：`/data/hypic-ascendc-validation-20261103`。

- `build9.log`：独立扩展编译、链接通过。
- `smoke5.log`：0/1/2 段最小测试通过。
- `validation-final.log`：**37 passed**；12 组 benchmark 正确性检查通过。
  Ascend C 两种调用方式与 FP32 参考的最终状态最大绝对差均为 0。
  Triton Cube 同样为 0，Vector 最大约 1.64e-7；不要求不同归约次序逐 bit 相等。
- `validation-second-seed.log`：seed=20261104，BS=1/4、N=4/16/64 的 6 组复验通过，
  Ascend C 最终状态最大绝对差仍为 0。BS=4、64 段完整调用/prepared 为 0.854/0.701 ms。
- 六个实现/测试/构建文件的本地与服务器 SHA256 逐一一致；Python Ruff 检查及格式检查通过。
  这是算子级回归，不是整个 vLLM-Ascend 测试集。

8 heads、K=V=128、seed=20261103。N 为最大段数，请求 r 使用 `max(0, N-r)` 段，
slot 为 `(i*3+r) % max(1,N)`，不附加 Query。单位 ms：

| BS | N | Triton Vector launcher | Triton Cube cached launcher | Ascend C 完整调用 | Ascend C prepared |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.107 | 0.015 | 0.150 | 0.084 |
| 1 | 1 | 0.112 | 0.092 | 0.171 | 0.087 |
| 1 | 2 | 0.130 | 0.079 | 0.185 | 0.078 |
| 1 | 4 | 0.207 | 0.082 | 0.156 | 0.101 |
| 1 | 16 | 0.641 | 0.139 | 0.211 | 0.113 |
| 1 | 64 | 2.382 | 0.356 | 0.397 | 0.260 |
| 4 | 0 | 0.141 | 0.021 | 0.148 | 0.086 |
| 4 | 1 | 0.143 | 0.096 | 0.158 | 0.075 |
| 4 | 2 | 0.173 | 0.101 | 0.167 | 0.087 |
| 4 | 4 | 0.316 | 0.102 | 0.167 | 0.087 |
| 4 | 16 | 1.905 | 0.193 | 0.307 | 0.175 |
| 4 | 64 | 8.262 | 0.586 | 0.815 | 0.702 |

**结论：cached-only Ascend C 已实现并通过 mock 正确性验证，但首版不是全面优于 Triton Cube。**
长链相对 Vector 有明显收益；BS=4、64 段时，Ascend C prepared 仍慢于原 Cube launcher。
这里两个后端的 Host 工作量不同，不能把表中比值称为纯 kernel 加速。
第一版的 32 行切分增加了长链任务数，且每段使用高阶 Matmul 的 Cube/Vector 通信；
这些是下一步需要 profiler 验证的优化方向，而不是本轮已证实的瓶颈归因。

第一版 CANN system workspace 为每次调用 16 MiB；不随 segment 数线性增长。
第二轮压力测试发现，第一版捕获 Tensor 的回调会持续保留这些分配，
因此本节短时性能记录不代表其具备长期稳定运行能力；见后文修复说明。
全空请求仍启动专用 kernel 输出零，
而 Triton cached 路径可以直接返回零张量，因此空链不应宣称有收益。
尚未进行完整模型、LongBench 或 MCPAgentBench 的集成回归，也没有切换生产后端。

## CANN 9.0 接入注意事项

1. 旧版 `SetSysWorkspace` 不足以初始化 kernel-launch 使用的 KFC 地址寄存器；
   本直调实现使用当前 SDK 中的 `SetSysWorkspaceForce`。
2. 此 kernel 参数顺序不是 `HAVE_WORKSPACE/HAVE_TILING` 约定格式，
   因而不会自动走生成包装器的 workspace 初始化分支。
   在 AIC 上显式调用 `matmul::clearWorkspace`，清理 KFC 通信区并通知等待中的 AIV。
   漏掉这一步会在第一次设备同步处等待，不是矩阵精度问题。
3. CANN 旧版 CMake helper 禁用了普通 CMake RPATH 生成；使用显式链接选项，
   从 `$ORIGIN/lib` 加载配套 kernel 库。当前构建包含 Host 库、顺序 kernel 库和树形
   kernel 库，部署必须保留三个 `.so` 及相对目录。
4. 不应把未经校验的裸 GPU packet/tiling 交给底层接口。Python 入口在 Host 校验 slot，
   kernel 也在读取池之前检查长度和 slot 范围；底层非法 packet 输出 NaN 而不越界访问。
5. 当前 SDK 的 `IterateAll(LocalTensor)` 要求 L1 输出，不能将 UB 输出直接传入。
   本版使用 `CType=GM` 的 `IterateAll(GlobalTensor)`，输出到独立 scratch 后再搬入 UB。

## 第二轮优化与验证（2026-11-03）

### 改动与正确性约束

1. 从固定 32 行扩展为 32/64/128 行，并按设备核数、batch、heads 自动选择。
   910B2C 的 24 个 Cube 核上，8 heads 的 BS=1 选择 32 行，BS=4 选择 128 行。
2. 复用旧状态的 UB 加载 S，把三个行块缩为两个；保留 FP32 顺序组合，
   无树形重排、HF32/BF16 降精度、预计算输出或修改缓存池。
3. Matmul 输出使用每个 AIV 独享的 GM scratch，不走内部 KFC 输出环形分配。
   试验中的 UB 输出版本在 128 行、4 段时等待；改成受支持的
   `Iterate + GetTensorC` 仍能复现。SDK 的 `AllocUB` 环形边界逻辑与两个 64 KiB
   分配的回绕吻合；这属于源码分析推断，未用设备指令级采样确认。
   最终改用显式 GM scratch 后，多轮长链与不同任务数的回归通过。
4. 修复 Host 回调持续持有 Tensor 的问题。扩大首版压力测试时，分配累计到约
   54.73 GiB 后 OOM；在回调里释放 Tensor 的中间尝试还出现了 Host 等待。
   最终遵循仓库直调适配器的 POD 参数捕获方式，在调用线程释放 Tensor，
   并记录消费 stream。新版重复调用测试不再随调用次数保留工作区。

scratch 容量为 `AIV工作核数 * tile_rows * 128 * sizeof(float)`，
910B2C 上最多 3 MiB，加上 CANN 返回的 16 MiB system workspace，
单次请求容量与 segment 数无关。BS=4、8 heads、自动 128 行时 scratch 为 2 MiB。
仍通过 allocator 为每次调用分配独立逻辑工作区，没有引入全局可变工作区或 plan 输出别名。

### 测试结果

- `build-v2-final.log` / `build-v2-candidate-final.log`：默认 namespace 与独立 A/B namespace 均编译通过。
- `validation-v2-final.log`：**151 passed**；默认自动分块的 12 组 benchmark 全部通过。
  Ascend C 最终状态与关闭 HF32 的 FP32 `torch.bmm` 参考最大绝对差均为 **0**。
- `ab-v2-seed2-b{BS}-n64.log`：seed=20261104，BS=1/4、64 段再次通过，
  Ascend C 最大绝对差仍为 0；BS=4 完整/prepared 为 0.4937/0.3800 ms，
  对应第一版为 0.8151/0.7015 ms，与主轮结论一致。
- 分块回归包含 0（auto）/32/64/128，最多 65 段、17 个请求、4 heads，
  覆盖超过工作核数量的循环任务、非交换矩阵、重复 slot、空请求/池和非法元数据。
- 每种分块额外执行 256 次重复调用，检查同步后活跃分配增长小于 2 MiB；
  还检查临时 plan 销毁后的排队任务和跨 stream 创建的池生命周期。
  多轮 benchmark 后，活跃分配随当时输入规模变化（约 6–132 MiB），没有逐轮累积。

### 与第一版的同机同输入对照

先冻结第一版共享库和 Python 包装器，再使用 `HYPIC_OP_NAMESPACE=hypic_ascendc_candidate`
构建新版。两者有独立算子 namespace、kernel 符号与 kernel 库名，避免符号替换造成假对照。
**每个形状单独启动进程**，在该进程内以相同输入交替测量旧版、新版和 Triton。
第一版保留回调的缺陷使长时间同进程 sweep 不安全，因此限制每个版本最多 100 次计时调用；
未修改冻结的旧实现，也不把它当作长期稳定性基线。

8 heads，K=V=128，seed=20261103，N 为最大段数，ragged 规则同第一版。
每轮 10 次、5 轮取中位数。单位 ms，日志 `ab-v2-b{BS}-n{N}.log`：

| BS | N | 旧版完整 | 新版完整 | 旧版 prepared | 新版 prepared | Triton Cube cached launcher |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.1568 | 0.1262 | 0.0880 | 0.0478 | 0.0908 |
| 1 | 16 | 0.1946 | 0.1587 | 0.0874 | 0.0730 | 0.1413 |
| 1 | 64 | 0.3538 | 0.3345 | 0.2543 | 0.2409 | 0.3381 |
| 4 | 4 | 0.1517 | 0.1268 | 0.0761 | 0.0434 | 0.0954 |
| 4 | 16 | 0.2740 | 0.2027 | 0.1851 | 0.1051 | 0.1898 |
| 4 | 64 | 0.8254 | 0.4998 | 0.7076 | 0.3802 | 0.5967 |

BS=4、64 段时，新版完整调用较第一版降低约 **39%**，prepared 降低约 **46%**。
完整调用较本轮 Triton Cube cached launcher 低约 **16%**。
BS=1 长链改善较小；短链完整调用仍可能慢于 Triton Cube 的 launcher，
不能宣称所有形状都更快。prepared 排除了每请求准备，不能拿其比值代表模型 TTFT。

复现实例（`baseline_v1/` 必须先保存真正的旧库及旧包装器）：

```bash
cmake -S csrc/hypic_compose_cached -B build/candidate_v2 \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
  -DHYPIC_OP_NAMESPACE=hypic_ascendc_candidate
cmake --build build/candidate_v2 -j 4
python examples/offline_inference/hypic_compose_ascendc_bench.py \
  --library build/candidate_v2/libhypic_compose_ascendc.so \
  --namespace hypic_ascendc_candidate --baseline-root baseline_v1 \
  --triton --batch-sizes 4 --segments 64 --heads 8 --iterations 10 --repeats 5
```

第一版冻结二进制 SHA256：

```text
libhypic_compose_ascendc.so
6590c227be81d409566351a92c75a89fa49b12d45ec2c1a0ab084b8e6144cc99
lib/libhypic_cached_kernel.so
262462fe51b3323764ad046395d06c307c167a9a33dd61ea6accd3fce6897712
```

本轮仍为服务器上的 mock 算子验证；未下载模型，未修改生产后端默认值，
没有重新运行 LongBench/MCPAgentBench 或宣称模型端到端收益。

## 第三轮：有序树形与混合归并（2026-11-03）

### 计算方式

增加显式实验接口 `CachedComposeTreeAscendC` / `compose_cached_tree_ascendc`。
顺序入口 `CachedComposeAscendC` 的 kernel 保持第二轮实现，未切换默认后端。
树形版本只合并相邻节点，按以下顺序计算：

```text
S_AB = S_A @ T_B + S_B
T_AB = T_A @ T_B
```

允许改变括号，不允许交换 A/B。输入仍为 K=V=128、FP32、只读 cached S/T，
初始状态为 0，只输出最终 H。这不是支持任意初始 H 的通用 `(S,T)` 归约接口。
由于初始 H=0，每个请求最左侧节点的 T 永远不会被使用，可以沿左侧分支省掉它。
例如四段只需要 `S01`、`S23`、`T23`，再计算 `S01 @ T23 + S23`，不计算 `T01`。
奇数尾节点直接复制；空请求保留零状态，不补齐单位矩阵再计算。

- `merge_levels=0`：完整树形，直到每个请求剩一个节点。
- `merge_levels=1/2/3`：只归并前几层，随后调用已有顺序 kernel 合并剩余节点。
  `merge_levels=2` 相当于先并行组合相邻四段，再按原顺序组合这些分组。
  若请求更短，会自然提前完成，不额外运行空层。

`tree_kernel.cpp` 每层从 GM 读取 S/T，使用 FP32 Cube Matmul，Vector 加入右侧 S。
不同请求、节点、head、行块并行；层与层之间以同一 stream 的 kernel 边界保证依赖。
Host 一次提交包含全部层及可选顺序尾部的回调，不逐层回到 Python，不进行逐层设备同步。
所有元数据与 tiling 合并为一次 H2D 传输；prepared plan 不缓存任何中间计算结果。

每次调用分配两组 S/T 中间缓冲区，交替读写；两组分别按各自最大层的节点数分配，
只归并一层时只需要一组。空间随节点数增长，明显大于顺序版本。
调用线程释放张量，`recordStream` 保护异步使用，回调不捕获 Tensor 所有者。

树形与顺序组合在精确算术下等价，但 FP32 的舍入顺序不同。
测试必须检查误差，不能要求或宣称逐 bit 一致，也不能从 mock 误差推导模型 token 必然一致。

### 实机复现

```bash
cmake -S csrc/hypic_compose_cached -B build/tree_release \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0
cmake --build build/tree_release -j 4
python examples/offline_inference/hypic_compose_ascendc_bench.py \
  --library build/tree_release/libhypic_compose_ascendc.so \
  --tree --merge-levels 0 2 --tests \
  --batch-sizes 1 4 --segments 4 16 64 128 256 --heads 8 \
  --iterations 20 --repeats 7 --seed 20261104
```

调用例子：

```python
from vllm_ascend.hypic.compose_ascendc import CachedComposeTreeAscendC

# 显式选择前两层并行归并；slot 的 pin/generation 管理仍由调用方负责。
plan = CachedComposeTreeAscendC(S_pool, T_pool, slots, merge_levels=2)
final = plan(S_pool, T_pool)
```

bench 输出顺序、完整树形及指定深度的完整/prepared 耗时，另有 `max_abs`、
`relative_l2` 和 `peak_extra_mib`。后者是在正确性检查阶段单独读取的
allocator 峰值活跃分配增量，包含输出和临时工作区，不包含原有输入池，
也不是设备全部物理显存占用。重置峰值计数及误差计算不在计时区间内。

与前两轮一样，计时是预热后连续调用的轮均耗时、多轮取中位数，
不是纯 Cube 时间、单请求 RTT 或模型 TTFT。

### 验证与对照结果

环境沿用第二轮的 Ascend910B2C / CANN 9.0.0 / torch-npu 2.10.0。
仍使用 mock S/T，未下载模型。第二轮库及源码冻结在服务器 `baseline_v2/`。
本轮对照使用同一扩展中的原顺序路径；`kernel.cpp` 与冻结版 SHA256 一致，
树形实现在独立的 `tree_kernel.cpp` 中。没有修改输入、FP32 模式、计时口径或校验阈值。

- `tree-first.log`：完整树形首次运行，220 项测试通过，16 组 benchmark 通过。
- `hybrid-first.log`：追加 1/2/3 层混合方案，341 项测试通过。
- `tree-head1.log`：seed=20261103、1 head 的 8 组对照通过。
- `tree-final-h8.log`：缩减两组临时缓冲区后，**341 passed**，seed=20261104 的 10 组对照通过。
- `tree-final-h1.log`：同一最终实现、seed=20261104、1 head 的 8 组对照通过。

顺序路径相对 FP32 `torch.bmm` 参考仍为零差异。最终 18 组输入中，
树形/混合最大绝对差为 **1.6391e-7**，最大相对 L2 误差为 **3.1095e-7**，
均通过 `atol=2e-5, rtol=2e-4`。回归覆盖奇数/非二次幂长度、空请求、重复 slot、
非交换矩阵、接近单位且略有放大的 T、缓存只读、临时 plan、返回值不别名和显存不累积。
这验证了本次 mock 范围内的数值正确性，不等于实际模型输出 token 或准确率验证。

最终主轮使用 seed=20261104、每轮 20 次、7 轮中位数，N 是最大段数，
请求 r 的长度为 `max(0,N-r)`。下表为 **prepared 调用耗时**（ms）：

| Heads | BS | N | 原顺序 | 完整树形 | 前两层树形 + 顺序 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 4 | 0.0392 | 0.0448 | 0.0431 |
| 8 | 1 | 16 | 0.0665 | 0.0961 | 0.0882 |
| 8 | 1 | 64 | 0.2403 | 0.2202 | 0.1999 |
| 8 | 1 | 128 | 0.4691 | 0.4172 | 0.4045 |
| 8 | 1 | 256 | 0.9407 | 0.7964 | 0.8160 |
| 8 | 4 | 4 | 0.0342 | 0.0460 | 0.0462 |
| 8 | 4 | 16 | 0.0966 | 0.1599 | 0.1436 |
| 8 | 4 | 64 | 0.3718 | 0.5726 | 0.5069 |
| 8 | 4 | 128 | 0.7324 | 1.3436 | 1.1821 |
| 8 | 4 | 256 | 1.7534 | 3.0178 | 2.5184 |
| 1 | 1 | 16 | 0.0619 | 0.0599 | 0.0560 |
| 1 | 1 | 64 | 0.2290 | 0.1044 | 0.1087 |
| 1 | 1 | 128 | 0.4523 | 0.1377 | 0.1773 |
| 1 | 1 | 256 | 0.8981 | 0.1820 | 0.3104 |
| 1 | 4 | 16 | 0.0624 | 0.0745 | 0.0685 |
| 1 | 4 | 64 | 0.2295 | 0.1516 | 0.1437 |
| 1 | 4 | 128 | 0.4552 | 0.2238 | 0.2393 |
| 1 | 4 | 256 | 0.9054 | 0.3529 | 0.4385 |

prepared 中，8 heads、BS=1、64 段的两层混合减少约 17%；1 head、BS=1、256 段的
完整树形约为顺序版本的 **4.93 倍速度**。8 heads、BS=4 时，完整树形和混合方案均退化。
这支持“小 batch 且独立 head 工作量少时，可通过 segment 并行提高吞吐”的判断，
不支持“树形普遍更快”。增加 T 的组合工作与中间读写会抵消原本已经存在的并行度。

### 完整调用与显存代价

完整调用还要生成各层 metadata/tiling。主轮中的代表结果（ms）：

| Heads | BS | N | 原顺序完整调用 | 完整树形调用 | 两层混合完整调用 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 64 | 0.3597 | 0.4610 | 0.3661 |
| 8 | 1 | 128 | 0.5552 | 0.5072 | 0.4858 |
| 8 | 4 | 64 | 0.4755 | 0.6848 | 0.6202 |
| 1 | 1 | 128 | 0.5436 | 0.3432 | 0.2813 |
| 1 | 1 | 256 | 1.0039 | 0.4666 | 0.4195 |
| 1 | 4 | 256 | 1.0909 | 1.1478 | 1.0011 |

因此 cached S/T 已就绪，并不自动意味着完整请求可以获得 prepared 表中的加速。
8 heads、BS=1、64 段的完整调用收益接近临界点，并受 Host 波动影响；不能只引用
prepared 的 17% 来声称这类完整调用稳定加速。相反，1 head 的长链即使算入准备也有收益。

针对上述 8 heads、64 段完整调用的波动，再将每轮调用数增加到 100、重复 11 轮，
保持相同 seed、输入及实现（`tree-confirm-b{BS}-h8-n64.log`），得到：

| BS | 口径 | 原顺序 | 完整树形 | 两层混合 |
| ---: | --- | ---: | ---: | ---: |
| 1 | prepared | 0.2367 | 0.2200 | 0.1982 |
| 1 | 完整调用 | 0.3160 | 0.3209 | 0.2754 |
| 4 | prepared | 0.3728 | 0.5757 | 0.5189 |
| 4 | 完整调用 | 0.4693 | 0.6755 | 0.6096 |

长轮均复测中，BS=1 的两层混合完整调用下降约 **13%**，prepared 下降约 **16%**；
完整树形的完整调用仍与顺序接近。BS=4 的退化结论不变。
这组长轮均结果用于评估连续调用吞吐，不掩盖前面短轮均记录中的 Host 波动，
也不能据此承诺单个动态请求的相同延迟收益。

64 段时的实测峰值额外活跃分配（包含输出，排除原有池，MiB）：

| Heads | BS | 原顺序 | 完整树形 | 两层混合 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 17.0 | 64.5 | 65.0 |
| 8 | 4 | 20.0 | 208.0 | 210.0 |
| 1 | 1 | 16.1 | 22.1 | 22.1 |
| 1 | 4 | 16.5 | 40.0 | 40.3 |

保留两个显式实验入口，不按本轮有限形状自动替换模型的生产后端。
若要接入真实请求，应同时考虑 head/TP 分片后的并行度、batch、段数、临时显存预算、
metadata 是否能安全复用，以及重关联误差对实际模型输出的影响。

## 第四轮：单 kernel 完整树形（2026-11-03）

### 一次启动与层间同步

新增 `CachedComposeTreeAscendC(..., single_kernel=True)`，一次启动完成全部有序二叉归并。
64 段仍按 `64→32→16→8→4→2→1` 计算；S/T 组合公式、奇数尾节点直接搬运、
每条请求左侧无需计算 T 的规则与第三轮相同。没有改为段重排或近似精度。

顺序 kernel 的每个 worker 只更新自己负责的 H 行块，读取已经存在的 S/T。
树形 kernel 则需要读取其他 worker 刚生成的 T，因此单次启动不意味着没有层间同步：

1. 启动固定的 AIC/AIV worker，AIC 运行 Matmul 服务，AIV 分配 node/head/行块任务。
2. 一层内，完成同步 Matmul、Vector Add 及写回；没有任务的 AIV 也进入层末等待。
3. 用 `CrossCoreSetFlag<0, PIPE_MTE3>(8)` / `CrossCoreWaitFlag(8)` 做全部 AIV 的层间同步。
4. 切换两组 GM 中间缓冲，继续下一层；最后一层直接写输出。

这里不能直接插入要求 AIC 共同进入的 mixed `SyncAll`：AIC 此时仍在 Matmul 服务循环中。
本实现仅注册一个 Matmul 对象，核间 flag 8 与其 0/1 以及 workspace 初始化的 15 分开。
这些约束依据本机 CANN 9.0.0 头文件，并在 910B2C 实测；升级 SDK 需重新检查占用。
官方接口说明见 [CrossCoreSetFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html)。

kernel 使用 `__schedmode__(1)` 和 `__mix__(1,2)`，Host 同时验证 worker 数不超过物理核数，
避免多流并发下部分核尚未调度、已运行核却等待它们的情况。
所有层复用一个 Matmul 对象，按首层最大行块分配 UB；树变窄时用 `SetTail` 减小有效 M。
`tile_rows=0` 仍按每层任务数选择 32/64/128 行，而不是强制所有层都使用 128 行。

注意：一次启动仍需要层间等待、额外 T 矩阵乘法和 GM 中间读写。
没有将 O(N) 中间缓冲压成 O(1)，也没有把全部摘要放进片上存储。

### 构建与使用

本入口要求 CANN 9 的原生 ASC 编译工具链。旧 `ascendc.cmake` 会将入口改写为 inline，
导致 `__schedmode__` 不能正确保留。为此，顺序/分层基线保持旧编译方式，
单 kernel 在 `csrc/hypic_compose_cached/native_single/` 子工程中使用 `find_package(ASC)`。
不修改安装目录下的 SDK 脚本，也不移除 batch 调度属性来绕过编译错误。

```bash
cmake -S csrc/hypic_compose_cached -B build/tree_single_final \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0
cmake --build build/tree_single_final -j 4

python examples/offline_inference/hypic_compose_ascendc_bench.py \
  --library build/tree_single_final/libhypic_compose_ascendc.so \
  --tests --tree --merge-levels 0 2 --single-tree \
  --batch-sizes 1 4 --segments 4 16 64 128 256 \
  --heads 8 --iterations 40 --repeats 9 --seed 20261104
```

部署时保留这四个库及相对位置：

```text
libhypic_compose_ascendc.so
lib/libhypic_cached_kernel_v2.so
lib/libhypic_tree_kernel.so
native_single/libhypic_tree_single_kernel.so
```

```python
plan = CachedComposeTreeAscendC(S_pool, T_pool, slots, single_kernel=True)
final = plan(S_pool, T_pool)
```

默认 `single_kernel=False` 保持第三轮行为。第四轮的初始单 kernel 版本仅支持完整树形；
第五轮已支持非零 `merge_levels`，同样只启动一次，详见下文。
元数据只做一次 H2D，prepared plan 不缓存 S/T 内容，也不保存 pool 裸地址；
每次调用的输出、两组缓冲和 workspace 独立分配，维持原来的 stream/slot 生命周期契约。

### 正确性与性能

实机环境与前轮相同，只使用 mock FP32 S/T。第三轮源码/库冻结在服务器
`/data/hypic-ascendc-validation-20261103/baseline_tree_v1/`；顺序与分层 kernel 源码 SHA256 均未变化。

`single-h8.log`：**444 项 NPU 回归通过**。新增覆盖 0/32/64/128 行块、奇数/非二次幂长度、
空 pool/batch、非交换/近单位矩阵、缓存只读及更新、临时 plan、显存不累积、非法 header、
物理核数约束，以及两个 stream 同时提交单 kernel 归并。
18 组主轮 benchmark 最大绝对误差 **1.6391e-7**、最大相对 L2 **3.1095e-7**，
均通过 `atol=2e-5, rtol=2e-4`。这是 mock 正确性验证，不是实际模型输出一致性结论。

主轮 seed=20261104、每轮 40 次、9 轮中位数；prepared 耗时（ms）：

| Heads | BS | N | 顺序 | 分层完整树形 | 两层树形 + 顺序 | 单 kernel 完整树形 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 4 | 0.0307 | 0.0391 | 0.0394 | 0.0392 |
| 8 | 1 | 16 | 0.0647 | 0.0938 | 0.0862 | 0.0654 |
| 8 | 1 | 64 | 0.2368 | 0.2229 | 0.2018 | 0.1651 |
| 8 | 1 | 128 | 0.4691 | 0.4133 | 0.3987 | 0.3452 |
| 8 | 1 | 256 | 0.9596 | 0.7975 | 0.8171 | 0.7224 |
| 8 | 4 | 64 | 0.3795 | 0.5735 | 0.5059 | 0.5216 |
| 8 | 4 | 256 | 1.6345 | 2.8169 | 2.4783 | 2.7465 |
| 1 | 1 | 16 | 0.0605 | 0.0584 | 0.0550 | 0.0403 |
| 1 | 1 | 64 | 0.2282 | 0.1014 | 0.1063 | 0.0634 |
| 1 | 1 | 256 | 0.8988 | 0.1839 | 0.3102 | 0.1229 |
| 1 | 4 | 64 | 0.2287 | 0.1499 | 0.1418 | 0.1057 |
| 1 | 4 | 256 | 0.9039 | 0.3527 | 0.4365 | 0.2810 |

第二个 seed=20261105 的长轮均复测（100 次 × 11 轮，`single-confirm.log`）：

| Heads | BS | N | 口径 | 顺序 | 分层完整树形 | 两层树形 + 顺序 | 单 kernel 完整树形 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 1 | 64 | prepared | 0.2377 | 0.2218 | 0.1981 | 0.1642 |
| 8 | 1 | 64 | 完整调用 | 0.3163 | 0.3217 | 0.2742 | 0.2222 |
| 8 | 4 | 64 | prepared | 0.3667 | 0.5718 | 0.5032 | 0.5206 |
| 8 | 4 | 64 | 完整调用 | 0.4648 | 0.6703 | 0.6019 | 0.5814 |

BS=1、8 heads、64 段：相对分层完整树形，prepared 降低约 **26%**，完整调用降低约 **31%**；
相对顺序版两项均降低约 **30%**。BS=4、8 heads 仍比顺序慢，不应统一替换后端。
1 head、BS=1、256 段的 prepared 相对顺序约 **7.31 倍速度**，完整调用为 1.0031→0.3500 ms。
计时仍是含校验/分配/提交开销的轮均调用时间，不是纯 Cube 时间或模型 TTFT。

单 kernel 与分层树形的临时分配相同：64 段、8 heads 时，BS=1 约 64.5 MiB，BS=4 约 208.0 MiB；
顺序版分别约 17.0 / 20.0 MiB。一次启动解决的是启动和重复初始化开销，未消除树形的显存代价。

最后在新的 `build/tree_single_final/` 目录从零构建，再跑 **444 项回归全部通过**，
`build-single-final.log` 记录 `FINAL_EXIT=0`。同样的 100 次 × 11 轮、seed=20261105 复核结果
（`single-final.log`，8 heads、64 段，ms）：

| BS | 口径 | 顺序 | 分层树形 | 两层混合 | 单 kernel 树形 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | prepared | 0.2370 | 0.2221 | 0.2014 | 0.1667 |
| 1 | 完整调用 | 0.3165 | 0.3241 | 0.2797 | 0.2254 |
| 4 | prepared | 0.3599 | 0.5730 | 0.5028 | 0.5171 |
| 4 | 完整调用 | 0.4607 | 0.6749 | 0.5998 | 0.5797 |

复核维持同一结论：BS=1 时单 kernel 比分层树形的 prepared 低约 25%、完整调用低约 30%。
短序列或已具备较高 head/batch 并行度的情况，仍需与顺序方案比较，不能只看启动次数。

另用 `torch_npu.profiler` 在预先创建 plan 后采集一次调用（1 次 warmup、1 次 active），
`single-profile-clean/` 下的 `kernel_details.csv` 确认：64 段时顺序 **1** 个、分层树形 **6** 个、
两层混合 **3** 个、新完整树形 **1** 个 device kernel。
这是实际启动数量核验，不只是把多次启动包装成一个 Python/OpCommand 调用。
profiler 不参与上表计时；不能将这些调用延迟直接称为纯 kernel 延迟。

原始日志、汇总 `single-summary.json` 及 profiler 结果位于上述服务器验证目录，
主轮日志为 `single-h8.log`、`single-h1.log`。该入口继续作为显式实验选项，不修改生产后端默认值。

## 第五轮：单 kernel 树形＋顺序尾段（2026-11-03）

### 接口与执行过程

现在可以把前 1/2/3 层树形和后续顺序融合放进同一个 kernel：

```python
plan = CachedComposeTreeAscendC(
    S_pool, T_pool, slots, single_kernel=True, merge_levels=2
)
final = plan(S_pool, T_pool)
```

- `single_kernel=True, merge_levels=0`：第四轮的单 kernel 完整树形。
- `single_kernel=True, merge_levels=L`：最多进行 L 层相邻二叉归并，再在同一个 kernel 内顺序融合剩余摘要。
- 若所有请求提前归并成一个摘要，则直接输出，不额外执行尾段。
- `single_kernel=False` 保留原来的逐层启动/顺序尾段启动方式。

H0=0、段顺序、S/T 公式和左侧 T 剪枝不变。以一条请求、一个 head 的 64 段为例：

| 方案 | 矩阵乘法次数 | 单 kernel 入口中的启动次数 |
| --- | ---: | ---: |
| 顺序 | 63 | 1 |
| 一层树形＋顺序 | 94 | 1 |
| 两层树形＋顺序 | 109 | 1 |
| 三层树形＋顺序 | 116 | 1 |
| 完整树形 | 120 | 1 |

这些计数不含直接搬运和 Vector Add；ragged/奇数段的具体次数随请求变化。
减少总乘法次数会拉长剩余顺序链，因此不代表延迟一定下降。

实现仍在 `tree_single_kernel.cpp`，新增 `hypic_tree_hybrid_kernel` 入口：

1. 树形阶段使用 GM A/B/C 的 Matmul，按层做全部 AIV 同步。
2. 最后一层的 S/T 写入中间缓冲并同步后，AIV 按 request/head/行块领取顺序尾段任务。
3. 尾段使用第二个 VECOUT A 的 Matmul，把 H 保存在 UB，先读取第一组 S，
   后续依次计算 `H @ T + S`。每步乘积仍经过该 worker 独享的 GM scratch，最终 H 只写出一次。
4. 两个 Matmul 对象同一时间注册到一个 AIC 服务中，没有在 Device 内重新启动 kernel。

两种阶段复用已有的两份 UB 行块。尾段的 tiling 按树形最大行块预留 UB，
不能只按较小的 tail 行数估计可用空间。同步 flag 8 与两个 Matmul 使用的 0..3 分开；
继续使用 batch 调度，空闲 AIV 也参与最后一次树形同步。

Host 一次拷贝 tree header、节点 packet、两套 tiling 和 tail packet。
`cached_compose_tree_single` 的新增尾段参数有默认值，原低层无尾段调用仍有效；
使用新的 Python wrapper 前应重新构建扩展。每次调用独立分配输出、缓冲和工作区，
回调仍只持有地址/标量，不保留 Tensor 所有权。

### 实机验证

沿用 Ascend910B2C / CANN 9.0.0 / FP32 mock 池，未下载模型。
第四轮库和源码冻结在验证目录的 `baseline_single_tree_v1/`，本轮构建目录为
`build/tree_hybrid_single/`。顺序 `kernel.cpp` 与逐层 `tree_kernel.cpp` 的 SHA256 未变。

```bash
cmake -S csrc/hypic_compose_cached -B build/tree_hybrid_single \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0
cmake --build build/tree_hybrid_single -j 4

python examples/offline_inference/hypic_compose_ascendc_bench.py \
  --library build/tree_hybrid_single/libhypic_compose_ascendc.so \
  --tests --tree --single-tree --merge-levels 0 1 2 3 \
  --batch-sizes 1 4 --segments 4 16 64 128 256 \
  --heads 8 --iterations 40 --repeats 9 --seed 20261104
```

`--single-tree` 现在也遵循 `--merge-levels`，输出 `single_tree`、`single_hybrid1/2/3` 指标，
同时保留 `--tree` 的多次启动对照。

**704 项 NPU 测试通过**。新增三个切换深度下的 ragged/奇数长度、非交换矩阵、
近单位矩阵、缓存只读和更新、临时 plan、显存不累积、多 stream 不同请求，以及非法尾段 packet。
18 组主轮 benchmark 最大绝对误差 **1.6391e-7**、最大相对 L2 **3.1095e-7**，
通过原有 `atol=2e-5, rtol=2e-4`。仍是算子 mock 验证，不等同于模型 token 或准确率评估。

### 结果与取舍

主轮 seed=20261104、40 次 × 9 轮中位数。以下均为 prepared 调用耗时（ms），
保留 Host 校验、分配和提交成本；不含元数据准备，不是纯 kernel 时间。

| Heads | BS | N | 顺序 | 单 kernel 完整树形 | 单 kernel 一层＋顺序 | 单 kernel 两层＋顺序 | 单 kernel 三层＋顺序 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 16 | 0.0656 | 0.0622 | 0.0737 | 0.0698 | 0.0703 |
| 8 | 1 | 64 | 0.2383 | 0.1621 | 0.2051 | 0.1835 | 0.1736 |
| 8 | 1 | 128 | 0.4669 | 0.3439 | 0.4102 | 0.3871 | 0.3664 |
| 8 | 1 | 256 | 0.9386 | 0.7234 | 0.8715 | 0.8081 | 0.7707 |
| 8 | 4 | 16 | 0.0963 | 0.1253 | 0.1163 | 0.1272 | 0.1375 |
| 8 | 4 | 64 | 0.3796 | 0.5177 | 0.4446 | 0.4892 | 0.5113 |
| 8 | 4 | 128 | 0.7388 | 1.2790 | 0.9778 | 1.1646 | 1.2429 |
| 8 | 4 | 256 | 1.6246 | 2.7350 | 2.1784 | 2.4566 | 2.6145 |
| 1 | 1 | 64 | 0.2279 | 0.0643 | 0.1405 | 0.0943 | 0.0736 |
| 1 | 1 | 256 | 0.8975 | 0.1248 | 0.5051 | 0.2999 | 0.1998 |
| 1 | 4 | 64 | 0.2284 | 0.1035 | 0.1650 | 0.1353 | 0.1171 |
| 1 | 4 | 256 | 0.9032 | 0.2835 | 0.5832 | 0.4259 | 0.3506 |

8 heads、BS=4 的一层混合比完整树形减少约 14%–24% prepared 时间，但仍慢于顺序版。
在上述 prepared 口径中，BS=1 或 head 较少时，完整树形更快。该结果说明减少总乘法次数与缩短关键依赖路径之间存在取舍，
不能只按 FLOPs 或启动次数选择后端，也不支持把混合方案统一设为默认。

一层混合只需要一组 S/T 中间缓冲；两层及以上需要两组。因此混合方案的显存收益也依赖切换层数。
64 段、8 heads 的峰值额外活跃分配（MiB，prepared，排除输入池）约为：

| BS | 顺序 | 单 kernel 完整树形 | 单 kernel 一层＋顺序 | 单 kernel 两层＋顺序 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.0 | 64.5 | 49.3 | 65.3 |
| 4 | 20.0 | 208.0 | 148.0 | 212.0 |

这些额外分配包含输出、CANN system workspace、中间摘要和尾段 scratch。
一层混合降低了临时显存，两层以上未必比完整树形更省。

主轮日志为服务器验证目录下的 `hybrid-single-h8.log`、`hybrid-single-h1.log`，
汇总为 `hybrid-single-summary.json`。保留所有显式选项，生产后端默认值不变。

### 独立种子复测与启动数量核验

继续使用 seed=20261106、100 次 × 11 轮复测 6 组形状，全部通过同一数值阈值。
8 heads 的 `hybrid-single-confirm.log` 以及 1 head 的 `hybrid-single-confirm-h1.log`
记录了如下结果（ms）：

| Heads | BS | N | 口径 | 顺序 | 单 kernel 完整树形 | 单 kernel 一层＋顺序 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 8 | 1 | 64 | prepared | 0.2371 | 0.1664 | 0.2042 |
| 8 | 1 | 128 | prepared | 0.4676 | 0.3554 | 0.4008 |
| 8 | 4 | 64 | prepared | 0.3774 | 0.5208 | 0.4520 |
| 8 | 4 | 128 | prepared | 0.7460 | 1.2684 | 0.9926 |
| 1 | 4 | 64 | prepared | 0.2283 | 0.1036 | 0.1660 |
| 1 | 4 | 256 | prepared | 0.9029 | 0.2887 | 0.5894 |
| 1 | 4 | 64 | 完整调用 | 0.3249 | 0.3398 | 0.3303 |
| 1 | 4 | 256 | 完整调用 | 1.0689 | 0.9825 | 0.8748 |

8 heads 的 prepared 结论复现：BS=1 完整树形最快；BS=4 一层混合比完整树形快约
13%–22%，但仍慢于顺序。1 head、BS=4、256 段的完整调用则出现另一种取舍：一层混合
比完整树形快约 **11%**，也比顺序快约 **18%**；主轮的对应结果为 0.9005 / 0.9974 / 1.0780 ms，
方向一致。

完整调用会重新构造计划、生成和拷贝元数据。一层混合减少了完整树形的逐层节点元数据，
这能解释完整调用与 prepared 排名不同；尚未单独剖析其 Host 阶段耗时，
不能把两种计时直接相减当作精确的元数据耗时，也不能据此宣称混合 kernel 本身更快。
如果调度器可以复用 prepared plan，应优先参考 prepared 指标。

在独立的 profiler 进程中，对 BS=1、8 heads、64 段的预建计划做 1 次 warmup、1 次 active：

| 方案 | `kernel_details.csv` 中的实际 device kernel 数 | kernel 名称 |
| --- | ---: | --- |
| 单 kernel 完整树形 | 1 | `hypic_tree_single_kernel` |
| 单 kernel 一层＋顺序 | 1 | `hypic_tree_hybrid_kernel` |
| 单 kernel 两层＋顺序 | 1 | `hypic_tree_hybrid_kernel` |
| 单 kernel 三层＋顺序 | 1 | `hypic_tree_hybrid_kernel` |

采集结果位于服务器验证目录 `hybrid-single-profile/`，日志为 `hybrid-single-profile.log`。
profiling 与 benchmark 串行进行，不参与上表计时。两轮复测的退出标记分别为
`hybrid-single-confirm.done` 中的 `CONFIRM_EXIT=0` 和
`hybrid-single-confirm-h1.done` 中的 `CONFIRM_H1_EXIT=0`。
