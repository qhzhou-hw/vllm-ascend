# HYPIC LongBench-E 测试指南

本文说明如何在 vLLM Ascend 上使用 LongBench v1 的 LongBench-E 子集，对比 Full Recompute、HYPIC chunk 512 和 HYPIC chunk 1024 的准确率，并与 SGLang Ascend 结果对齐。

HYPIC 的实现原理和适配注意事项见 [HYPIC 在 vLLM Ascend 上的适配说明](../hypic_ascend_porting_zh.md)。

## 1. 实验目标和口径

测试四个 LongBench-E 数据集：

| 数据集 | 任务 | 样本数 | Metric |
|---|---|---:|---|
| Qasper | 长文档问答 | 224 | QA F1 |
| GovReport | 政府报告摘要 | 300 | `rouge_score` |
| HotpotQA | 多跳问答 | 300 | QA F1 |
| MultiNews | 多文档摘要 | 294 | `rouge_score` |

三种运行模式：

| run name | vLLM 配置 | chunk |
|---|---|---:|
| `full_recompute` | 关闭 prefix caching | 0 |
| `hypic_512` | 开启 HYPIC | 512 |
| `hypic_1024` | 开启 HYPIC | 1024 |

所有模式必须保持一致：

- 同一个模型目录和 tokenizer；
- LongBench 官方 prompt template；
- LongBench 官方 dataset-specific max output length；
- `temperature=0`；
- `enable_thinking=False`；
- 相同数据文件和 index 顺序；
- TP=2、eager execution。

Full Recompute 和 HYPIC 都推荐使用 batch=4，与 SGLang B4 基线保持一致。HYPIC
调度器会将一组 HYPIC prefill 与已有 decode 请求隔离；`max_num_batched_tokens` 控制每步
总 token 预算。因此 batch=4 表示最大并发数为 4，超长 prompt 的 token 总数超过预算时，
调度器可能分成多个 admission group。报告中应同时记录 batch size 和 token budget。

## 2. 已验证配置

以下是本次 Ascend 实机使用的配置：

| 参数 | Full Recompute | HYPIC-512/1024 |
|---|---:|---:|
| 模型 | Qwen3.5-35B-A3B | Qwen3.5-35B-A3B |
| NPU / TP | 2 / 2 | 2 / 2 |
| max model length | 45056 | 45056 |
| batch size | 4 | 4 |
| max batched tokens | 45056 | 45056 |
| GPU/NPU memory utilization | 0.85 | 0.85 |
| max cache segments | - | 96 |
| seam sink tokens | - | 8 |

`max_model_len=45056` 是根据四个数据集中最长约 41.2k-token 的输入留余量得到的。不要用默认 8192 跑完整 LongBench-E。

HYPIC 的 Full Attention public KV、conv tail 和 GDN `S/T` 使用模型构造阶段预分配的 96-slot 固定 pool。vLLM 会先把这些 buffer 计入模型占用，再按 `gpu_memory_utilization` 计算普通 KV cache，因此不需要在运行期为按需 `clone()` 额外预留一块不可见空间。该实机配置在每卡 profile 到约 39.01 GiB 模型占用、4.15 GiB peak activation 和 8.19 GiB 普通 KV cache。不同模型、NPU 容量或 kernel 版本仍应重新测量。

## 3. 目录和环境

示例目录：

```bash
export VLLM_REPO=/data/vllm
export VLLM_ASCEND_REPO=/data/vllm-ascend
export HYPIC_PYTHON=/data/venvs/vllm-hypic/bin/python
export HYPIC_MODEL=/data/models/Qwen3.5-35B-A3B
export LONGBENCH_ROOT=/data/hypic-validation/longbench_v1
export LONGBENCH_DATA=$LONGBENCH_ROOT/datasets_e
export LONGBENCH_CONFIG=$LONGBENCH_ROOT/official_config
export SGLANG_RESULTS=$LONGBENCH_ROOT/results_e_b4
export VLLM_RESULTS=$LONGBENCH_ROOT/results_e_vllm_ascend
```

加载 Ascend 和 Python 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/venvs/vllm-hypic/bin/activate

export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH=/data/vllm:/data/vllm-ascend:${PYTHONPATH:-}

cd "$VLLM_ASCEND_REPO"
```

确认当前代码和设备：

```bash
git branch --show-current
git status --short
npu-smi info -l
python -c "import torch; print(torch.npu.is_available(), torch.npu.device_count())"
```

## 4. 准备 LongBench-E 数据

下载官方 LongBench：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 可选
hf download THUDM/LongBench --repo-type dataset \
  --local-dir /data/datasets/LongBench
```

将 LongBench-E 文件链接为 runner 使用的 canonical 名称：

```bash
mkdir -p "$LONGBENCH_DATA" "$LONGBENCH_CONFIG" "$VLLM_RESULTS"

ln -s /data/datasets/LongBench/data/qasper_e.jsonl \
  "$LONGBENCH_DATA/qasper.jsonl"
ln -s /data/datasets/LongBench/data/gov_report_e.jsonl \
  "$LONGBENCH_DATA/gov_report.jsonl"
ln -s /data/datasets/LongBench/data/hotpotqa_e.jsonl \
  "$LONGBENCH_DATA/hotpotqa.jsonl"
ln -s /data/datasets/LongBench/data/multi_news_e.jsonl \
  "$LONGBENCH_DATA/multi_news.jsonl"
```

准备官方配置：

```bash
cp /data/datasets/LongBench/config/dataset2prompt.json "$LONGBENCH_CONFIG/"
cp /data/datasets/LongBench/config/dataset2maxlen.json "$LONGBENCH_CONFIG/"
cp /data/datasets/LongBench/metrics.py "$LONGBENCH_CONFIG/"
```

不同 LongBench checkout 中 `metrics.py` 的位置可能不同，但最终目录必须包含：

```text
official_config/
├── dataset2maxlen.json
├── dataset2prompt.json
└── metrics.py
```

检查输入和 SGLang reference：

```bash
wc -l "$LONGBENCH_DATA"/*.jsonl
wc -l "$SGLANG_RESULTS"/*.jsonl
```

每种 SGLang mode 应分别包含 224、300、300、294 条，文件名形式为：

```text
full_recompute.qasper.jsonl
hypic_512.qasper.jsonl
hypic_1024.qasper.jsonl
...
```

## 5. 先做固定样本闸门验证

仓库中的 `examples/offline_inference/hypic_longbench.py` 会：

1. 从指定 SGLang reference mode 中按 index 排序；
2. 用固定 seed 抽样；
3. 使用相同 dataset row 运行 vLLM；
4. 计算官方单样本 metric；
5. 报告 SGLang/vLLM 文本是否一致和 score delta；
6. HYPIC 模式额外重复一次相同请求，检查 cold/warm token IDs。

使用完整 token 长度范围可确保三种 reference mode 从同一 index 集合抽样：

### 5.1 Full Recompute

```bash
python examples/offline_inference/hypic_longbench.py \
  --model "$HYPIC_MODEL" \
  --data-dir "$LONGBENCH_DATA" \
  --config-dir "$LONGBENCH_CONFIG" \
  --reference-dir "$SGLANG_RESULTS" \
  --reference-prefix full_recompute \
  --mode full_recompute \
  --seed 20260825 \
  --samples-per-dataset 1 \
  --min-input-tokens 0 \
  --max-input-tokens 50000 \
  --max-model-len 45056 \
  --gpu-memory-utilization 0.85 \
  --output "$LONGBENCH_ROOT/smoke_full_recompute.json"
```

### 5.2 HYPIC chunk 512

```bash
python examples/offline_inference/hypic_longbench.py \
  --model "$HYPIC_MODEL" \
  --data-dir "$LONGBENCH_DATA" \
  --config-dir "$LONGBENCH_CONFIG" \
  --reference-dir "$SGLANG_RESULTS" \
  --reference-prefix hypic_512 \
  --mode hypic \
  --chunk-size 512 \
  --max-cache-segments 96 \
  --seed 20260825 \
  --samples-per-dataset 1 \
  --min-input-tokens 0 \
  --max-input-tokens 50000 \
  --max-model-len 45056 \
  --gpu-memory-utilization 0.85 \
  --output "$LONGBENCH_ROOT/smoke_hypic_512.json"
```

### 5.3 HYPIC chunk 1024

```bash
python examples/offline_inference/hypic_longbench.py \
  --model "$HYPIC_MODEL" \
  --data-dir "$LONGBENCH_DATA" \
  --config-dir "$LONGBENCH_CONFIG" \
  --reference-dir "$SGLANG_RESULTS" \
  --reference-prefix hypic_1024 \
  --mode hypic \
  --chunk-size 1024 \
  --max-cache-segments 96 \
  --seed 20260825 \
  --samples-per-dataset 1 \
  --min-input-tokens 0 \
  --max-input-tokens 50000 \
  --max-model-len 45056 \
  --gpu-memory-utilization 0.85 \
  --output "$LONGBENCH_ROOT/smoke_hypic_1024.json"
```

建议的闸门判断：

- 四个数据集均有非空、可读输出；
- `finish_reason` 没有异常；
- QA 任务不出现大面积完全无关答案；
- 摘要长度和内容正常；
- 4 条样本宏平均与 SGLang 没有系统性大幅下降；
- 单条分数差异要结合答案和文本检查，不能因为某条 F1 为 0 就直接判断 runtime 错误。

`temperature=0` 仍不保证 SGLang 与 vLLM 逐 token 相同。不同 Attention/GDN kernel 的 BF16 数值差异可能改变贪心路径，摘要任务尤其敏感。因此以官方 metric 和完整数据集趋势为主，exact text match 作为辅助信号。

### 5.4 固定种子采样实测结果

静态 segment pool 实现完成后，使用 seed `20260824`、每个数据集 2 条，在
Qwen3.5-35B-A3B、TP=2、`max_model_len=45056`、
`gpu_memory_utilization=0.85`、`max_cache_segments=96` 的 Ascend 实机上执行上述
闸门。采样严格沿用各 SGLang mode 的原始 index；下面的差值均为 vLLM cold 减
SGLang，单位为百分点：

| 模式 | Qasper | HotpotQA | GovReport | MultiNews | 8 条均分差 | 平均绝对差 | 最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYPIC-512 | +0.000 | +0.000 | +1.959 | -0.325 | +0.408 | 0.571 | 2.639 |
| HYPIC-1024 | +0.000 | +0.000 | -1.130 | +0.417 | -0.178 | 0.387 | 1.897 |

这里 Qasper 和 HotpotQA 各列是两条样本的均值，两种 chunk 的 4 条 QA 样本都与
SGLang 零分差。HYPIC-512 和 HYPIC-1024 的 SGLang/vLLM cold 总体均分分别为
`42.523/42.931` 和 `42.934/42.756`，没有系统性下降，因此采样闸门通过。

同一请求的 warm hit 相对 cold 的平均绝对分差分别为 0.760 和 0.409，最大绝对分差
分别为 1.532 和 2.419；cold/warm token IDs 完全一致分别为 2/8 和 4/8。HYPIC 会用
BF16 public state 和 transition 重建前缀，命中路径与 cold 全量路径不保证 bitwise
相同，因此应同时检查官方分数和文本语义。此次所有 warm 输出均非空、可读，未出现
明显上下文丢失。

采样结果 JSON 可用以下字段复核：`index`、`sglang_score`、`vllm_cold_score`、
`vllm_warm_score`、`sglang_vllm_text_match` 和 `cold_warm_token_match`。只有采样闸门
通过后才启动下一节的完整实验；采样结论不能替代四个数据集的完整准确率。

## 6. 运行完整测试

完整 runner 是 `examples/offline_inference/hypic_longbench_full.py`。它按 dataset 顺序逐条写 JSONL，每条完成后 flush 和 `fsync`，每个 dataset 完成后更新 summary。

### 6.1 Full Recompute，batch 4

```bash
python examples/offline_inference/hypic_longbench_full.py \
  --model "$HYPIC_MODEL" \
  --data-dir "$LONGBENCH_DATA" \
  --config-dir "$LONGBENCH_CONFIG" \
  --output-dir "$VLLM_RESULTS" \
  --mode full_recompute \
  --chunk-size 0 \
  --run-name full_recompute \
  --max-model-len 45056 \
  --tensor-parallel-size 2 \
  --batch-size 4 \
  --gpu-memory-utilization 0.85
```

### 6.2 HYPIC chunk 512，batch 4

```bash
python examples/offline_inference/hypic_longbench_full.py \
  --model "$HYPIC_MODEL" \
  --data-dir "$LONGBENCH_DATA" \
  --config-dir "$LONGBENCH_CONFIG" \
  --output-dir "$VLLM_RESULTS" \
  --mode hypic \
  --chunk-size 512 \
  --run-name hypic_512 \
  --max-model-len 45056 \
  --tensor-parallel-size 2 \
  --batch-size 4 \
  --max-num-batched-tokens 45056 \
  --gpu-memory-utilization 0.85 \
  --max-cache-segments 96
```

### 6.3 HYPIC chunk 1024，batch 4

```bash
python examples/offline_inference/hypic_longbench_full.py \
  --model "$HYPIC_MODEL" \
  --data-dir "$LONGBENCH_DATA" \
  --config-dir "$LONGBENCH_CONFIG" \
  --output-dir "$VLLM_RESULTS" \
  --mode hypic \
  --chunk-size 1024 \
  --run-name hypic_1024 \
  --max-model-len 45056 \
  --tensor-parallel-size 2 \
  --batch-size 4 \
  --max-num-batched-tokens 45056 \
  --gpu-memory-utilization 0.85 \
  --max-cache-segments 96
```

提高 `--max-num-batched-tokens` 可以让更多长 prompt 同时 prefill，但会增大 activation
和 workspace 峰值。应先保持 45056，通过 NPU 峰值确认有余量后再逐步增加。

## 7. 后台串行运行

两卡机器不能同时加载三个 TP=2 engine，应串行执行：

```bash
tmux new-session -d -s vllm_longbench_e \
  "cd $VLLM_ASCEND_REPO && \
  python examples/offline_inference/hypic_longbench_full.py \
    --model $HYPIC_MODEL --data-dir $LONGBENCH_DATA \
    --config-dir $LONGBENCH_CONFIG --output-dir $VLLM_RESULTS \
    --mode full_recompute --chunk-size 0 --run-name full_recompute \
    --max-model-len 45056 --tensor-parallel-size 2 \
    --batch-size 4 --gpu-memory-utilization 0.85 \
    >> $VLLM_RESULTS/run.log 2>&1 && \
  python examples/offline_inference/hypic_longbench_full.py \
    --model $HYPIC_MODEL --data-dir $LONGBENCH_DATA \
    --config-dir $LONGBENCH_CONFIG --output-dir $VLLM_RESULTS \
    --mode hypic --chunk-size 512 --run-name hypic_512 \
    --max-model-len 45056 --tensor-parallel-size 2 \
    --batch-size 4 --max-num-batched-tokens 45056 \
    --gpu-memory-utilization 0.85 \
    --max-cache-segments 96 >> $VLLM_RESULTS/run.log 2>&1 && \
  python examples/offline_inference/hypic_longbench_full.py \
    --model $HYPIC_MODEL --data-dir $LONGBENCH_DATA \
    --config-dir $LONGBENCH_CONFIG --output-dir $VLLM_RESULTS \
    --mode hypic --chunk-size 1024 --run-name hypic_1024 \
    --max-model-len 45056 --tensor-parallel-size 2 \
    --batch-size 4 --max-num-batched-tokens 45056 \
    --gpu-memory-utilization 0.85 \
    --max-cache-segments 96 >> $VLLM_RESULTS/run.log 2>&1"
```

`&&` 保证某一阶段失败时不会掩盖错误并继续下一阶段。

## 8. 监控和完整性检查

查看进度：

```bash
tmux ls
tail -f "$VLLM_RESULTS/run.log"

grep -E 'DATASET_START|DATASET_DONE|RUN_DONE|Traceback|ERROR|OutOfMemory' \
  "$VLLM_RESULTS/run.log"
```

查看当前行数：

```bash
for mode in full_recompute hypic_512 hypic_1024; do
  for dataset in qasper gov_report hotpotqa multi_news; do
    file="$VLLM_RESULTS/$mode.$dataset.jsonl"
    if [[ -f "$file" ]]; then
      printf '%-16s %-12s ' "$mode" "$dataset"
      wc -l < "$file"
    fi
  done
done
```

完整行数应为：

```text
qasper      224
gov_report  300
hotpotqa    300
multi_news  294
```

检查最后一行没有损坏：

```bash
tail -1 "$VLLM_RESULTS/hypic_512.gov_report.jsonl" \
  | python -m json.tool >/dev/null
```

### 8.1 断点续跑注意事项

runner 会按 `index` 跳过已有结果：

- Full Recompute 没有跨请求 HYPIC 状态，可以安全续跑。
- HYPIC engine 重启后 cache 为空。跳过旧行可以完成数据集，但 cache history 与一次连续运行不同。

如果目标是严格可复现的 HYPIC 准确率，建议把 partial 文件移到备份目录后从该 mode 的 index 0 重跑；不要直接删除，保留现场便于定位失败点。

## 9. 输出文件

每种 mode、每个 dataset 一个 JSONL：

```text
full_recompute.qasper.jsonl
full_recompute.gov_report.jsonl
hypic_512.qasper.jsonl
hypic_1024.qasper.jsonl
...
```

每行主要字段：

- `dataset`、`index`、`pred`、`answers`；
- 官方单样本 `score`；
- `length` 和 `input_tokens`；
- `segments`；
- `output_ids`、`completion_tokens` 和 `finish_reason`；
- `latency`。

`cached_tokens` 当前固定为 0，不代表 HYPIC 没有命中，不能用于命中率或性能分析。

每个 dataset 完成后生成或更新：

```text
full_recompute.summary.json
hypic_512.summary.json
hypic_1024.summary.json
```

summary 包含 dataset score、macro average，以及 `0-4k`、`4-8k`、`8k+` 长度桶。

## 10. 与 SGLang 结果比较

先确认两侧行数相同，再比较每个 dataset 和宏平均，不要只看总分。下面的脚本直接读取 JSONL 中已经计算好的单样本 score：

```bash
python - "$SGLANG_RESULTS" "$VLLM_RESULTS" <<'PY'
import json
import pathlib
import sys

reference = pathlib.Path(sys.argv[1])
candidate = pathlib.Path(sys.argv[2])
modes = ("full_recompute", "hypic_512", "hypic_1024")
datasets = ("qasper", "gov_report", "hotpotqa", "multi_news")

def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

for mode in modes:
    ref_scores = []
    vllm_scores = []
    print(f"\n[{mode}]")
    for dataset in datasets:
        ref = load(reference / f"{mode}.{dataset}.jsonl")
        out = load(candidate / f"{mode}.{dataset}.jsonl")
        if len(ref) != len(out):
            raise RuntimeError(f"row count mismatch: {mode}/{dataset}: {len(ref)} != {len(out)}")
        ref_score = 100 * sum(float(row["score"]) for row in ref) / len(ref)
        out_score = 100 * sum(float(row["score"]) for row in out) / len(out)
        ref_scores.append(ref_score)
        vllm_scores.append(out_score)
        print(f"{dataset:12s} sglang={ref_score:6.2f} vllm={out_score:6.2f} delta={out_score-ref_score:+6.2f}")
    ref_macro = sum(ref_scores) / len(ref_scores)
    out_macro = sum(vllm_scores) / len(vllm_scores)
    print(f"{'macro':12s} sglang={ref_macro:6.2f} vllm={out_macro:6.2f} delta={out_macro-ref_macro:+6.2f}")
PY
```

本次使用的 SGLang Ascend B4 reference 为：

| 模式 | Qasper | GovReport | HotpotQA | MultiNews | Macro |
|---|---:|---:|---:|---:|---:|
| Full Recompute | 47.48 | 31.41 | 72.02 | 23.11 | 43.50 |
| HYPIC-512 | 48.24 | 31.39 | 72.10 | 23.20 | 43.73 |
| HYPIC-1024 | 48.12 | 31.26 | 71.56 | 23.13 | 43.52 |

参考值只适用于相同数据、模型、prompt 和 scoring 版本。任何一项变化都应重新生成基线。

### 10.1 完整实测结果

Qwen3.5-35B-A3B、TP=2 的完整 LongBench-E 实测结果如下。HYPIC 使用 batch=4、
`max_num_batched_tokens=45056`、`gpu_memory_utilization=0.85` 和 96-slot 静态
segment pool；每个 mode 都包含 Qasper 224 条、GovReport 300 条、HotpotQA 300 条和
MultiNews 294 条：

| 模式 / Runtime | Qasper | GovReport | HotpotQA | MultiNews | Macro |
|---|---:|---:|---:|---:|---:|
| Full Recompute / SGLang | 47.48 | 31.41 | 72.02 | 23.11 | 43.50 |
| Full Recompute / vLLM | 48.36 | 31.48 | 71.90 | 23.13 | 43.72 |
| 差值 | +0.88 | +0.07 | -0.12 | +0.02 | +0.22 |
| HYPIC-512 / SGLang | 48.24 | 31.39 | 72.10 | 23.20 | 43.73 |
| HYPIC-512 / vLLM | 48.48 | 31.52 | 71.72 | 23.07 | 43.70 |
| 差值 | +0.24 | +0.13 | -0.38 | -0.13 | -0.03 |
| HYPIC-1024 / SGLang | 48.12 | 31.26 | 71.56 | 23.13 | 43.52 |
| HYPIC-1024 / vLLM | 47.84 | 31.43 | 71.97 | 23.06 | 43.58 |
| 差值 | -0.28 | +0.17 | +0.41 | -0.07 | +0.06 |

三种 mode 的 Macro 差值绝对值均不超过 0.22，两个 HYPIC mode 分别为 -0.03 和
+0.06；没有数据集出现系统性准确率下降。运行中覆盖了 batch=4 和约 37k-token 的
长输入，未出现 NPU OOM、cache divergence 或 `mamba_pool exhausted`。因此完整准确率
验证通过。

本次 HYPIC-512 的 Qasper 在 188/224 处为执行固定样本验证而主动中断，随后从结果
JSONL 续跑；其余 HYPIC 数据集以及完整 HYPIC-1024 均在各自 engine 生命周期内连续
完成。断点后的进程内 cache history 会重新建立，因此该结果用于准确率验证，不用于
严格复现连续 rolling-cache 的命中率或吞吐。

结果目录：

- Full Recompute：`/data/hypic-validation/longbench_v1/results_e_vllm_ascend_b1`；
- HYPIC-512/1024：`/data/hypic-validation/longbench_v1/results_e_vllm_ascend_static_b4`。

## 11. 常见异常排查

### 11.1 Qasper 分数异常

先检查单条内容，而不是只看分数：

```bash
python - "$SGLANG_RESULTS/hypic_512.qasper.jsonl" \
  "$VLLM_RESULTS/hypic_512.qasper.jsonl" <<'PY'
import json
import sys

def rows(path):
    return {int(x["index"]): x for x in map(json.loads, open(path))}

left, right = rows(sys.argv[1]), rows(sys.argv[2])
for index in sorted(left):
    if abs(float(left[index]["score"]) - float(right[index]["score"])) >= 0.25:
        print(index, left[index]["score"], right[index]["score"])
        print("SGLang:", left[index]["pred"])
        print("vLLM:  ", right[index]["pred"])
        print("answers:", right[index]["answers"])
        break
PY
```

重点确认：

- 使用的是 `qasper_e.jsonl`，不是普通 Qasper；
- reference 的 `index` 与 dataset 原始行一致；
- chat template 使用 `enable_thinking=False`；
- QA metric 取所有可接受答案中的最大值；
- 两个 runtime 都回答错同一条时，不是适配差异。

### 11.2 warm 输出明显失去上下文

优先检查：

1. sparse absolute positions 是否正确；
2. slot mapping 是否按绝对位置重建；
3. hit Attention KV 是否写回标准 paged cache；
4. GDN 最终 state 是否写入当前 decode slot；
5. scheduler/device cache 是否 divergence。

### 11.3 NPU OOM

- 确认启动日志已经把 HYPIC pool 计入模型占用，并记录 KV cache 和 peak activation；
- 检查 `max_cache_segments >= ceil(max_num_batched_tokens / chunk_size) - 1`；
- 在满足最低 slot 数后，必要时降低 `gpu_memory_utilization`，让 vLLM 缩小普通 KV cache；
- 用最长 prompt 和 batch=4 重现峰值，同时扫描 NPU OOM，不能只看 engine 启动时的剩余显存。

### 11.4 结果文件条数不够

检查 `run.log` 中第一个 `Traceback` 或 OOM。串行命令使用 `&&` 时，前一 mode 失败后后续 mode 不会启动，这是预期行为。

## 12. 实验报告至少包含

- vLLM、vLLM Ascend、CANN、torch_npu 和 `sgl-kernel-npu` 版本；
- 模型路径或不可变 revision；
- NPU 型号、卡数和 TP；
- Full/HYPIC batch size；
- chunk、seam、max cache segments、memory utilization；
- 数据集和官方配置 revision；
- 每个 dataset 分数、macro 和长度桶；
- SGLang reference 的对应数值和 delta；
- 是否连续运行、是否从 checkpoint 恢复；
- 任何 OOM、cache divergence、异常 finish reason 或人工排除的样本。
