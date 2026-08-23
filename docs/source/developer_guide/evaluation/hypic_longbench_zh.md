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

当前 HYPIC 实现要求 `max_num_seqs=1`，所以推荐：

- Full Recompute 使用 batch=4，与 SGLang B4 基线更接近；
- HYPIC 使用 batch=1。

这可以比较任务准确率，但 HYPIC 的请求 admission 和 cache hit 历史不与 SGLang rolling B4 完全相同。报告中必须注明这一点，不应把该实验当成严格的吞吐或逐 token 等价对拍。

## 2. 已验证配置

以下是本次 Ascend 实机使用的配置：

| 参数 | Full Recompute | HYPIC-512/1024 |
|---|---:|---:|
| 模型 | Qwen3.5-35B-A3B | Qwen3.5-35B-A3B |
| NPU / TP | 2 / 2 | 2 / 2 |
| max model length | 45056 | 45056 |
| batch size | 4 | 1 |
| GPU/NPU memory utilization | 0.85 | 0.70 |
| max cache segments | - | 256 |
| seam sink tokens | - | 8 |

`max_model_len=45056` 是根据四个数据集中最长约 41.2k-token 的输入留余量得到的。不要用默认 8192 跑完整 LongBench-E。

HYPIC cache tensor 是按需分配的。256 segments 时降低 vLLM KV cache 的内存比例，可以为 Full Attention public KV 和 GDN state 留出空间。不同模型、NPU 容量或 kernel 版本应重新测量，不能照抄 0.70 后假设一定安全。

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
  --max-cache-segments 256 \
  --seed 20260825 \
  --samples-per-dataset 1 \
  --min-input-tokens 0 \
  --max-input-tokens 50000 \
  --max-model-len 45056 \
  --gpu-memory-utilization 0.70 \
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
  --max-cache-segments 256 \
  --seed 20260825 \
  --samples-per-dataset 1 \
  --min-input-tokens 0 \
  --max-input-tokens 50000 \
  --max-model-len 45056 \
  --gpu-memory-utilization 0.70 \
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

### 6.2 HYPIC chunk 512，batch 1

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
  --batch-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-cache-segments 256
```

### 6.3 HYPIC chunk 1024，batch 1

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
  --batch-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-cache-segments 256
```

runner 会拒绝 HYPIC `--batch-size` 大于 1，避免把正确性限制误当作调优建议绕过。

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
    --batch-size 1 --gpu-memory-utilization 0.70 \
    --max-cache-segments 256 >> $VLLM_RESULTS/run.log 2>&1 && \
  python examples/offline_inference/hypic_longbench_full.py \
    --model $HYPIC_MODEL --data-dir $LONGBENCH_DATA \
    --config-dir $LONGBENCH_CONFIG --output-dir $VLLM_RESULTS \
    --mode hypic --chunk-size 1024 --run-name hypic_1024 \
    --max-model-len 45056 --tensor-parallel-size 2 \
    --batch-size 1 --gpu-memory-utilization 0.70 \
    --max-cache-segments 256 >> $VLLM_RESULTS/run.log 2>&1"
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

- 降低 HYPIC `gpu_memory_utilization`，为按需 segment cache 留空间；
- 降低 `max_cache_segments`；
- 先用 128 segments 确认正确性，再逐步增加；
- 同时观察 vLLM KV cache、HYPIC public KV/GDN state 和长 prefill workspace，不能只看 engine 启动时的剩余显存。

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
