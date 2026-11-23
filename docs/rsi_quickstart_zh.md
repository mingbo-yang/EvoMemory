# 在本地运行 EvoMemory 双分支搜索

这个版本的入口是 `python -m rsi_paired`。一次工程验收用六个不同的 ALFWorld 训练实例，学习、初选和双分支比较各两题。初选产生三个候选。若触发 Evolver 策略修改，旧策略与新策略都回到**初选三个候选生成之前**的机制与学习记忆，各自重新生成三个候选。双方看到同一份已封存的初选证据，按相同任务顺序和配对种子评分。每个候选槽位最多有三次评分前的语法或接口返修机会。两组各自的最优方案都可以选择母体或初选赢家。新组最优多解出至少一道题，才采纳新的提案策略。

## 1. 安装并检查入口

需要 Linux、Python 3.10 或 3.11、Git 和系统库 libseccomp。在 Debian/Ubuntu 上库名为 `libseccomp2`。开发分支包含本次入口，请在克隆后的仓库根目录执行后面的命令。

```bash
git clone --branch rsi-original-parent-paired-v2 https://github.com/mingbo-yang/EvoMemory.git
cd EvoMemory
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[alfworld,models,dev]'
python -m rsi_paired --help
```

## 2. 准备六题和本地模型登记

`prepare_assets` 会下载固定版本的 ALMA 和 ALFWorld 资产。终端可能显示公共映射有 200 题，实际运行题目由接下来的六题清单确定。若需要排除自己之前使用的实例，可给 `prepare-smoke` 传入一个或多个 `--exclude` 清单。每份排除清单是 JSON 数组，每项包含 `group_id` 和 `game_sha256`。

```bash
python -m scripts.prepare_assets alfworld
python -m rsi_paired prepare-smoke \
  --games data/downloads/alfworld-tw-0.4.2.zip \
  --metadata data/downloads/json_2.1.1_json.zip \
  --seed 42 --out runs/panel.json
```

使用已启动的本机模型服务。下面的服务名、端口和上下文是 Gemma 的例子，请按实际部署替换。检查 `/models` 中的模型 ID、checkpoint 路径和 `max_model_len`。角色调用还需要服务支持 OpenAI 兼容的 JSON Schema 结构化输出。将 `MODEL_CONTEXT_TOKENS` 设为服务实际支持的正整数。将 `TMPDIR` 指向可写的本地磁盘。

```bash
export EVOMEMORY_BASE_URL=http://127.0.0.1:8141/v1
export EVOMEMORY_MODEL_NAME=gemma-4-31b-it
export EVOMEMORY_MODEL_PATH=/absolute/path/to/served/checkpoint
export TMPDIR=/tmp EVOMEMORY_ENABLE_OPTIONAL_MODELS=0
curl --fail --noproxy '*' "${EVOMEMORY_BASE_URL}/models"
export MODEL_CONTEXT_TOKENS=262144

python -m scripts.register_model --model "$EVOMEMORY_MODEL_PATH" \
  --output runs/checkpoint.json
python - <<'PY'
import json, os
from pathlib import Path

config = {
    "base_url": os.environ["EVOMEMORY_BASE_URL"],
    "model_name": os.environ["EVOMEMORY_MODEL_NAME"],
    "tokenizer_path": os.environ["EVOMEMORY_MODEL_PATH"],
    "context_tokens": int(os.environ["MODEL_CONTEXT_TOKENS"]),
    "checkpoint_manifest": str(Path("runs/checkpoint.json").resolve()),
}
Path("runs/local-model.json").write_text(json.dumps(config), encoding="utf-8")
PY
```

checkpoint 只登记一次，记录元数据哈希和权重分片大小。推理复用现有模型服务。示例只开放本地文本生成能力，如启用其他本地模型，要单独记录这个条件。

## 3. 运行和复核

若要验收完整双分支路径，运行以下命令。`--exercise-pair` 强制执行这条工程路径，仍会记录模型自然触发与否。研究自然触发时，删去该参数并使用新的输出目录。

```bash
python -m rsi_paired smoke --manifest runs/panel.json \
  --model-config runs/local-model.json --out runs/smoke-paired --exercise-pair
python -m rsi_paired.audit runs/smoke-paired --require-complete
```

运行期间可在另一个终端执行 `python -m rsi_paired.audit runs/smoke-paired` 查看实际完成的学习任务、封存槽位和原生执行。完成双分支验收需要同时看到审计状态 `verified`、`coverage.paired_sealed` 为真、九个已封存槽位，以及 `runs/smoke-paired/batch/commit.json` 中 `state.history[-1].status` 等于 `paired_complete`。未触发策略更新也能自然结束一批，因此只看进程退出不足以判断双分支是否跑过。若历史批次缺少可核对原始字节的评测文件，完成后审计会显示 `unverified_native_wire`，原生分数仍可查阅。

一次双分支验收最多计划 26 个原生 Episode、9 个候选槽位和 36 次机制稿件提交，额外的角色调用和基础设施重试另有收据。恢复时使用同一版本、清单、模型配置、参数和输出目录；进程中断且原生请求结果不明时，须先核对未完成收据。CPU 回归入口是 `python -m pytest -q`。原生轨迹、提示词与本地路径保存在私有的 `runs/` 目录。

工程验收检验程序是否按协议运行。效果结论还需要独立任务和重复实验。[英文协议及审计说明](rsi_paired.md)
