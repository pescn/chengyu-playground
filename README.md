# 成语接龙 LLM 对战平台

基于 FastAPI 的成语接龙对战平台。接入两个大语言模型，让它们进行成语接龙对战，平台负责裁判和验证。同时提供 GRPO 强化学习训练数据生成和奖励函数，可直接对接 [veRL](https://github.com/volcengine/verl) 框架。

## 快速开始

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 即可使用 Web 界面进行 LLM 对战。

Docker 部署：

```bash
docker build -t chengyu-playground .
docker run -p 8000:8000 -v chengyu-data:/app/data chengyu-playground
```

## 项目结构

```
.
├── main.py               # FastAPI 应用入口、路由
├── battle.py             # 对战核心逻辑
├── benchmark.py          # 批量对战 benchmark
├── llm.py                # LLM 调用封装 & prompt 构建
├── validator.py          # 成语验证（存在性/接龙/唯一性）
├── models.py             # Pydantic 请求/响应模型
├── db_models.py          # Tortoise ORM 数据库模型
├── index.html            # 前端页面
├── chengyu.json          # 成语词库（30000+ 条）
├── idiom.json            # 成语词库（含拼音信息）
├── reward.py             # GRPO 单步奖励函数核心模块
├── reward_verl.py        # veRL 奖励函数适配器
├── gen_dataset.py        # 训练数据集生成脚本
└── pyproject.toml        # 项目依赖
```

## 对战规则

1. 用户提供两个模型的配置和起始成语
2. 模型 A 先手，之后 A/B 交替出招
3. 每次出招需同时给出 `next_word`（能接在自己成语后面的下一个成语），用于验证回合机制
4. 平台验证每步的合法性（存在性 + 接龙匹配 + 唯一性）
5. 当一方失败时，平台检查上一位的 `next_word` 是否合法——不合法则反转判定

### 验证模式

| 模式 | 说明 |
|------|------|
| `same_char` | 首字必须与上一个成语末字相同（默认） |
| `homophone` | 首字读音与上一个成语末字相同即可 |
| `same_char_sound` | 首字相同且读音一致（注意多音字） |

---

## GRPO 强化学习训练

本项目提供完整的 veRL GRPO 训练支持。训练流程：

```
生成对战数据集 (gen_dataset.py)  →  veRL GRPO 训练 (reward_verl.py)  →  用新模型重新生成数据（迭代）
```

### 1. 生成训练数据集

`gen_dataset.py` 通过两个模型实时对战来收集训练样本。每步在调用 LLM **之前**将当前游戏状态构建为一条训练数据。LLM 的实际回复仅驱动游戏推进，不存入训练数据——veRL 训练时由 policy 自行生成回复，再由奖励函数打分。

```bash
uv pip install pyarrow  # 额外依赖

python gen_dataset.py \
    --base-url http://172.16.56.3:2998/v1 \
    --api-key sk-xxx \
    --model-a qwen3-max \
    --model-b doubao-seed-1-6 \
    --num-games 100 \
    --max-concurrency 5 \
    --output dataset.parquet
```

双方也可使用不同的 API 地址和 Key：

```bash
python gen_dataset.py \
    --base-url-a http://host-a/v1 --api-key-a sk-aaa --model-a model-a \
    --base-url-b http://host-b/v1 --api-key-b sk-bbb --model-b model-b \
    --num-games 50
```

完整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | — | LLM API 地址（双方共用） |
| `--api-key` | — | API Key（双方共用） |
| `--base-url-a/b` | — | 模型 A/B 独立 API 地址（优先于共用） |
| `--api-key-a/b` | — | 模型 A/B 独立 API Key（优先于共用） |
| `--model-a` | 必填 | 对战模型 A 名称 |
| `--model-b` | 必填 | 对战模型 B 名称 |
| `--num-games` | 100 | 对局数量 |
| `--max-concurrency` | 5 | 最大并发对局数 |
| `--validation-mode` | same_char | 验证模式 |
| `--system-prompt` | 内置默认 | 自定义系统提示词 |
| `--output` | dataset.parquet | 输出文件路径 |

检查生成的数据：

```bash
python -c "import pandas as pd; df=pd.read_parquet('dataset.parquet'); print(f'样本数: {len(df)}'); print(df.head())"
```

### 2. 数据集格式

Parquet 文件每行一条训练样本，4 列均为字符串（veRL 的 `RLHFDataset` 会自动 `json.loads`）：

| 列名 | 内容 |
|------|------|
| `data_source` | `"chengyu"` |
| `prompt` | messages 数组的 JSON（模型输入上下文） |
| `reward_model` | `{"style":"rule","ground_truth":"上一个成语"}` |
| `extra_info` | `{"previous_word":"...","used_words":[...],"round_num":N,"validation_mode":"..."}` |

`prompt` 示例（第 3 回合 B 视角）：

```json
[
    {"role": "system", "content": "你是成语接龙玩家...本局起始成语为「一心一意」。"},
    {"role": "user", "content": "意气风发"},
    {"role": "assistant", "content": "发愤图强"},
    {"role": "user", "content": "强词夺理"}
]
```

### 3. 奖励函数

**`reward.py`** — 核心奖励计算，公式：`total = round_penalty + compliance + strategy + foresight`

| 组件 | 分值 | 说明 |
|------|------|------|
| `round_penalty` | -0.1 | 每轮固定惩罚，激励快速终结 |
| `compliance` | -1.0 ~ +0.3 | JSON 解析失败 -1.0 / 认输 -1.0 / 成语不存在 -0.8 / 接龙不匹配 -0.8 / 重复 -0.6 / 全部通过 +0.3 |
| `strategy` | 0 ~ +0.5 | 对手可接成语越少越好：0 个 +0.5 / ≤5 +0.3 / ≤20 +0.1 |
| `foresight` | 0 / +0.3 | next_word 合法 +0.3 |

**`reward_verl.py`** — veRL 适配器，提供 `compute_score()` 入口函数。

### 4. veRL 训练配置

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=dataset.parquet \
    data.val_files=dataset.parquet \
    reward.custom_reward_function.path=/path/to/reward_verl.py \
    reward.custom_reward_function.name=compute_score \
    ...
```

`reward_verl.py` 运行时需要同目录下的文件：`reward.py`、`validator.py`、`chengyu.json`、`idiom.json`。

### 5. 迭代训练示例

```bash
# 第 1 轮：基座模型对战生成初始数据
python gen_dataset.py \
    --base-url http://vllm-server:8000/v1 --api-key sk-xxx \
    --model-a base-model --model-b base-model \
    --num-games 200 --output round1.parquet

# 第 1 轮 veRL GRPO 训练
python3 -m verl.trainer.main_ppo \
    data.train_files=round1.parquet \
    reward.custom_reward_function.path=reward_verl.py \
    reward.custom_reward_function.name=compute_score ...

# 第 2 轮：部署训练后的模型，重新生成数据并训练
python gen_dataset.py \
    --base-url http://vllm-server:8000/v1 --api-key sk-xxx \
    --model-a trained-r1 --model-b trained-r1 \
    --num-games 200 --output round2.parquet

# ...以此迭代
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 前端页面 |
| POST | `/battle` | 发起对战（SSE 流式返回） |
| POST | `/benchmark` | 批量 benchmark（SSE 流式返回） |
| GET | `/battles` | 对战记录列表 |
| GET | `/battles/{id}` | 对战详情 |

## 依赖

- Python >= 3.12
- FastAPI、Uvicorn、OpenAI SDK、Tortoise ORM、aiosqlite、sse-starlette
- pyarrow（仅 `gen_dataset.py` 需要）
