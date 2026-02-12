# 成语接龙 LLM 对战平台

## 项目概述

基于 FastAPI 的成语接龙对战平台，用户接入两个大语言模型（LLM），让它们进行成语接龙对战。两个模型互相生成复杂且对方难以接续的成语，平台负责裁判和验证。

## 技术栈

- **框架**: FastAPI
- **LLM 调用**: OpenAI-compatible API（通过 `openai` Python SDK，支持自定义 `base_url`）
- **响应方式**: SSE（Server-Sent Events）流式推送，每回合实时推送
- **数据验证**: Pydantic v2
- **数据库**: SQLite + Tortoise ORM
- **前端**: 纯 HTML 单页面（内联 CSS/JS，简洁实用）
- **成语词库**: `chengyu.json`（同级目录，30000+ 成语的 JSON 数组）

## 核心规则

### 游戏流程

1. 用户提供：模型A配置、模型B配置、起始成语
2. 平台验证起始成语是否在词库中
3. 模型A先手，根据起始成语进行接龙
4. 模型B接续模型A的输出，交替进行
5. 游戏结束条件（满足任一即结束）：
   - 某模型主动认输（`success: false`）
   - 某模型输出不合规（判定为输）
   - 达到最大回合数上限：**30 回合**（平局）

### 成语验证规则（平台裁判）

模型每次输出的成语必须同时满足以下三项，否则判定该模型输：

1. **存在性验证**：成语必须存在于 `chengyu.json` 词库中
2. **接龙验证**：成语首字必须与上一个成语的末字相同（同字，非同音）
3. **唯一性验证**：该成语未在本局对战历史中出现过

### 验证回合机制（next_word）

模型每次出招时，除了给出自己的成语（`word`），还必须给出一个"验证成语"（`next_word`）——即它认为可以接在自己成语后面的下一个成语。

当对手失败（认输/违规/调用失败）时：
1. 取上一位玩家提供的 `next_word`
2. 验证 `next_word`：存在于词库 + 首字匹配上一位 word 的末字 + 未使用过
3. **合法** → 上一位玩家胜（证明自己有路可走）
4. **不合法** → 反转！失败方胜（上一位的成语是死胡同）

若第一回合就失败（无 `next_word` 可验证），保持原判定逻辑。

### 模型输出格式（结构化输出）

模型必须返回严格的 JSON 结构：

```json
{
  "word": "成语内容",
  "next_word": "你认为可接龙的下一个成语",
  "success": true
}
```

- `word`（string）：模型认为应该接龙的成语
- `next_word`（string）：模型认为可以接在 `word` 后面的下一个成语，用于验证回合机制
- `success`（boolean）：模型是否完成了接龙。若模型认为无法接续，设为 `false` 表示认输

使用 OpenAI SDK 的 `response_format` 结构化输出功能，确保 JSON 格式可靠可验证。

## API 设计

### 接口：`POST /battle`

#### 请求体

```json
{
  "model_a": {
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-xxx",
    "model": "gpt-4o"
  },
  "model_b": {
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-xxx",
    "model": "claude-sonnet-4-20250514"
  },
  "start_word": "一心一意"
}
```

#### 响应：SSE 流式推送

每个回合推送一个 SSE event，`data` 为 JSON：

**回合事件** (`event: round`)：
```json
{
  "round": 1,
  "player": "A",
  "model": "gpt-4o",
  "word": "意气风发",
  "next_word": "发愤图强",
  "success": true,
  "valid": true,
  "message": ""
}
```

- `valid` 为平台验证结果。若为 `false`，`message` 说明原因（如"成语不在词库中"、"首字不匹配"、"成语已使用过"）

**结束事件** (`event: result`)：
```json
{
  "winner": "A",
  "reason": "模型B认输",
  "rounds": 12,
  "history": ["一心一意", "意气风发", "发愤图强", ...],
  "battle_id": 1
}
```

- `winner`：`"A"` / `"B"` / `"draw"`（达到30回合平局）
- `reason`：结束原因描述
- `battle_id`：数据库记录 ID，可用于查询详情

### 接口：`GET /battles`

返回所有对战记录列表（按时间倒序）。

### 接口：`GET /battles/{id}`

返回单条对战记录详情。

### 接口：`GET /`

返回前端 HTML 页面。

## 模型 System Prompt

两个模型使用相同的 system prompt（详见 `llm.py` 中的 `SYSTEM_PROMPT`），核心要点：

- 说明成语接龙规则（同字非同音）
- 说明验证回合机制：每次出招需同时给出 `next_word`，对手失败时平台会验证 `next_word`，无效则反判
- 对战策略引导：选末字生僻的成语，同时确保自己有后路
- JSON 格式要求：`{"word": "...", "next_word": "...", "success": true}`

## 模型上下文构建

每次调用模型时，messages 结构如下：

- `system`: 上述 system prompt
- 交替的 `user` / `assistant` 消息，内容为历史对话中每轮的 `word`（纯成语文本，不是 JSON）
- 当前模型的历史输出为 `assistant`，对手的历史输出为 `user`

例如模型A视角（起始成语"一心一意"，已进行2轮）：

```
system: [system prompt]
user: "一心一意"          ← 起始成语作为 user 消息
assistant: "意气风发"      ← A 自己的第1轮输出
user: "发愤图强"          ← B 的第1轮输出
```

## 数据库设计（SQLite + Tortoise ORM）

### Battle 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | IntField (PK, auto) | 主键 |
| model_a_name | CharField | 模型A名称（如 `gpt-4o`） |
| model_b_name | CharField | 模型B名称（如 `claude-sonnet-4-20250514`） |
| start_word | CharField | 起始成语 |
| history | JSONField | 接龙过程，JSON 数组，每项结构见下方 |
| winner | CharField | 最终判定：`"A"` / `"B"` / `"draw"` |
| reason | CharField | 判定原因 |
| created_at | DatetimeField | 创建时间，自动填充 |

### history JSON 结构

```json
[
  {
    "round": 1,
    "player": "A",
    "model": "gpt-4o",
    "word": "意气风发",
    "next_word": "发愤图强",
    "success": true,
    "valid": true,
    "message": ""
  },
  ...
]
```

### 判定原因枚举（reason）

- `"模型A认输"` / `"模型B认输"` — 模型主动 `success: false`
- `"模型A成语不在词库中"` / `"模型B成语不在词库中"` — 存在性验证失败
- `"模型A首字不匹配"` / `"模型B首字不匹配"` — 接龙验证失败
- `"模型A成语重复使用"` / `"模型B成语重复使用"` — 唯一性验证失败
- `"模型A调用失败"` / `"模型B调用失败"` — LLM 调用异常
- `"模型A无法证明可以继续接龙"` / `"模型B无法证明可以继续接龙"` — 验证回合反转判定
- `"达到最大回合数"` — 30 回合平局

### API：获取历史记录

**`GET /battles`** — 返回所有对战记录列表

**`GET /battles/{id}`** — 返回单条对战记录详情

## 前端界面

纯 HTML 单文件（`index.html`），由 FastAPI 静态文件 / 模板直接提供服务。界面简洁实用，不引入任何前端框架。

### 页面布局

1. **配置区域**（顶部）
   - 模型A配置：base_url、API Key、模型名称
   - 模型B配置：base_url、API Key、模型名称
   - 起始成语输入框
   - 「开始对战」按钮

2. **对战展示区域**（中部）
   - 左右双栏 / 对话气泡形式展示双方接龙过程
   - 每条显示：回合数、模型名、成语、验证状态
   - 无效成语高亮标红，附原因说明
   - 实时滚动展示（SSE 驱动）

3. **结果区域**（底部）
   - 对战结束后显示：胜者、原因、总回合数

4. **历史记录区域**
   - 展示历史对战列表（从 `GET /battles` 获取）
   - 可点击查看详情

### 技术实现

- 使用 `EventSource` API 消费 SSE 流
- 原生 `fetch` 调用 REST API
- 内联 CSS 样式，简洁美观即可

## 项目结构

```
.
├── CLAUDE.md
├── chengyu.json          # 成语词库（30000+ 成语数组）
├── main.py               # FastAPI 应用入口、路由、生命周期
├── models.py             # Pydantic 请求/响应模型
├── db_models.py          # Tortoise ORM 数据库模型
├── battle.py             # 对战核心逻辑
├── validator.py          # 成语验证逻辑
├── llm.py                # LLM 调用封装
├── index.html            # 前端页面
└── requirements.txt      # 依赖
```

## 依赖

- `fastapi`
- `uvicorn`
- `openai` (OpenAI Python SDK, 用于调用 OpenAI-compatible API)
- `pydantic`
- `sse-starlette` (FastAPI SSE 支持)
- `tortoise-orm` (异步 ORM)
- `aiosqlite` (SQLite 异步驱动)

## 开发规范

- 使用 Python 3.11+
- 类型注解完整
- 异步编程（async/await）
- 错误处理覆盖 LLM 调用超时、格式异常等边界情况
- LLM 调用失败（网络错误、超时等）视为该模型输