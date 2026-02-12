from openai import AsyncOpenAI

from models import LLMResponse, ModelConfig

SYSTEM_PROMPT = """\
你的任务是作为一个成语接龙玩家，与玩家玩成语接龙游戏，并且尝试在游戏中击败玩家。在回复中直接回复成语。或者在判断无法接龙时回复“我输了”

成语接龙游戏规则：

双方轮流说一个成语。成语必须真实存在，不可生造或重复使用。新成语的首字需与上一个成语的尾字音或字相同。无法接龙者判为失败。需要直接回复"我输了"。

请以 json 格式回复：{"word": "你的成语", "success": true}，如果无法接龙则回复 {"word": "", "success": false}。"""

LLM_TIMEOUT = 60.0


def build_messages(
    history_words: list[str], player: str, start_word: str
) -> list[dict[str, str]]:
    """构建当前模型视角的 messages 列表。

    - history_words 按实际出场顺序排列（A1, B1, A2, B2, ...）
    - 当前 player 的历史输出为 assistant，对手的历史输出为 user

    Model A 视角: user(start) → assistant(A1) → user(B1) → ...
    Model B 视角: user(A1) → assistant(B1) → user(A2) → ...
      注意：B 视角下 start_word 与 A 的第一步会连续两个 user，
      因此对 B 跳过 start_word，直接从 A 的第一步作为首条 user 消息。
    """
    if player == "A":
        system_content = SYSTEM_PROMPT
    else:
        # B 视角：将起始成语补充到 system prompt，避免丢失上下文
        system_content = f"{SYSTEM_PROMPT}\n\n本局起始成语为「{start_word}」。"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]

    if player == "A":
        # A 视角：start_word 作为第一条 user，然后交替 assistant/user
        messages.append({"role": "user", "content": start_word})
        for i, word in enumerate(history_words):
            role = "assistant" if i % 2 == 0 else "user"
            messages.append({"role": role, "content": word})
    else:
        # B 视角：A 的输出为 user，B 的输出为 assistant
        # history_words[0]=A1, [1]=B1, [2]=A2, [3]=B2, ...
        # 起始成语已在 system prompt 中，A1 作为首条 user，严格交替
        if not history_words:
            messages.append({"role": "user", "content": start_word})
        else:
            for i, word in enumerate(history_words):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": word})

    return messages


async def call_llm(
    config: ModelConfig,
    history_words: list[str],
    player: str,
    start_word: str,
) -> LLMResponse:
    """调用 LLM 获取结构化输出。异常向上抛出由调用方处理。"""
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    messages = build_messages(history_words, player, start_word)

    completion = await client.beta.chat.completions.parse(
        model=config.model,
        messages=messages,
        response_format=LLMResponse,
        timeout=LLM_TIMEOUT,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM 返回结果解析失败")
    return parsed
