from openai import AsyncOpenAI

from models import LLMResponse, ModelConfig

DEFAULT_SYSTEM_PROMPT = """\
你是成语接龙玩家。对手给你一个成语，你必须用该成语的最后一个字作为首字，说出一个新的成语。

规则：
1. 新成语的第一个字必须与上一个成语的最后一个字完全相同（同字，不是同音）
2. 成语必须真实存在，不可编造
3. 不可重复使用已出现过的成语
4. 无法接龙则认输

回复要求：
- word：你接龙的成语（必须以对手成语末字开头）
- next_word：一个能接在你的成语后面的成语（以你的成语末字开头），用于证明你的成语不是死路
- success：能接龙为true，无法接龙为false

策略提示：选末字生僻的成语来增加对手难度，但你自己必须知道至少一个能接上的成语（填入next_word）。"""

LLM_TIMEOUT = 60.0


def build_messages(
    history_words: list[str],
    player: str,
    start_word: str,
    system_prompt: str = "",
) -> list[dict[str, str]]:
    """构建当前模型视角的 messages 列表。

    - history_words 按实际出场顺序排列（A1, B1, A2, B2, ...）
    - 当前 player 的历史输出为 assistant，对手的历史输出为 user

    Model A 视角: user(start) → assistant(A1) → user(B1) → ...
    Model B 视角: user(A1) → assistant(B1) → user(A2) → ...
      注意：B 视角下 start_word 与 A 的第一步会连续两个 user，
      因此对 B 跳过 start_word，直接从 A 的第一步作为首条 user 消息。
    """
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if player == "A":
        system_content = prompt
    else:
        # B 视角：将起始成语补充到 system prompt，避免丢失上下文
        system_content = f"{prompt}\n\n本局起始成语为「{start_word}」。"

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
    system_prompt: str = "",
) -> LLMResponse:
    """调用 LLM 获取结构化输出。异常向上抛出由调用方处理。"""
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    messages = build_messages(history_words, player, start_word, system_prompt)
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
