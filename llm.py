from openai import AsyncOpenAI

from models import LLMResponse, ModelConfig

SYSTEM_PROMPT = """\
你是一个成语接龙高手，正在与对手进行成语接龙对战。你的目标是击败对手，让对手无法继续接龙。

成语接龙规则：
- 双方轮流说一个成语，新成语的首字必须与上一个成语的尾字相同（同字，非同音）
- 成语必须真实存在，不可生造，不可重复使用已出现过的成语
- 无法接龙者判负

对战策略（非常重要）：
- 优先选择末字生僻、不常见的成语，让对手难以找到以该字开头的成语继续接龙
- 避免使用末字为常见字（如"人""大""天""心""不"等）的成语，因为这些字开头的成语太多，对手很容易接上
- 尽量把对手逼入"死胡同"——选择那些末字几乎没有成语可以接续的成语

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
