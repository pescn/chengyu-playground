import logging
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

from db_models import Battle
from llm import call_llm
from models import BattleRequest, RoundEvent, ResultEvent
from validator import validate_idiom

MAX_ROUNDS = 30


def _player_label(player: str) -> str:
    return "模型A" if player == "A" else "模型B"


def _opponent(player: str) -> str:
    return "B" if player == "A" else "A"


def _sse_event(event: str, data: str) -> dict[str, str]:
    return {"event": event, "data": data}


async def run_battle(request: BattleRequest) -> AsyncGenerator[dict[str, str], None]:
    configs = {"A": request.model_a, "B": request.model_b}
    start_word = request.start_word

    history_words: list[str] = []  # 模型输出的成语序列（不含起始成语）
    used_words: set[str] = {start_word}
    history_records: list[dict] = []

    current_player = "A"
    winner: str | None = None
    reason = ""
    round_num = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        config = configs[current_player]
        model_name = config.model
        label = _player_label(current_player)

        # 调用 LLM
        try:
            response = await call_llm(
                config, history_words, current_player, start_word
            )
        except Exception as e:
            logger.exception(f"{label} LLM 调用异常: {e}")
            winner = _opponent(current_player)
            reason = f"{label}调用失败"
            round_event = RoundEvent(
                round=round_num,
                player=current_player,
                model=model_name,
                word="",
                success=False,
                valid=False,
                message="LLM 调用失败",
            )
            history_records.append(round_event.model_dump())
            yield _sse_event("round", round_event.model_dump_json())
            break

        # 检查认输
        if not response.success:
            winner = _opponent(current_player)
            reason = f"{label}认输"
            round_event = RoundEvent(
                round=round_num,
                player=current_player,
                model=model_name,
                word=response.word,
                success=False,
                valid=True,
                message="",
            )
            history_records.append(round_event.model_dump())
            yield _sse_event("round", round_event.model_dump_json())
            break

        # 验证成语
        previous_word = history_words[-1] if history_words else start_word
        valid, message = validate_idiom(response.word, previous_word, used_words)

        round_event = RoundEvent(
            round=round_num,
            player=current_player,
            model=model_name,
            word=response.word,
            success=True,
            valid=valid,
            message=message,
        )
        history_records.append(round_event.model_dump())
        yield _sse_event("round", round_event.model_dump_json())

        if not valid:
            winner = _opponent(current_player)
            # 映射验证错误到具体 reason
            if message == "成语不在词库中":
                reason = f"{label}成语不在词库中"
            elif message == "首字不匹配":
                reason = f"{label}首字不匹配"
            elif message == "成语已使用过":
                reason = f"{label}成语重复使用"
            break

        # 有效，记录并切换玩家
        history_words.append(response.word)
        used_words.add(response.word)
        current_player = _opponent(current_player)
    else:
        # 达到最大回合数
        winner = "draw"
        reason = "达到最大回合数"

    # 保存到数据库
    battle = await Battle.create(
        model_a_name=request.model_a.model,
        model_b_name=request.model_b.model,
        start_word=start_word,
        history=history_records,
        winner=winner,
        reason=reason,
    )

    # 构建完整历史（含起始成语）
    full_history = [start_word] + history_words

    result = ResultEvent(
        winner=winner,
        reason=reason,
        rounds=round_num,
        history=full_history,
        battle_id=battle.id,
    )
    yield _sse_event("result", result.model_dump_json())
