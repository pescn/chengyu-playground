import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from db_models import Battle
from llm import call_llm
from models import BattleRequest, RoundEvent, ResultEvent, ModelConfig
from validator import validate_idiom

MAX_ROUNDS = 30


@dataclass
class BattleResult:
    winner: str  # "A", "B", "draw"
    reason: str
    rounds: int
    history: list[str]  # 完整历史（含起始成语）
    battle_id: int


def _player_label(player: str) -> str:
    return "模型A" if player == "A" else "模型B"


def _opponent(player: str) -> str:
    return "B" if player == "A" else "A"


def _sse_event(event: str, data: str) -> dict[str, str]:
    return {"event": event, "data": data}


async def execute_battle(
    model_a: ModelConfig,
    model_b: ModelConfig,
    start_word: str,
    on_round: Callable[[RoundEvent], Awaitable[None]] | None = None,
) -> BattleResult:
    """执行完整对战循环，返回 BattleResult。

    可选 on_round 回调在每回合结束后触发，供 SSE 包装器使用。
    """
    configs = {"A": model_a, "B": model_b}

    history_words: list[str] = []
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
            if on_round:
                await on_round(round_event)
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
            if on_round:
                await on_round(round_event)
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
        if on_round:
            await on_round(round_event)

        if not valid:
            winner = _opponent(current_player)
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
        winner = "draw"
        reason = "达到最大回合数"

    # 保存到数据库
    battle = await Battle.create(
        model_a_name=model_a.model,
        model_b_name=model_b.model,
        start_word=start_word,
        history=history_records,
        winner=winner,
        reason=reason,
    )

    full_history = [start_word] + history_words

    return BattleResult(
        winner=winner,
        reason=reason,
        rounds=round_num,
        history=full_history,
        battle_id=battle.id,
    )


async def run_battle(request: BattleRequest) -> AsyncGenerator[dict[str, str], None]:
    """SSE 包装器：通过 Queue + on_round 回调将 execute_battle 的回合事件流式输出。"""
    queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()

    async def on_round(event: RoundEvent) -> None:
        await queue.put(_sse_event("round", event.model_dump_json()))

    async def run() -> BattleResult:
        result = await execute_battle(
            request.model_a, request.model_b, request.start_word, on_round=on_round
        )
        return result

    task = asyncio.create_task(run())

    # 持续从 queue 读取回合事件并 yield
    while not task.done() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.1)
            if event is not None:
                yield event
        except asyncio.TimeoutError:
            continue

    # task 完成后，排空 queue
    while not queue.empty():
        event = queue.get_nowait()
        if event is not None:
            yield event

    # yield 最终结果
    battle_result = task.result()
    result_event = ResultEvent(
        winner=battle_result.winner,
        reason=battle_result.reason,
        rounds=battle_result.rounds,
        history=battle_result.history,
        battle_id=battle_result.battle_id,
    )
    yield _sse_event("result", result_event.model_dump_json())
