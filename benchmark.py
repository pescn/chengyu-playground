import asyncio
import random
from collections.abc import AsyncGenerator

from battle import execute_battle
from models import (
    BenchmarkRequest,
    BenchmarkBattleResult,
    BenchmarkProgressEvent,
    BenchmarkSummaryEvent,
    ModelConfig,
)
from validator import get_chengyu_list


def _select_random_words(n: int) -> list[str]:
    """从词库中随机抽取 n 个不重复的成语。"""
    pool = get_chengyu_list()
    return random.sample(pool, min(n, len(pool)))


async def _run_single_benchmark_battle(
    model_a: ModelConfig,
    model_b: ModelConfig,
    start_word: str,
    first_player: str,
    semaphore: asyncio.Semaphore,
) -> BenchmarkBattleResult:
    """运行单场 benchmark 对战。

    first_player="A" 时正常调用；
    first_player="B" 时交换 model_a/model_b 参数，结果中的 winner 反转映射。
    """
    async with semaphore:
        if first_player == "A":
            result = await execute_battle(model_a, model_b, start_word)
            winner = result.winner
        else:
            # B 先手：交换模型位置
            result = await execute_battle(model_b, model_a, start_word)
            # 反转 winner 映射（execute_battle 中 A 位置实际是 model_b）
            if result.winner == "A":
                winner = "B"
            elif result.winner == "B":
                winner = "A"
            else:
                winner = "draw"

        return BenchmarkBattleResult(
            start_word=start_word,
            first_player=first_player,
            winner=winner,
            reason=result.reason,
            rounds=result.rounds,
            battle_id=result.battle_id,
        )


def _sse_event(event: str, data: str) -> dict[str, str]:
    return {"event": event, "data": data}


async def run_benchmark(
    request: BenchmarkRequest,
) -> AsyncGenerator[dict[str, str], None]:
    """批量测试主流程，yield SSE 事件（progress + summary）。"""
    words = _select_random_words(request.num_words)
    semaphore = asyncio.Semaphore(request.max_concurrency)

    # 每个成语跑两场：A先手 + B先手
    tasks: list[asyncio.Task] = []
    for word in words:
        for first in ("A", "B"):
            task = asyncio.create_task(
                _run_single_benchmark_battle(
                    request.model_a, request.model_b, word, first, semaphore
                )
            )
            tasks.append(task)

    total = len(tasks)
    completed = 0
    results: list[BenchmarkBattleResult] = []

    for coro in asyncio.as_completed(tasks):
        battle_result = await coro
        completed += 1
        results.append(battle_result)

        progress = BenchmarkProgressEvent(
            completed=completed,
            total=total,
            current_result=battle_result,
        )
        yield _sse_event("progress", progress.model_dump_json())

    # 计算统计
    model_a_wins = sum(1 for r in results if r.winner == "A")
    model_b_wins = sum(1 for r in results if r.winner == "B")
    draws = sum(1 for r in results if r.winner == "draw")

    # 先手分项：A先手的场次
    a_first = [r for r in results if r.first_player == "A"]
    model_a_first_wins = sum(1 for r in a_first if r.winner == "A")
    model_a_first_losses = sum(1 for r in a_first if r.winner == "B")
    model_a_first_draws = sum(1 for r in a_first if r.winner == "draw")

    # 先手分项：B先手的场次
    b_first = [r for r in results if r.first_player == "B"]
    model_b_first_wins = sum(1 for r in b_first if r.winner == "B")
    model_b_first_losses = sum(1 for r in b_first if r.winner == "A")
    model_b_first_draws = sum(1 for r in b_first if r.winner == "draw")

    summary = BenchmarkSummaryEvent(
        total_battles=total,
        model_a_name=request.model_a.model,
        model_b_name=request.model_b.model,
        model_a_wins=model_a_wins,
        model_b_wins=model_b_wins,
        draws=draws,
        model_a_win_rate=model_a_wins / total if total > 0 else 0,
        model_b_win_rate=model_b_wins / total if total > 0 else 0,
        model_a_first_wins=model_a_first_wins,
        model_a_first_losses=model_a_first_losses,
        model_a_first_draws=model_a_first_draws,
        model_b_first_wins=model_b_first_wins,
        model_b_first_losses=model_b_first_losses,
        model_b_first_draws=model_b_first_draws,
        battles=sorted(results, key=lambda r: (r.start_word, r.first_player)),
    )
    yield _sse_event("summary", summary.model_dump_json())
