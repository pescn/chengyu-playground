"""生成 veRL GRPO 训练用 Parquet 数据集。

通过两个模型实时对战收集训练样本：每步在调用 LLM 之前，
将当前游戏状态构建为一条训练数据（prompt），由 veRL 训练时
让 policy 自行生成回复并用 reward_verl.py 打分。

用法示例：
    python gen_dataset.py \
        --base-url http://172.16.56.3:2998/v1 \
        --api-key sk-xxx \
        --model-a qwen3-max \
        --model-b doubao-seed-1-6 \
        --num-games 100 \
        --output dataset.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys

from models import ModelConfig
from llm import build_messages, call_llm, DEFAULT_SYSTEM_PROMPT
from validator import validate_idiom, get_chengyu_list

logger = logging.getLogger(__name__)

MAX_ROUNDS = 30


async def play_game(
    config_a: ModelConfig,
    config_b: ModelConfig,
    start_word: str,
    system_prompt: str,
    validation_mode: str,
) -> list[dict]:
    """执行一局对战，收集训练样本。

    不写数据库，不产生 SSE 事件。
    每步在调用 LLM **之前**收集当前玩家视角的训练数据。
    """
    configs = {"A": config_a, "B": config_b}
    history_words: list[str] = []
    used_words: set[str] = {start_word}
    samples: list[dict] = []

    current_player = "A"

    for round_num in range(1, MAX_ROUNDS + 1):
        config = configs[current_player]
        previous_word = history_words[-1] if history_words else start_word

        # --- 收集训练样本（调用 LLM 之前）---
        messages = build_messages(
            history_words, current_player, start_word,
            system_prompt, validation_mode,
        )
        samples.append({
            "data_source": "chengyu",
            "prompt": json.dumps(messages, ensure_ascii=False),
            "reward_model": json.dumps(
                {"style": "rule", "ground_truth": previous_word},
                ensure_ascii=False,
            ),
            "extra_info": json.dumps(
                {
                    "previous_word": previous_word,
                    "used_words": list(used_words),
                    "round_num": round_num,
                    "validation_mode": validation_mode,
                },
                ensure_ascii=False,
            ),
        })

        # --- 调用 LLM ---
        try:
            response = await call_llm(
                config, history_words, current_player, start_word,
                system_prompt, validation_mode,
            )
        except Exception as e:
            logger.warning("Game %s round %d: %s LLM 调用失败: %s",
                           start_word, round_num, current_player, e)
            break

        if not response.success:
            break

        valid, _ = validate_idiom(response.word, previous_word, used_words, validation_mode)
        if not valid:
            break

        history_words.append(response.word)
        used_words.add(response.word)
        current_player = "B" if current_player == "A" else "A"

    return samples


async def main() -> None:
    parser = argparse.ArgumentParser(description="生成 veRL GRPO 训练用 Parquet 数据集")

    # 通用 API 配置
    parser.add_argument("--base-url", default=None, help="LLM API 地址（双方共用）")
    parser.add_argument("--api-key", default=None, help="API Key（双方共用）")

    # 模型 A 独立配置
    parser.add_argument("--base-url-a", default=None, help="模型 A 的 API 地址")
    parser.add_argument("--api-key-a", default=None, help="模型 A 的 API Key")
    parser.add_argument("--model-a", required=True, help="模型 A 名称")

    # 模型 B 独立配置
    parser.add_argument("--base-url-b", default=None, help="模型 B 的 API 地址")
    parser.add_argument("--api-key-b", default=None, help="模型 B 的 API Key")
    parser.add_argument("--model-b", required=True, help="模型 B 名称")

    # 对战参数
    parser.add_argument("--num-games", type=int, default=100, help="对局数量")
    parser.add_argument("--max-concurrency", type=int, default=5, help="最大并发对局数")
    parser.add_argument("--validation-mode", default="same_char",
                        choices=["same_char", "homophone", "same_char_sound"],
                        help="验证模式")
    parser.add_argument("--system-prompt", default="", help="自定义系统提示词（空则用默认）")
    parser.add_argument("--output", default="dataset.parquet", help="输出路径")

    args = parser.parse_args()

    # 解析 API 配置，独立配置优先，否则用共用配置
    base_url_a = args.base_url_a or args.base_url
    api_key_a = args.api_key_a or args.api_key
    base_url_b = args.base_url_b or args.base_url
    api_key_b = args.api_key_b or args.api_key

    if not base_url_a or not api_key_a:
        parser.error("模型 A 需要 --base-url-a/--api-key-a 或 --base-url/--api-key")
    if not base_url_b or not api_key_b:
        parser.error("模型 B 需要 --base-url-b/--api-key-b 或 --base-url/--api-key")

    config_a = ModelConfig(base_url=base_url_a, api_key=api_key_a, model=args.model_a)
    config_b = ModelConfig(base_url=base_url_b, api_key=api_key_b, model=args.model_b)

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    # 随机采样起始成语
    all_idioms = get_chengyu_list()
    start_words = random.sample(all_idioms, min(args.num_games, len(all_idioms)))

    semaphore = asyncio.Semaphore(args.max_concurrency)
    all_samples: list[dict] = []
    completed = 0

    async def run_one(start_word: str) -> list[dict]:
        nonlocal completed
        async with semaphore:
            samples = await play_game(
                config_a, config_b, start_word,
                system_prompt, args.validation_mode,
            )
            completed += 1
            print(f"\r进度: {completed}/{args.num_games} 局"
                  f"  样本数: {sum(len(s) for s in all_samples) + len(samples)}",
                  end="", flush=True)
            return samples

    tasks = [run_one(w) for w in start_words]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            all_samples.extend(result)
        else:
            logger.warning("对局异常: %s", result)

    print(f"\n总样本数: {len(all_samples)}")

    if not all_samples:
        print("未收集到任何样本，跳过写入。", file=sys.stderr)
        sys.exit(1)

    # 写 Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "data_source": [s["data_source"] for s in all_samples],
        "prompt": [s["prompt"] for s in all_samples],
        "reward_model": [s["reward_model"] for s in all_samples],
        "extra_info": [s["extra_info"] for s in all_samples],
    })
    pq.write_table(table, args.output)
    print(f"数据集已写入: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    asyncio.run(main())
