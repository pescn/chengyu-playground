"""veRL GRPO 奖励函数适配器。

用法：
  python3 -m verl.trainer.main_ppo \\
      algorithm.adv_estimator=grpo \\
      reward.custom_reward_function.path=/path/to/reward_verl.py \\
      reward.custom_reward_function.name=compute_score \\
      ...

Parquet 数据集格式：
  {
      "data_source": "chengyu",
      "prompt": [
          {"role": "system", "content": "你是成语接龙玩家..."},
          {"role": "user", "content": "一心一意"},
          {"role": "assistant", "content": "意气风发"},
          {"role": "user", "content": "发人深省"}
      ],
      "reward_model": {
          "style": "rule",
          "ground_truth": "发人深省"          // previous_word
      },
      "extra_info": {
          "used_words": ["一心一意", "意气风发", "发人深省"],
          "round_num": 3,
          "validation_mode": "same_char"
      }
  }

也支持将完整游戏状态 JSON 编码在 ground_truth 中（自动检测）：
  "ground_truth": "{\"previous_word\":\"发人深省\",\"used_words\":[...],\"round_num\":3,\"validation_mode\":\"same_char\"}"
"""

from __future__ import annotations

import json

from reward import compute_step_reward


def _parse_game_state(ground_truth: str, extra_info: dict | None) -> dict:
    """从 ground_truth + extra_info 解析游戏状态。

    优先使用 extra_info 中的结构化字段；
    如果 extra_info 缺失关键字段，尝试将 ground_truth 作为 JSON 解析。
    最终 fallback：ground_truth 直接作为 previous_word。
    """
    state = {
        "previous_word": "",
        "used_words": set(),
        "round_num": 1,
        "validation_mode": "same_char",
    }

    extra = extra_info or {}

    # 优先从 extra_info 取值
    if "previous_word" in extra:
        state["previous_word"] = str(extra["previous_word"])
        state["used_words"] = set(extra.get("used_words", []))
        state["round_num"] = int(extra.get("round_num", 1))
        state["validation_mode"] = str(extra.get("validation_mode", "same_char"))
        return state

    # 尝试将 ground_truth 作为 JSON 解析
    try:
        obj = json.loads(ground_truth)
        if isinstance(obj, dict) and "previous_word" in obj:
            state["previous_word"] = str(obj["previous_word"])
            state["used_words"] = set(obj.get("used_words", []))
            state["round_num"] = int(obj.get("round_num", 1))
            state["validation_mode"] = str(obj.get("validation_mode", "same_char"))
            return state
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # fallback：ground_truth 就是 previous_word
    state["previous_word"] = ground_truth.strip()
    state["used_words"] = {ground_truth.strip()} if ground_truth.strip() else set()
    state["round_num"] = int(extra.get("round_num", 1))
    state["validation_mode"] = str(extra.get("validation_mode", "same_char"))
    return state


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """veRL 奖励函数入口。

    Parameters
    ----------
    data_source : str
        数据集标识（如 "chengyu"），本函数不区分。
    solution_str : str
        模型生成的回复文本（应为 JSON：{"word":..., "next_word":..., "success":...}）。
    ground_truth : str
        上一个成语（previous_word），或包含完整游戏状态的 JSON。
    extra_info : dict, optional
        额外上下文：used_words, round_num, validation_mode 等。

    Returns
    -------
    dict
        {"score": float, ...} — score 为 veRL 使用的奖励值，
        其余字段作为 reward_extra_info 记录到日志。
    """
    state = _parse_game_state(ground_truth, extra_info)

    reward, info = compute_step_reward(
        response_text=solution_str,
        previous_word=state["previous_word"],
        used_words=state["used_words"],
        round_num=state["round_num"],
        mode=state["validation_mode"],
    )

    return {
        "score": reward,
        "valid": info["valid"],
        "word": info["word"],
        "reason": info["reason"],
        "compliance": info["compliance"],
        "strategy": info["strategy"],
        "foresight": info["foresight"],
        "continuation_count": info["continuation_count"],
        "round_num": info["round_num"],
    }
