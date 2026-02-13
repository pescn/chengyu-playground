"""成语接龙 GRPO 单步奖励函数核心模块。

奖励公式（每步）:
  total = round_penalty + compliance + strategy + foresight

各组件：
  round_penalty : 每轮固定惩罚（默认 -0.1），激励尽快终结对局
  compliance    : 合规性判定（-1.0 ~ +0.3）
                  - JSON 解析失败 → -1.0
                  - 认输 (success=false) → fail_penalty（默认 -1.0）
                  - 成语不存在 → -0.8
                  - 接龙不匹配 → -0.8
                  - 成语重复   → -0.6
                  - 全部通过   → +0.3
  strategy      : 留给对手的可选成语数量越少越好（0 ~ +0.5）
  foresight     : next_word 合法得 +0.3

依赖文件（运行时需在同级目录）：
  - validator.py   : validate_existence / validate_chain / validate_uniqueness / validate_idiom
  - idiom.json     : 成语词库（含拼音）
  - chengyu.json   : 成语列表
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from validator import (
    validate_chain,
    validate_existence,
    validate_idiom,
    validate_uniqueness,
)

# ---------------------------------------------------------------------------
# 预计算：接龙候选索引（按首字/首拼音/首字+声调分组）
# ---------------------------------------------------------------------------
_IDIOM_PATH = Path(__file__).parent / "idiom.json"
_idiom_list: list[dict] = json.loads(_IDIOM_PATH.read_text(encoding="utf-8"))

# word → (first_no_tone, last_no_tone)
_pinyin_map: dict[str, tuple[str, str]] = {}
# word → (first_tone, last_tone)
_pinyin_tone_map: dict[str, tuple[str, str]] = {}

# 首字 → 成语集合（same_char 模式用）
_by_first_char: dict[str, set[str]] = {}
# 去声调首拼音 → 成语集合（homophone 模式用）
_by_first_pinyin: dict[str, set[str]] = {}
# (首字, 带声调首拼音) → 成语集合（same_char_sound 模式用）
_by_first_char_tone: dict[tuple[str, str], set[str]] = {}

for _entry in _idiom_list:
    _w = _entry["word"]
    _first_nt = _entry["first"]
    _last_nt = _entry["last"]
    _pinyin_map[_w] = (_first_nt, _last_nt)

    _tokens = _entry["pinyin"].split()
    if _tokens:
        _pinyin_tone_map[_w] = (_tokens[0], _tokens[-1])

    _fc = _w[0]
    _by_first_char.setdefault(_fc, set()).add(_w)
    _by_first_pinyin.setdefault(_first_nt, set()).add(_w)
    if _tokens:
        _by_first_char_tone.setdefault((_fc, _tokens[0]), set()).add(_w)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def count_continuations(word: str, used_words: set[str], mode: str = "same_char") -> int:
    """统计 ``word`` 之后对手的可选成语数量（排除已用过的）。"""
    last_char = word[-1]

    if mode == "same_char":
        candidates = _by_first_char.get(last_char, set())
    elif mode == "homophone":
        last_pinyin = _pinyin_map.get(word, ("", ""))[1]
        candidates = _by_first_pinyin.get(last_pinyin, set()) if last_pinyin else set()
    elif mode == "same_char_sound":
        last_tone = _pinyin_tone_map.get(word, ("", ""))[1]
        candidates = _by_first_char_tone.get((last_char, last_tone), set()) if last_tone else set()
    else:
        candidates = _by_first_char.get(last_char, set())

    return len(candidates - used_words)


def parse_llm_response(text: str) -> tuple[bool, dict]:
    """解析 LLM 的 JSON 输出。

    尝试直接解析；失败时尝试从文本中提取第一个 ``{...}`` 块。
    返回 ``(解析成功, 解析结果 dict)``。
    """
    text = text.strip()

    def _try_parse(s: str) -> dict | None:
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return None

    data = _try_parse(text)
    if data is None:
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            data = _try_parse(match.group())

    if data is None:
        return False, {}

    return True, {
        "word": str(data.get("word", "")),
        "next_word": str(data.get("next_word", "")),
        "success": bool(data.get("success", False)),
    }


# ---------------------------------------------------------------------------
# 核心：单步奖励计算
# ---------------------------------------------------------------------------

def compute_step_reward(
    response_text: str,
    previous_word: str,
    used_words: set[str],
    round_num: int,
    mode: str = "same_char",
    *,
    round_penalty: float = -0.1,
    fail_penalty: float = -1.0,
) -> tuple[float, dict]:
    """计算单步奖励。

    Parameters
    ----------
    response_text : str
        模型生成的原始文本（应为 JSON）。
    previous_word : str
        上一个成语（模型需要接龙的目标）。
    used_words : set[str]
        本局已经使用过的成语集合（含起始成语和所有历史成语）。
    round_num : int
        当前回合序号（从 1 开始）。
    mode : str
        验证模式：same_char / homophone / same_char_sound。
    round_penalty : float
        每轮固定惩罚，默认 -0.1。
    fail_penalty : float
        认输 / 无法继续的惩罚，默认 -1.0。

    Returns
    -------
    (total_reward, info)
        info 包含各组件明细，便于日志分析。
    """
    info: dict = {
        "round_num": round_num,
        "round_penalty": round_penalty,
        "compliance": 0.0,
        "strategy": 0.0,
        "foresight": 0.0,
        "total": 0.0,
        "valid": False,
        "reason": "",
        "word": "",
        "continuation_count": -1,
    }

    total = round_penalty

    # ---- 1. JSON 解析 ----
    parse_ok, parsed = parse_llm_response(response_text)
    if not parse_ok:
        info["compliance"] = -1.0
        info["reason"] = "JSON解析失败"
        info["total"] = total + (-1.0)
        return info["total"], info

    info["word"] = parsed["word"]

    # ---- 2. 认输 ----
    if not parsed["success"]:
        info["compliance"] = fail_penalty
        info["reason"] = "认输"
        info["total"] = total + fail_penalty
        return info["total"], info

    word = parsed["word"]

    # ---- 3. 存在性 ----
    if not validate_existence(word):
        info["compliance"] = -0.8
        info["reason"] = "成语不在词库中"
        info["total"] = total + (-0.8)
        return info["total"], info

    # ---- 4. 接龙匹配 ----
    chain_valid, chain_msg = validate_chain(word, previous_word, mode)
    if not chain_valid:
        info["compliance"] = -0.8
        info["reason"] = chain_msg
        info["total"] = total + (-0.8)
        return info["total"], info

    # ---- 5. 唯一性 ----
    if not validate_uniqueness(word, used_words):
        info["compliance"] = -0.6
        info["reason"] = "成语已使用过"
        info["total"] = total + (-0.6)
        return info["total"], info

    # ---- 合法走子 ----
    info["valid"] = True
    info["compliance"] = 0.3
    total += 0.3

    # ---- 6. 策略：对手可接成语数 ----
    updated_used = used_words | {word}
    cont = count_continuations(word, updated_used, mode)
    info["continuation_count"] = cont

    if cont == 0:
        strategy = 0.5
    elif cont <= 5:
        strategy = 0.3
    elif cont <= 20:
        strategy = 0.1
    else:
        strategy = 0.0

    info["strategy"] = strategy
    total += strategy

    # ---- 7. 远见：next_word 合法性 ----
    next_word = parsed.get("next_word", "")
    if next_word:
        nw_valid, _ = validate_idiom(next_word, word, updated_used, mode)
        if nw_valid:
            info["foresight"] = 0.3
            total += 0.3

    info["total"] = total
    return total, info
