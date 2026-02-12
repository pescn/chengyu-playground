import json
from pathlib import Path

_CHENGYU_PATH = Path(__file__).parent / "chengyu.json"
_chengyu_set: set[str] = set(json.loads(_CHENGYU_PATH.read_text(encoding="utf-8")))


def validate_existence(word: str) -> bool:
    """检查成语是否在词库中。"""
    return word in _chengyu_set


def validate_chain(word: str, previous_word: str) -> bool:
    """检查首字 == 上一个成语末字（同字，非同音）。"""
    return word[0] == previous_word[-1]


def validate_uniqueness(word: str, used_words: set[str]) -> bool:
    """检查本局未使用过。"""
    return word not in used_words


def validate_idiom(
    word: str, previous_word: str, used_words: set[str]
) -> tuple[bool, str]:
    """综合验证，返回 (是否有效, 错误原因)。"""
    if not validate_existence(word):
        return False, "成语不在词库中"
    if not validate_chain(word, previous_word):
        return False, "首字不匹配"
    if not validate_uniqueness(word, used_words):
        return False, "成语已使用过"
    return True, ""


def is_valid_start_word(word: str) -> bool:
    """检查起始成语是否在词库中。"""
    return word in _chengyu_set
