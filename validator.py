import json
from pathlib import Path

_CHENGYU_PATH = Path(__file__).parent / "chengyu.json"
_chengyu_set: set[str] = set(json.loads(_CHENGYU_PATH.read_text(encoding="utf-8")))

# 加载 idiom.json，构建拼音查找字典
_IDIOM_PATH = Path(__file__).parent / "idiom.json"
_idiom_list: list[dict] = json.loads(_IDIOM_PATH.read_text(encoding="utf-8"))

# word -> (first_no_tone, last_no_tone)  来自 first/last 字段
_pinyin_map: dict[str, tuple[str, str]] = {}
# word -> (first_tone, last_tone)  从 pinyin 字段拆分首尾 token
_pinyin_tone_map: dict[str, tuple[str, str]] = {}

for _entry in _idiom_list:
    _word = _entry["word"]
    _pinyin_map[_word] = (_entry["first"], _entry["last"])
    _tokens = _entry["pinyin"].split()
    if _tokens:
        _pinyin_tone_map[_word] = (_tokens[0], _tokens[-1])


def get_chengyu_list() -> list[str]:
    """返回词库中所有成语列表，供随机采样使用。"""
    return list(_chengyu_set)


def validate_existence(word: str) -> bool:
    """检查成语是否在词库中。"""
    return word in _chengyu_set


def validate_chain(word: str, previous_word: str, mode: str = "same_char") -> tuple[bool, str]:
    """检查接龙条件，根据 mode 选择验证逻辑。

    返回 (是否通过, 错误信息)。
    """
    if mode == "same_char":
        if word[0] != previous_word[-1]:
            return False, "首字不匹配"
        return True, ""

    elif mode == "homophone":
        # 同音即可：去声调拼音比较
        prev_last = _pinyin_map.get(previous_word, ("", ""))[1]
        curr_first = _pinyin_map.get(word, ("", ""))[0]
        if not prev_last or not curr_first:
            # 查不到拼音时回退到同字比较
            if word[0] != previous_word[-1]:
                return False, "首字读音不匹配"
            return True, ""
        if prev_last != curr_first:
            return False, "首字读音不匹配"
        return True, ""

    elif mode == "same_char_sound":
        # 同字同音：首字相同 + 带声调拼音相同
        if word[0] != previous_word[-1]:
            return False, "首字不匹配"
        prev_last_tone = _pinyin_tone_map.get(previous_word, ("", ""))[1]
        curr_first_tone = _pinyin_tone_map.get(word, ("", ""))[0]
        if not prev_last_tone or not curr_first_tone:
            return True, ""
        if prev_last_tone != curr_first_tone:
            return False, "首字读音不匹配（多音字声调不同）"
        return True, ""

    # 未知 mode 回退到 same_char
    if word[0] != previous_word[-1]:
        return False, "首字不匹配"
    return True, ""


def validate_uniqueness(word: str, used_words: set[str]) -> bool:
    """检查本局未使用过。"""
    return word not in used_words


def validate_idiom(
    word: str, previous_word: str, used_words: set[str], mode: str = "same_char"
) -> tuple[bool, str]:
    """综合验证，返回 (是否有效, 错误原因)。"""
    if not validate_existence(word):
        return False, "成语不在词库中"
    valid, message = validate_chain(word, previous_word, mode)
    if not valid:
        return False, message
    if not validate_uniqueness(word, used_words):
        return False, "成语已使用过"
    return True, ""


def is_valid_start_word(word: str) -> bool:
    """检查起始成语是否在词库中。"""
    return word in _chengyu_set
