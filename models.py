from datetime import datetime

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """LLM 结构化输出 schema，传给 OpenAI SDK response_format。"""

    word: str
    success: bool


class ModelConfig(BaseModel):
    """单个 LLM 配置。"""

    base_url: str
    api_key: str
    model: str


class BattleRequest(BaseModel):
    """POST /battle 请求体。"""

    model_a: ModelConfig
    model_b: ModelConfig
    start_word: str


class RoundEvent(BaseModel):
    """SSE round 事件数据。"""

    round: int
    player: str  # "A" or "B"
    model: str
    word: str
    success: bool
    valid: bool
    message: str = ""


class ResultEvent(BaseModel):
    """SSE result 事件数据。"""

    winner: str  # "A", "B", or "draw"
    reason: str
    rounds: int
    history: list[str]
    battle_id: int


class BattleListItem(BaseModel):
    """对战列表项。"""

    id: int
    model_a_name: str
    model_b_name: str
    start_word: str
    winner: str
    reason: str
    created_at: datetime


class BattleDetail(BattleListItem):
    """对战详情（含 history）。"""

    history: list[RoundEvent]
