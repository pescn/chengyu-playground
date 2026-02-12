from datetime import datetime

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """LLM 结构化输出 schema，传给 OpenAI SDK response_format。"""

    word: str
    next_word: str = ""
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
    next_word: str = ""
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


# ========== Benchmark 模型 ==========


class BenchmarkRequest(BaseModel):
    """POST /benchmark 请求体。"""

    model_a: ModelConfig
    model_b: ModelConfig
    num_words: int = Field(default=5, ge=1, le=50)
    max_concurrency: int = Field(default=3, ge=1, le=10)


class BenchmarkBattleResult(BaseModel):
    """单场 benchmark 对战结果。"""

    start_word: str
    first_player: str  # "A" or "B" — 谁先手
    winner: str  # "A", "B", "draw"
    reason: str
    rounds: int
    battle_id: int


class BenchmarkProgressEvent(BaseModel):
    """SSE progress 事件数据。"""

    completed: int
    total: int
    current_result: BenchmarkBattleResult


class BenchmarkSummaryEvent(BaseModel):
    """SSE summary 事件数据。"""

    total_battles: int
    model_a_name: str
    model_b_name: str
    model_a_wins: int
    model_b_wins: int
    draws: int
    model_a_win_rate: float
    model_b_win_rate: float
    # 先手分项统计
    model_a_first_wins: int  # A先手时A赢的场数
    model_a_first_losses: int  # A先手时A输的场数
    model_a_first_draws: int
    model_b_first_wins: int  # B先手时B赢的场数
    model_b_first_losses: int  # B先手时B输的场数
    model_b_first_draws: int
    battles: list[BenchmarkBattleResult]
