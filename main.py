import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from tortoise.contrib.fastapi import RegisterTortoise

from battle import run_battle
from benchmark import run_benchmark
from llm import DEFAULT_SYSTEM_PROMPT
from db_models import Battle
from models import BattleDetail, BattleListItem, BattleRequest, BenchmarkRequest, RoundEvent
from validator import is_valid_start_word

DB_PATH = os.environ.get("DB_PATH", "battles.db")

TORTOISE_ORM = {
    "connections": {"default": f"sqlite://{DB_PATH}"},
    "apps": {
        "models": {
            "models": ["db_models"],
            "default_connection": "default",
        }
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with RegisterTortoise(
        app,
        config=TORTOISE_ORM,
        generate_schemas=True,
    ):
        yield


app = FastAPI(title="成语接龙 LLM 对战平台", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/default-system-prompt")
async def get_default_system_prompt():
    return {"system_prompt": DEFAULT_SYSTEM_PROMPT}


@app.post("/battle")
async def battle(request: BattleRequest):
    if not is_valid_start_word(request.start_word):
        raise HTTPException(status_code=400, detail="起始成语不在词库中")
    return EventSourceResponse(run_battle(request))


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    return EventSourceResponse(run_benchmark(request))


@app.get("/battles", response_model=list[BattleListItem])
async def list_battles():
    battles = await Battle.all().order_by("-created_at")
    return [
        BattleListItem(
            id=b.id,
            model_a_name=b.model_a_name,
            model_b_name=b.model_b_name,
            start_word=b.start_word,
            winner=b.winner,
            reason=b.reason,
            created_at=b.created_at,
        )
        for b in battles
    ]


@app.get("/battles/{battle_id}", response_model=BattleDetail)
async def get_battle(battle_id: int):
    b = await Battle.get_or_none(id=battle_id)
    if b is None:
        raise HTTPException(status_code=404, detail="对战记录不存在")
    return BattleDetail(
        id=b.id,
        model_a_name=b.model_a_name,
        model_b_name=b.model_b_name,
        start_word=b.start_word,
        winner=b.winner,
        reason=b.reason,
        created_at=b.created_at,
        history=[RoundEvent(**r) for r in b.history],
    )
