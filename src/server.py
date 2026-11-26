"""API 스펙(3-2): POST /query  {"question": ...} -> {"answer", "contexts", "trace"}

실행: uvicorn src.server:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .agent import run_agent

app = FastAPI(title="TechScout", description="실무형 최신 기술 리서치 어시스턴트")


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Context(BaseModel):
    doc_id: str
    url: str
    text: str


class TraceStep(BaseModel):
    step: str
    input: str
    output: str


class QueryOut(BaseModel):
    answer: str
    contexts: list[Context]
    trace: list[TraceStep]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryOut)
async def query(body: QueryIn):
    try:
        return await run_in_threadpool(run_agent, body.question)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
