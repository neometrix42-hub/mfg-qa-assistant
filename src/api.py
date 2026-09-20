"""FastAPI wrapper around the agent.

Small on purpose. The interesting code is in agent.py and evals/.
Week 7 - do not build this before the evals exist.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent import ask
from src.db import healthcheck

app = FastAPI(title="Manufacturing QA Assistant", version="0.1.0")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    question: str
    answer: str
    tools_called: list[str]
    latency_ms: int


@app.get("/health")
def health() -> dict:
    return {"ok": healthcheck()}


@app.post("/ask", response_model=Answer)
def post_ask(payload: Question) -> Answer:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    result = ask(payload.question)
    return Answer(**{k: result[k] for k in ("question", "answer", "tools_called", "latency_ms")})


# Run: uvicorn src.api:app --reload
