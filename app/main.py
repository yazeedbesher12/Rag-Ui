from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .rag import init_rag, run_query, run_query_stream


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the CV collection")
    include_context: bool = Field(False, description="Return retrieved CV chunks with the answer")


class AskResponse(BaseModel):
    question: str
    answer: str
    context: Optional[list] = None
    formatted_context: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_rag()
        app.state.rag_ready = True
        app.state.rag_error = None
    except Exception as exc:  # Keep API alive so /health shows the error.
        app.state.rag_ready = False
        app.state.rag_error = str(exc)
    yield


app = FastAPI(
    title="CV RAG API",
    description="Ask questions about CVs indexed in Pinecone using a RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "CV RAG API",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask",
        "ask_stream": "POST /ask-stream",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if getattr(app.state, "rag_ready", False) else "error",
        "rag_ready": getattr(app.state, "rag_ready", False),
        "error": getattr(app.state, "rag_error", None),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        return run_query(payload.question, include_context=payload.include_context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-stream")
async def ask_stream(payload: AskRequest):
    async def generator() -> AsyncGenerator[bytes, None]:
        try:
            async for token in run_query_stream(payload.question):
                yield token.encode("utf-8")
        except Exception as exc:
            yield f"\n[ERROR] {exc}".encode("utf-8")

    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")
