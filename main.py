from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from routers import agent, models


@asynccontextmanager
async def lifespan(_: FastAPI):
    import logging
    logger = logging.getLogger("mars-ai-agent")

    from llm.client import is_available as ollama_ok
    from rag.embedder import get_client as chroma_client

    if await ollama_ok():
        logger.info("Ollama 연결 확인")
    else:
        logger.warning("Ollama 연결 실패 — LLM 기능이 동작하지 않을 수 있습니다")

    try:
        chroma_client().list_collections()
        logger.info("ChromaDB 연결 확인")
    except Exception as e:
        logger.warning(f"ChromaDB 연결 실패 — RAG 기능이 동작하지 않을 수 있습니다: {e}")

    yield


app = FastAPI(
    title="MARS AI Agent",
    description="MARS AI 플랫폼 AI Agent 서버 - 임상 해석 및 모델 실행 계획 수립",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(models.router, prefix="/agent/models", tags=["models"])


@app.get("/health")
async def health():
    from llm.client import is_available as ollama_ok
    from rag.embedder import get_client as chroma_client
    import os

    ollama_status = "ok" if await ollama_ok() else "unavailable"

    try:
        col = chroma_client()
        col.list_collections()
        chroma_status = "ok"
    except Exception:
        chroma_status = "unavailable"

    overall = "ok" if ollama_status == "ok" and chroma_status == "ok" else "degraded"

    return {
        "status": overall,
        "service": "mars-ai-agent",
        "ollama": ollama_status,
        "chromadb": chroma_status,
        "model": os.getenv("LLM_MODEL", "gemma4:31b"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
