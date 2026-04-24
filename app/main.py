"""
NeuroAgent — FastAPI Application Entry Point
Async gateway with lifespan management, middleware, and Lambda compatibility.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.research import router as research_router
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("app.main")
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: warm up DB pool and pre-import heavy modules.
    Shutdown: gracefully dispose DB connections.
    """
    logger.info(
        "NeuroAgent starting",
        environment=settings.environment,
        mock_mode=settings.mock_llm,
    )

    # Pre-import agent modules to reduce first-request latency
    # (critical for Lambda cold-start < 800ms target)
    import app.agent.orchestrator  # noqa: F401
    import app.rag.pipeline        # noqa: F401

    if not settings.mock_llm:
        # Only init DB pool when not in mock mode
        try:
            from app.db.session import get_engine
            engine = get_engine()
            async with engine.connect():
                pass
            logger.info("Database connection pool initialized")
        except Exception as exc:
            logger.warning("Database unavailable at startup", error=str(exc))

    logger.info("NeuroAgent ready to serve requests")
    yield

    # Shutdown
    if not settings.mock_llm:
        try:
            from app.db.session import dispose_engine
            await dispose_engine()
        except Exception:
            pass

    logger.info("NeuroAgent shutdown complete")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="NeuroAgent",
        description=(
            "Autonomous AI Research Assistant — multi-step agent with RAG, "
            "tool-calling, retry logic, and self-healing error recovery.\n\n"
            "**Tech stack:** Python · LangChain · OpenAI · FastAPI · PostgreSQL/pgvector · Docker · AWS Lambda"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.monotonic()

        response = await call_next(request)

        latency_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(round(latency_ms, 1))

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 1),
            request_id=request_id,
        )
        return response

    # ── Exception handler ─────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(research_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "NeuroAgent",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level,
        workers=settings.workers,
    )
