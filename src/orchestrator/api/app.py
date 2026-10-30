import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TextIO

import httpx
import ollama
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from orchestrator import __version__
from orchestrator.api.dependencies import get_retriever
from orchestrator.api.routes import router

# Ollama failures, mapped to fixed messages: exception text can carry hosts, model names or paths.
_BACKEND_ERRORS: list[tuple[type[Exception], int, str]] = [
    (ConnectionError, status.HTTP_503_SERVICE_UNAVAILABLE, "model backend unavailable"),
    (ollama.ResponseError, status.HTTP_503_SERVICE_UNAVAILABLE, "model backend error"),
    (httpx.TimeoutException, status.HTTP_504_GATEWAY_TIMEOUT, "model backend timed out"),
]


def _backend_error(status_code: int, detail: str) -> Callable[[Request, Exception], JSONResponse]:
    def handler(request: Request, exc: Exception) -> JSONResponse:
        # Operators get the exception type; the message stays out of logs and responses alike.
        logger.warning(detail, path=request.url.path, error=type(exc).__name__)
        return JSONResponse(status_code=status_code, content={"detail": detail})

    return handler


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build the retriever at startup, so an embedding model that does not match the store stops
    # the app here instead of failing on the first request. Tests swap it via dependency_overrides.
    app.dependency_overrides.get(get_retriever, get_retriever)()
    yield


def configure_logging(sink: TextIO = sys.stderr) -> None:
    """Replace loguru's human-readable default with one JSON object per line, ready for a log collector."""
    logger.remove()
    logger.add(sink, serialize=True, level="INFO")


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Orchestrator", version=__version__, lifespan=_lifespan)
    app.include_router(router)
    for exc_type, status_code, detail in _BACKEND_ERRORS:
        app.add_exception_handler(exc_type, _backend_error(status_code, detail))
    return app


configure_logging()
app = create_app()
