import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TextIO

import httpx
import ollama
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from orchestrator import __version__
from orchestrator.api.dependencies import get_retriever
from orchestrator.api.routes import router
from orchestrator.config import Settings

# Largest valid body: a 50,000-character document JSON-escaped at <=12 bytes per code point plus
# id and metadata stays under 0.7 MB, so 1 MB never rejects a valid request.
MAX_BODY_BYTES = 1_000_000

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


class BodySizeLimit:
    """Answer 413 to bodies over the limit before the app buffers or parses them.

    Pydantic's field caps only run after FastAPI has read and decoded the whole body, so without
    this a single request can make the shared process allocate several times any size it likes.
    """

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.limit:
            await _too_large(scope, receive, send)
            return
        # Read at most `limit` bytes ourselves (chunked uploads have no Content-Length), then replay them.
        messages: list[Message] = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break
            size += len(message.get("body", b""))
            if size > self.limit:
                await _too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        buffered = iter(messages)

        async def replay() -> Message:
            return next(buffered, None) or await receive()

        await self.app(scope, replay, send)


async def _too_large(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse({"detail": "request body too large"}, status_code=status.HTTP_413_CONTENT_TOO_LARGE)
    await response(scope, receive, send)


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    docs = settings.environment == "development"
    app = FastAPI(
        title="Enterprise AI Orchestrator",
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.include_router(router)
    for exc_type, status_code, detail in _BACKEND_ERRORS:
        app.add_exception_handler(exc_type, _backend_error(status_code, detail))
    # Added last, so it runs first: a foreign Host is refused before any body is read.
    app.add_middleware(BodySizeLimit)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    return app


configure_logging()
app = create_app()
