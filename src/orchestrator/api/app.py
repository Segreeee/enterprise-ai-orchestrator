import sys
from typing import TextIO

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from orchestrator import __version__
from orchestrator.api.routes import router


def _backend_unavailable(request: Request, exc: Exception) -> JSONResponse:
    # The ollama client raises ConnectionError when the server is down or unreachable.
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"model backend unavailable: {exc}"},
    )


def configure_logging(sink: TextIO = sys.stderr) -> None:
    """Replace loguru's human-readable default with one JSON object per line, ready for a log collector."""
    logger.remove()
    logger.add(sink, serialize=True, level="INFO")


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Orchestrator", version=__version__)
    app.include_router(router)
    app.add_exception_handler(ConnectionError, _backend_unavailable)
    return app


configure_logging()
app = create_app()
