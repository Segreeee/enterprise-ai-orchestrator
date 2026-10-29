from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from orchestrator import __version__
from orchestrator.api.routes import router


def _backend_unavailable(request: Request, exc: Exception) -> JSONResponse:
    # The ollama client raises ConnectionError when the server is down or unreachable.
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"model backend unavailable: {exc}"},
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Orchestrator", version=__version__)
    app.include_router(router)
    app.add_exception_handler(ConnectionError, _backend_unavailable)
    return app


app = create_app()
