from fastapi import FastAPI

from orchestrator import __version__
from orchestrator.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Orchestrator", version=__version__)
    app.include_router(router)
    return app


app = create_app()
