from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from orchestrator import __version__

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
