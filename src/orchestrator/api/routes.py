from typing import Annotated

from fastapi import APIRouter, Depends, status

from orchestrator import __version__
from orchestrator.agents.orchestrator import Orchestrator
from orchestrator.agents.schemas import Conversation
from orchestrator.api.dependencies import get_llm, get_retriever
from orchestrator.api.schemas import (
    ConversationRequest,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from orchestrator.models.llm import LLMClient
from orchestrator.rag.retriever import Retriever
from orchestrator.rag.schemas import Document

router = APIRouter()

RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest(document: Document, retriever: RetrieverDep) -> IngestResponse:
    return IngestResponse(document_id=document.id, chunks=retriever.ingest(document))


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, retriever: RetrieverDep) -> QueryResponse:
    return QueryResponse(results=retriever.retrieve(request.query, k=request.k))


@router.post("/conversations", response_model=Conversation)
def run_conversation(request: ConversationRequest, llm: LLMDep, retriever: RetrieverDep) -> Conversation:
    orchestrator = Orchestrator(request.agents, llm, retriever=retriever if request.use_rag else None, k=request.k)
    return orchestrator.run(request.task, max_turns=request.max_turns, stop_token=request.stop_token)
