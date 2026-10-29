"""Process-wide singletons, built lazily on first request; tests replace them via dependency_overrides."""

from functools import lru_cache

from orchestrator.config import Settings
from orchestrator.models.embeddings import OllamaEmbedder
from orchestrator.models.llm import LLMClient, OllamaClient
from orchestrator.rag.retriever import Retriever
from orchestrator.rag.store import VectorStore


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(
        embedder=OllamaEmbedder(host=str(settings.ollama_host)),
        store=VectorStore.persistent(settings.chroma_path),
    )


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return OllamaClient(model=settings.model_name, host=str(settings.ollama_host))
