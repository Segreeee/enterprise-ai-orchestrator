# Enterprise AI Orchestrator

Local-first GenAI service: RAG on ChromaDB and round-robin multi-agent conversations on Ollama,
behind a validated FastAPI API.

Servizio GenAI locale: RAG su ChromaDB e conversazioni multi-agente round-robin su Ollama,
dietro un'API FastAPI validata.

- 🇬🇧 **[English documentation](README_EN.md)**
- 🇮🇹 **[Documentazione in italiano](README_IT.md)**
- 🗺️ **[Interactive data-flow map](docs/data-flow/data-flow.html)**

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -m "not integration"
make docker-up
```
