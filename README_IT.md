# Enterprise AI Orchestrator

> Un servizio GenAI che gira in locale: Retrieval-Augmented Generation su ChromaDB e
> conversazioni multi-agente round-robin su Ollama, dietro un'API FastAPI validata.
> Sviluppato test-first e distribuito come immagine Docker multi-stage.

🇬🇧 [Read in English](README_EN.md) · 🗺️ [Mappa interattiva dei flussi di dati](docs/data-flow/data-flow.html)

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)
![Pydantic](https://img.shields.io/badge/Pydantic-v2%20strict-e92063)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black)
![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20integration-success)

---

## Indice

- [Panoramica](#panoramica)
- [Architettura](#architettura)
- [Scelte progettuali](#scelte-progettuali)
- [Struttura del repository](#struttura-del-repository)
- [Installazione](#installazione)
- [Test](#test)
- [Esecuzione con Docker](#esecuzione-con-docker)
- [API ed esempi](#api-ed-esempi)
- [Configurazione](#configurazione)
- [Limiti noti e roadmap](#limiti-noti-e-roadmap)

---

## Panoramica

L'Enterprise AI Orchestrator espone tre funzionalità via HTTP:

| Funzionalità | Endpoint | Cosa fa |
|---|---|---|
| **Ingestione della conoscenza** | `POST /ingest` | Divide un documento in chunk sovrapposti, ne calcola gli embedding con Ollama e li salva in ChromaDB |
| **Ricerca semantica** | `POST /query` | Calcola l'embedding della domanda e restituisce i chunk più simili, con uno score normalizzato in `[0, 1]` |
| **Ragionamento multi-agente** | `POST /conversations` | Fa lavorare più agenti LLM in ordine round-robin rigoroso su un task condiviso; a ogni turno può aggiungere contesto RAG |

### Obiettivi

- **Tutto in locale.** I modelli girano su Ollama: nessun dato lascia la macchina e non c'è un costo per token.
- **Ogni confine è validato.** I modelli Pydantic v2 sono strict, immutabili e rifiutano i campi sconosciuti. Vengono validati i payload HTTP, le risposte dell'LLM, gli embedding e i risultati del vector store.
- **Testabile senza GPU né modello.** LLM ed embedder stanno dietro piccoli `Protocol`, quindi tutta la suite unitaria gira in pochi secondi con fake deterministici. I test contro un Ollama reale sono un livello separato e facoltativo.
- **Distribuito come artefatto riproducibile.** Un'immagine Docker multi-stage che gira come utente non-root, con health check, più uno stack Compose che scarica i modelli al primo avvio.
- **Disciplina di sviluppo.** Ogni modulo è nato in TDD: prima un test che fallisce, poi l'implementazione.

---

## Architettura

```mermaid
flowchart LR
    C[Client API<br/>curl / httpx] -->|JSON| API

    subgraph API[FastAPI · api/]
        R1[POST /ingest]
        R2[POST /query]
        R3[POST /conversations]
    end

    subgraph Dominio
        RET[Retriever<br/>rag/retriever.py]
        CH[Chunker<br/>rag/chunker.py]
        ORC[Orchestrator<br/>agents/orchestrator.py]
        RR[RoundRobin<br/>agents/round_robin.py]
    end

    subgraph Backend
        EMB[(Ollama<br/>nomic-embed-text)]
        LLM[(Ollama<br/>llama3.2)]
        DB[(ChromaDB<br/>coseno)]
    end

    R1 --> RET
    R2 --> RET
    R3 --> ORC
    RET --> CH
    RET -->|OllamaEmbedder| EMB
    RET -->|VectorStore| DB
    ORC --> RR
    ORC -->|contesto top-k| RET
    ORC -->|OllamaClient| LLM
```

Per una versione più ricca e interattiva, con la spiegazione di ogni flusso, apri la
**[mappa dei flussi di dati](docs/data-flow/data-flow.html)** nel browser (tema chiaro/scuro, export SVG/PNG). La mappa è in inglese.

### Componenti

| Modulo | Tipi principali | Responsabilità |
|---|---|---|
| `config.py` | `Settings` | Configurazione tipizzata da variabili d'ambiente (`OLLAMA_HOST` validato come `HttpUrl`) |
| `api/app.py` | `create_app()` | Factory dell'app; trasforma `ConnectionError` in **503** |
| `api/routes.py` | `/health`, `/ingest`, `/query`, `/conversations` | Strato HTTP sottile, senza logica di business |
| `api/schemas.py` | `QueryRequest`, `ConversationRequest`, … | Contratti di richiesta e risposta con limiti (`k ≤ 50`, `max_turns ≤ 20`, nomi degli agenti univoci) |
| `api/dependencies.py` | `get_llm`, `get_retriever`, `get_settings` | Singleton creati alla prima richiesta; nei test si sostituiscono con `dependency_overrides` |
| `agents/schemas.py` | `Message`, `AgentSpec`, `Turn`, `Conversation` | Modello di dominio della conversazione |
| `agents/round_robin.py` | `RoundRobin` | Scheduler puro: ordine ciclico fisso, rifiuta liste vuote o con duplicati |
| `agents/orchestrator.py` | `Orchestrator` | Loop dei turni: sceglie l'agente, costruisce il prompt, aggiunge il contesto RAG, chiama l'LLM, si ferma a `max_turns` o allo `stop_token` |
| `models/llm.py` | `LLMClient` (Protocol), `OllamaClient` | `generate(list[Message]) -> Message` |
| `models/embeddings.py` | `Embedder` (Protocol), `OllamaEmbedder`, `Embedding` | Da testo a vettore di dimensione fissa e valori finiti (dimensione controllata da Pydantic) |
| `rag/schemas.py` | `Document`, `Chunk`, `RetrievalResult` | Modello di dominio RAG; score vincolato a `[0, 1]` |
| `rag/chunker.py` | `Chunker` | Finestre di caratteri con sovrapposizione; verifica che `overlap < size` |
| `rag/store.py` | `VectorStore` | Wrapper di ChromaDB (in memoria o persistente), spazio coseno, upsert idempotente |
| `rag/retriever.py` | `Retriever` | `ingest(Document)` e `retrieve(query, k)` |

### Come funziona una conversazione

1. `RoundRobin` sceglie il prossimo agente. La rotazione riparte da capo a ogni esecuzione, quindi i risultati sono riproducibili.
2. Se il RAG è attivo, il `Retriever` cerca con il task al primo turno e poi con l'ultima risposta, così il contesto segue la discussione.
3. L'agente di turno riceve:
   - il proprio system prompt, con in coda una sezione numerata `Relevant context`;
   - il task;
   - i propri turni precedenti come messaggi `assistant`;
   - i turni degli altri agenti come messaggi `user`, preceduti dal nome (`[nome-agente]`).
4. La risposta viene validata come `Message` e aggiunta come `Turn`.
5. Il loop termina a `max_turns` oppure appena una risposta contiene lo `stop_token`. Il campo `stop_reason` indica quale dei due è successo.

---

## Scelte progettuali

| Scelta | Perché |
|---|---|
| **`Protocol` per `LLMClient` ed `Embedder`** | Il dominio non importa mai tipi di un SDK specifico. Sostituire Ollama con un altro provider significa aggiungere una classe, non rifattorizzare. I fake non richiedono librerie di mocking. |
| **Modelli Pydantic strict, frozen, `extra="forbid"`** | Un errore di battitura nel payload fallisce subito con un 422 e gli oggetti non si possono modificare dopo la validazione. |
| **L'output dell'LLM viene rivalidato** | Una risposta vuota o con un ruolo inatteso solleva una `ValidationError` invece di corrompere in silenzio la conversazione. |
| **La re-ingestione sostituisce il documento in modo sicuro** | I nuovi chunk vengono calcolati e scritti per primi (upsert per id del chunk); solo dopo si cancellano i chunk in eccesso della versione vecchia. Se l'embedding fallisce o ChromaDB rifiuta la scrittura resta la versione precedente, e una re-ingestione più corta non lascia chunk orfani. |
| **Lavoro limitato per richiesta** | `Document` limita il testo a 50.000 caratteri, l'id a 256, i metadati a 16 voci con chiavi ≤ 64 e valori testuali ≤ 256 caratteri. I payload troppo grandi ricevono 422 prima di qualsiasi embedding. I client Ollama hanno un timeout di 60 s, così un backend bloccato non occupa un thread per sempre. |
| **Distanza coseno convertita in `[0, 1]`** | Chroma restituisce una distanza in `[0, 2]`; `score = 1 - d/2`, limitato all'intervallo, è una similarità intuitiva per il client. |
| **503 se il backend non risponde** | Se Ollama è giù il client riceve un *Service Unavailable* che si può ritentare, non un 500 generico. |
| **Limiti rigidi sui fattori di costo** | `max_turns ≤ 20` (ogni turno è una chiamata all'LLM) e `k ≤ 50` sono imposti a livello di API. |
| **Dipendenze create alla prima richiesta** | Importare l'app non apre connessioni né crea cartelle, quindi test e strumenti non hanno effetti collaterali. |
| **PyTorch solo CPU** | L'indice CPU di PyTorch evita di scaricare circa 2 GB di librerie CUDA nell'immagine. |

---

## Struttura del repository

```text
.
├── src/orchestrator/
│   ├── config.py
│   ├── api/          # factory dell'app, route, schemi di richiesta/risposta, DI
│   ├── agents/       # Message/AgentSpec/Turn/Conversation, RoundRobin, Orchestrator
│   ├── models/       # LLMClient + OllamaClient, Embedder + OllamaEmbedder
│   └── rag/          # Document/Chunk/RetrievalResult, Chunker, VectorStore, Retriever
├── tests/
│   ├── fakes.py      # ScriptedLLM, FakeRetriever, KeywordEmbedder (fake deterministici)
│   ├── conftest.py   # fixture condivise: embedder, store in memoria, llm
│   ├── api/  agents/  models/  rag/
│   └── models/test_ollama_integration.py   # @pytest.mark.integration
├── docs/data-flow/   # mappa interattiva dei flussi (HTML) e il suo sorgente
├── Dockerfile        # multi-stage, non-root, HEALTHCHECK
├── docker-compose.yml
├── Makefile
├── requirements.txt / requirements-dev.txt
└── pyproject.toml    # configurazione e marker di pytest
```

---

## Installazione

### Prerequisiti

- Python **3.12**
- [Ollama](https://ollama.com) in esecuzione in locale (serve solo per l'inferenza reale e i test di integrazione)
- Docker con Compose v2 (facoltativo)

### Installazione locale

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Scaricare i modelli

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Avviare l'API

```bash
make run
# oppure, senza make:
python -m uvicorn orchestrator.api.app:app --app-dir src --reload
```

La documentazione OpenAPI interattiva è su <http://localhost:8000/docs>.

> **Nota per Windows:** `make` non è installato di default (`choco install make`). Ogni target del
> `Makefile` è un solo comando che puoi anche lanciare a mano.

---

## Test

La suite ha due livelli, separati da un marker di pytest registrato in `pyproject.toml` (`--strict-markers` è attivo).

| Livello | Comando | Serve | Durata |
|---|---|---|---|
| Unità e API | `pytest -m "not integration"` | niente di esterno | circa 1-5 s |
| Integrazione | `pytest -m integration` | Ollama con `llama3.2` e `nomic-embed-text` | decine di secondi |
| Tutto | `pytest` (o `make test`) | — | — |

- I test unitari usano i fake deterministici di `tests/fakes.py` e un **ChromaDB in memoria** (`EphemeralClient`, con una collection dal nome univoco per ogni test).
- I test dell'API usano l'app FastAPI vera tramite `TestClient` e sostituiscono solo LLM e retriever con `app.dependency_overrides`.
- I test di integrazione **si saltano da soli**, spiegando il motivo, se Ollama non è raggiungibile o manca un modello. Così la CI senza server di modelli resta verde.

```bash
pytest -m integration -rs   # -rs mostra il motivo dei test saltati
```

---

## Esecuzione con Docker

### Stack completo (consigliato)

```bash
make docker-up      # docker compose up -d --build
make docker-down    # docker compose down
```

| Servizio | Immagine | Ruolo |
|---|---|---|
| `ollama` | `ollama/ollama` | Server dei modelli, con health check `ollama list`; i modelli restano nel volume `ollama-models` |
| `ollama-init` | `ollama/ollama` | Job una tantum: scarica `${MODEL_NAME:-llama3.2}` e `nomic-embed-text`, poi termina |
| `orchestrator` | costruita dal `Dockerfile` | L'API sulla porta **8000**; parte solo dopo che `ollama-init` è terminato con successo; ChromaDB resta nel volume `chroma-data` montato su `/data/chroma` |

Il primo avvio scarica i modelli (circa 2,3 GB) nel volume; quelli successivi li riusano.
Per usare una GPU NVIDIA, togli il commento al blocco `deploy` del servizio `ollama` (richiede `nvidia-container-toolkit`).

### Solo l'immagine

```bash
make docker-build   # docker build -t enterprise-ai-orchestrator .
make docker-run     # docker run --rm -p 8000:8000 enterprise-ai-orchestrator
```

L'immagine viene costruita in due stadi:

- **Builder:** un virtualenv con le dipendenze installate.
- **Runtime:** `python:3.12-slim` con solo il virtualenv e `src/`. Gira come utente `app` (uid 1000) e ha un `HEALTHCHECK` su `/health` scritto con la sola libreria standard.

---

## API ed esempi

Le risposte qui sotto vengono da un'esecuzione locale reale contro Ollama (`llama3.2`, `nomic-embed-text`).

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "version": "0.1.0"}
```

### `POST /ingest` → `201 Created`

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
        "id": "caching-guide",
        "text": "Redis supports per-key TTL expiration. An LRU policy evicts the least recently used keys when memory is full. Write-through caching updates the cache and the database in the same request.",
        "metadata": {"source": "caching.md"}
      }'
```
```json
{"document_id": "caching-guide", "chunks": 1}
```

Ingerire di nuovo lo stesso `id` sostituisce tutti i chunk precedenti. I valori di `metadata` devono essere semplici: testo, numero o booleano. Limiti: `text` ≤ 50.000 caratteri, `id` ≤ 256, al massimo 16 voci di metadati, chiavi ≤ 64 e valori testuali ≤ 256 caratteri; oltre si riceve 422.

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How are old keys evicted?", "k": 2}'
```
```json
{
  "results": [
    {
      "chunk": {
        "id": "caching-guide:0",
        "document_id": "caching-guide",
        "index": 0,
        "text": "Redis supports per-key TTL expiration. An LRU policy evicts ...",
        "metadata": {"source": "caching.md"}
      },
      "score": 0.797
    }
  ]
}
```

### `POST /conversations`

```bash
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{
        "task": "Propose a caching strategy for a product catalog API. Answer in at most 2 sentences.",
        "agents": [
          {"name": "architect", "system_prompt": "You are a pragmatic software architect. Be concise."},
          {"name": "reviewer",  "system_prompt": "You review designs and point out one risk. Say APPROVED when satisfied."}
        ],
        "max_turns": 4,
        "stop_token": "APPROVED",
        "use_rag": true,
        "k": 2
      }'
```
```json
{
  "task": "Propose a caching strategy for a product catalog API. Answer in at most 2 sentences.",
  "turns": [
    {"agent": "architect", "message": {"role": "assistant", "content": "Implement a hybrid caching strategy using Redis with a TTL ..."}},
    {"agent": "reviewer",  "message": {"role": "assistant", "content": "... APPROVED"}}
  ],
  "stop_reason": "stop_token"
}
```

| Campo | Default | Vincolo |
|---|---|---|
| `task` | — | non vuoto |
| `agents` | — | almeno 1, `name` univoci, nessun campo extra |
| `max_turns` | `6` | da 1 a 20 |
| `stop_token` | `null` | non vuoto se presente |
| `use_rag` | `true` | — |
| `k` | `4` | da 1 a 50 |

### Client Python

```python
import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=120) as api:
    api.post("/ingest", json={"id": "faq", "text": "I rimborsi vengono elaborati entro 5 giorni lavorativi."}).raise_for_status()

    hits = api.post("/query", json={"query": "Quanto tempo richiede un rimborso?", "k": 3}).json()["results"]
    for hit in hits:
        print(f"{hit['score']:.2f}  {hit['chunk']['text']}")

    conv = api.post("/conversations", json={
        "task": "Scrivi una risposta a un cliente che chiede i tempi di rimborso.",
        "agents": [
            {"name": "writer", "system_prompt": "Scrivi risposte di assistenza cordiali."},
            {"name": "editor", "system_prompt": "Rendi il testo più conciso. Scrivi DONE quando la risposta è definitiva."},
        ],
        "max_turns": 4,
        "stop_token": "DONE",
    }).json()
    for turn in conv["turns"]:
        print(f"[{turn['agent']}] {turn['message']['content']}")
```

### Codici di risposta

| Stato | Quando |
|---|---|
| `201` | Documento ingerito |
| `422` | Errore di validazione: query o task vuoti, `k`/`max_turns` fuori intervallo, nomi di agenti duplicati, campi sconosciuti, metadata non semplici |
| `503` | Ollama non raggiungibile (`{"detail": "model backend unavailable: ..."}`); la richiesta si può ritentare |

---

## Configurazione

| Variabile | Default | Usata da |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Client LLM ed embedding (deve essere un URL `http(s)` valido) |
| `CHROMA_PATH` | `chroma_db` | Cartella persistente di ChromaDB |
| `MODEL_NAME` | `llama3.2` | Modello di chat usato dagli agenti |

Il modello di embedding è `nomic-embed-text` (768 dimensioni), il default di `OllamaEmbedder`.

---

## Limiti noti e roadmap

Questi limiti sono elencati di proposito: sono le prossime iterazioni, non debito nascosto.

- **PyTorch è installato ma non ancora usato.** Gli embedding arrivano da Ollama. È presente per un embedder locale o un fine-tuning futuri, e occupa gran parte dei 2 GB dell'immagine. Se non ti serve, toglilo da `requirements.txt`.
- **Una richiesta di embedding per chunk.** L'API `embed` di Ollama accetta più testi insieme; raggrupparli velocizzerebbe le ingestioni grandi.
- **Nessuna autenticazione né rate limiting.** Metti l'API dietro un gateway prima di esporla.
- **Chiamate all'LLM sincrone.** Le conversazioni lunghe occupano un thread; streaming e client async sono il passo successivo naturale.
- **`ollama/ollama:latest` non è fissato a una versione.** Per deploy riproducibili va fissata.
- **Modello e dimensione degli embedding non sono ancora configurabili** tramite variabili d'ambiente.

---

<sub>Sviluppato test-first: ogni modulo è partito da un test che falliva. Vedi <code>tests/</code>.</sub>
