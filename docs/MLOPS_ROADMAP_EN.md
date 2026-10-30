# MLOps / LLMOps Roadmap — Enterprise AI Orchestrator

> 🇮🇹 Versione italiana: [MLOPS_ROADMAP_IT.md](MLOPS_ROADMAP_IT.md)

| | |
|---|---|
| **Status** | Preliminary audit done on 2026-10-30 · Phase 2 implemented (in-process lock only) · Phase 6 partial (errors, input limits, prompt injection, telemetry; auth and rate limiting missing) · Phases 1, 3, 4, 5, 7 planned |
| **Scope** | Evolve the existing system towards LLMOps/MLOps: reproducible, observable, secure, ready for an eventual production deployment |
| **Starting point** | The code in this repository and the [Archify data-flow map](data-flow/data-flow.html) |

> **Note on numbering.** Phases 1–7 of this roadmap are numbered independently of the development phases already completed in the repository history (scaffolding, RAG, agents, API, Docker, security hardening, enterprise upgrades).

## Contents

- [Objective](#objective)
- [1. Architectural constraints](#1-architectural-constraints)
- [2. Preliminary phase: repository audit](#2-preliminary-phase-repository-audit)
- [3. Phase 1 — RAG versioning and evaluation](#3-phase-1--rag-versioning-and-evaluation)
- [4. Phase 2 — Ingestion reliability and versioning](#4-phase-2--ingestion-reliability-and-versioning)
- [5. Phase 3 — Automated CI/CD](#5-phase-3--automated-cicd)
- [6. Phase 4 — Experiment tracking and traceability](#6-phase-4--experiment-tracking-and-traceability)
- [7. Phase 5 — Observability and monitoring](#7-phase-5--observability-and-monitoring)
- [8. Phase 6 — Security hardening](#8-phase-6--security-hardening)
- [9. Phase 7 — Controlled deployment and rollback](#9-phase-7--controlled-deployment-and-rollback)
- [10. Archify documentation update](#10-archify-documentation-update)
- [11. Execution method](#11-execution-method)
- [12. Final deliverables](#12-final-deliverables)
- [Definition of done](#definition-of-done)

---

## Objective

Analyse and improve the existing Enterprise AI Orchestrator architecture, evolving it into a more complete LLMOps/MLOps system that is **reproducible, observable and secure**, and ready for an eventual production deployment.

The work starts from the code actually in the repository and from the existing Archify documentation. Undocumented features must not be assumed absent: first verify the current implementation, then identify the real gaps and propose changes consistent with the technologies already in use.

---

## 1. Architectural constraints

Unless there is a documented technical reason, the existing stack stays:

| Area | Technologies |
|---|---|
| Language | Python 3.12 |
| API and validation | FastAPI, Uvicorn, Pydantic v2 |
| Models | Ollama with `llama3.2` and `nomic-embed-text` |
| Retrieval | ChromaDB, `rank_bm25`, Reciprocal Rank Fusion |
| Agents | Round-robin multi-agent orchestration |
| Logging | Loguru |
| Quality | pytest, mypy, mutation testing |
| Infrastructure | Docker, Docker Compose, optional NVIDIA GPU support |
| Process | Git, GitHub, Makefile |

**Rules:**

- Do not replace working components without a concrete reason.
- Do not introduce Kubernetes, distributed systems, mandatory cloud or enterprise infrastructure unless the requirements justify it.
- PyTorch is installed but unused: do **not** build an artificial training pipeline just to call the project MLOps.
- Keep local operation working, and keep the tests runnable **without** a live Ollama server.

---

## 2. Preliminary phase: repository audit

Before changing any code:

1. Review the repository layout, application code, tests, Dockerfile, Docker Compose file, Makefile and documentation.
2. Analyse the Archify diagram and compare the documented flows with the implemented ones.
3. Separate features that are present, partial and missing.
4. Run the existing test suite and static checks wherever possible.
5. Document the initial results and any pre-existing problems.

**Output:** a matrix with *area, current state, gap, proposed change, priority, acceptance criterion*.

> ⚠️ Never report tests or audits as passed unless they were actually run.

---

## 3. Phase 1 — RAG versioning and evaluation

Build a reproducible way to evaluate retrieval and generation quality.

**Requirements:**

- [ ] A **versioned** benchmark dataset: queries, reference documents and, when available, expected answers.
- [ ] A reproducible procedure to run the benchmark.
- [ ] **Separate** evaluation of vector retrieval, BM25 and hybrid RRF search.
- [ ] Retrieval metrics: **Recall@k**, **MRR** and, with graded relevance judgements, **nDCG@k**.
- [ ] Generation metrics: structured-output validity, faithfulness to sources, citation correctness, answer relevance.
- [ ] A split between deterministic tests and evaluations that depend on probabilistic LLM generation.
- [ ] Configurable thresholds to detect regressions.
- [ ] A report comparable across runs, including configuration, model version, retrieval parameters and results.

> Never use an LLM as the only judge of quality. Use deterministic metrics and checks against trusted references; add automated evaluations only where appropriate.

**✅ Acceptance criterion:** the benchmark runs repeatably and can compare two configurations without manual code changes.

---

## 4. Phase 2 — Ingestion reliability and versioning

Analyse the document ingestion flow and strengthen its consistency.

**Requirements:**

- [ ] Stable identifiers for document, version and chunk.
- [ ] Obsolete chunks must not remain retrievable after a document update.
- [ ] Handling of concurrent ingests of the same document.
- [ ] Handling of partial failures while embedding and while writing to ChromaDB.
- [ ] A compatibility check between embedding model, vector dimension and collection.
- [ ] A strategy to activate the new version and to recover on failure.
- [ ] No assumption that ChromaDB offers atomic transactions over the whole update: document and implement a strategy that fits its actual guarantees.
- [ ] Tests for longer or shorter documents after an update, duplicates, failures, retries and concurrency.

**✅ Acceptance criterion:** a failed update never leaves the system in an ambiguous state or silently exposes an inconsistent mix of versions.

---

## 5. Phase 3 — Automated CI/CD

Add GitHub Actions, unless an equivalent pipeline already exists.

**The pipeline runs:**

1. Reproducible dependency installation.
2. Linting and code-quality checks.
3. mypy.
4. Unit tests, plus integration tests that can run in CI.
5. Benchmark regression tests, when available.
6. Dependency checks and vulnerability scanning.
7. Docker image build.
8. Start-up and health-check verification in a test environment.

**Rules:**

- Separate jobs for tests that need Ollama or a GPU: they must not block the basic deterministic checks when those resources are missing.
- Tag published images with the commit and, where possible, immutable digests.
- Never auto-publish to production a model or image that has not passed the required checks.

**✅ Acceptance criterion:** every pull request automatically triggers the relevant checks, and merging is blocked when mandatory requirements fail.

---

## 6. Phase 4 — Experiment tracking and traceability

Evaluate MLflow for experiment tracking and artifact logging.

**For every significant run, record where applicable:**

| Category | Data |
|---|---|
| Identity | Run id, Git commit, image version |
| Environment | Python, dependency and key library versions |
| Models | LLM and embedding model actually used; artifact versions or digests, when available |
| Data | Dataset and benchmark version |
| Configuration | Prompts and agent configuration; chunking, retrieval and RRF parameters |
| Results | Evaluation metrics, test outcome, links to reports |

**Guidelines:**

- Keep experiment tracking, model registry and RAG-system configuration logging clearly apart.
- A pre-trained LLM served by Ollama does not need to be registered as a proprietary ML model.
- If MLflow adds disproportionate complexity, first propose a minimal alternative based on **versioned JSON manifests** and reliably stored artifacts.

**✅ Acceptance criterion:** a run's configuration can be reconstructed and its results compared with those of other versions.

---

## 7. Phase 5 — Observability and monitoring

Extend the existing logging with operational and application metrics. Evaluate Prometheus and Grafana, or a lighter solution if it is enough.

**Monitor at least:**

- Request count and error rate per endpoint.
- Latency per endpoint and per agent.
- Ollama timeouts and outages.
- Ingestion, embedding and retrieval duration.
- Number of retrieved chunks and score distribution.
- Input and prompt size.
- Tokens used and estimated cost, when available.
- CPU, memory and GPU usage, where accessible.
- Validation errors, retries and aborted requests.

**Rules:**

- Keep Loguru's JSON logging and the `conversation_id` context.
- Do not log personal data, secrets, sensitive prompts or full documents without a documented need.
- Separate health checks for process availability (*liveness*) and for external dependencies (*readiness*).
- Initial thresholds and alerts based on configurable values; do not invent SLAs or latency targets that were never agreed.

**✅ Acceptance criterion:** a significant error or slowdown can be found through correlated metrics and logs.

---

## 8. Phase 6 — Security hardening

**Verify and strengthen:**

- [ ] Endpoint authentication and authorisation, if the service must be reachable beyond a trusted local environment.
- [ ] Rate limiting and request size limits.
- [ ] Timeouts and per-request resource consumption.
- [ ] Secrets and sensitive configuration management.
- [ ] Dependency and Docker image scanning.
- [ ] Network restrictions for internal services, where appropriate.
- [ ] Protection against prompt injection and malicious content coming from RAG documents.
- [ ] Data and result isolation between users or conversations, when required.
- [ ] Logging, retention and handling of personal data.
- [ ] Error handling that exposes no internal details or secrets.

The previous security audit remains the reference; every fix is verified with reproducible tests.

**✅ Acceptance criterion:** every control declared as implemented has verifiable evidence in code, configuration or tests.

---

## 9. Phase 7 — Controlled deployment and rollback

Define a reproducible release procedure with environments separated, at least logically, for test and production.

**Provide:**

- [ ] Configuration kept outside the code.
- [ ] Versioned Docker images.
- [ ] Health checks and smoke tests.
- [ ] Dependency availability checks.
- [ ] Controlled migration or update of RAG artifacts.
- [ ] Rollback to the previous application version.
- [ ] Backup and recovery of persistent data.
- [ ] Documented release and rollback commands.

> Kubernetes is not an initial requirement. Docker Compose remains enough for local deployment and for many study scenarios.

**✅ Acceptance criterion:** a trial release can be run, verified and rolled back to the previous version by following a documented procedure.

---

## 10. Archify documentation update

Update the interactive map and the Mermaid diagrams so they faithfully show:

- Document ingestion and versioning.
- The benchmark and evaluation pipeline.
- CI/CD.
- Experiment and artifact tracking.
- Deployment and rollback.
- Monitoring and the feedback loop.
- Security boundaries and external dependencies.

Clearly separate **implemented** components from **planned** ones. Never show features still to be built as operational.

---

## 11. Execution method

Work in incremental phases. **For each phase:**

1. Analyse the affected code.
2. Briefly present the proposed change and the files involved.
3. Implement it following the existing conventions.
4. Add or update tests.
5. Run the relevant checks.
6. Document results, limits and any remaining issues.
7. Update the diagram and README **only** once the change is verified.

**Avoid:** large rewrites, duplicated responsibilities, unnecessary dependencies and unjustified breaking changes. Do not remove existing features or silently change API contracts, data schemas or orchestration behaviour.

If a significant architectural decision cannot be settled from the repository, state the alternatives and pick the simplest, most reversible option, unless confirmation is needed.

---

## 12. Final deliverables

- Initial gap matrix.
- Phase plan with progress status.
- Implemented code and configuration.
- Added tests and their actual results.
- Before/after benchmark report, when the data allows a comparison.
- CI/CD workflow.
- Monitoring configuration.
- Deployment and rollback procedures.
- Updated documentation.
- Updated Archify diagram.
- List of completed, partial and still-planned features.

---

## Definition of done

The project counts as improved when:

- the system is **automatically verifiable**;
- configurations are **traceable**;
- retrieval and generation quality is **measurable**;
- releases are **controlled**;
- operational problems are **observable**.

> Do not call the system "enterprise-ready MLOps" just because all dependencies are installed or all existing tests pass. Judge the result against the features actually implemented, the stated requirements and the evidence collected.
