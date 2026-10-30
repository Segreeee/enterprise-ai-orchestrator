# Roadmap MLOps / LLMOps — Enterprise AI Orchestrator

> 🇬🇧 English version: [MLOPS_ROADMAP_EN.md](MLOPS_ROADMAP_EN.md)

| | |
|---|---|
| **Stato** | Audit preliminare eseguito il 30/10/2026 · Fase 2 implementata (lock solo in-process) · Fase 6 parziale (errori, limiti di input, prompt injection, telemetria; mancano auth e rate limiting) · fasi 1, 3, 4, 5, 7 pianificate |
| **Ambito** | Evoluzione del sistema esistente verso LLMOps/MLOps: riproducibile, osservabile, sicuro, pronto per un eventuale rilascio in produzione |
| **Punto di partenza** | Il codice presente nel repository e la [mappa Archify dei flussi dati](data-flow/data-flow.html) |

> **Nota sulla numerazione.** Le fasi 1–7 di questa roadmap sono numerate in modo indipendente dalle fasi di sviluppo già completate nella storia del repository (scaffolding, RAG, agenti, API, Docker, security hardening, enterprise upgrades).

## Indice

- [Obiettivo](#obiettivo)
- [1. Vincoli architetturali](#1-vincoli-architetturali)
- [2. Fase preliminare: audit del repository](#2-fase-preliminare-audit-del-repository)
- [3. Fase 1 — Versionamento e valutazione del sistema RAG](#3-fase-1--versionamento-e-valutazione-del-sistema-rag)
- [4. Fase 2 — Affidabilità e versionamento dell'ingestion](#4-fase-2--affidabilità-e-versionamento-dellingestion)
- [5. Fase 3 — CI/CD automatizzata](#5-fase-3--cicd-automatizzata)
- [6. Fase 4 — Experiment tracking e tracciabilità](#6-fase-4--experiment-tracking-e-tracciabilità)
- [7. Fase 5 — Observability e monitoring](#7-fase-5--observability-e-monitoring)
- [8. Fase 6 — Security hardening](#8-fase-6--security-hardening)
- [9. Fase 7 — Deployment controllato e rollback](#9-fase-7--deployment-controllato-e-rollback)
- [10. Aggiornamento della documentazione Archify](#10-aggiornamento-della-documentazione-archify)
- [11. Metodo di esecuzione](#11-metodo-di-esecuzione)
- [12. Deliverable finali](#12-deliverable-finali)
- [Definizione di completamento](#definizione-di-completamento)

---

## Obiettivo

Analizzare e migliorare l'architettura esistente dell'Enterprise AI Orchestrator, evolvendola verso un sistema LLMOps/MLOps più completo, **riproducibile, osservabile, sicuro** e pronto per un eventuale deployment in produzione.

L'intervento parte dal codice realmente presente nel repository e dalla documentazione Archify esistente. Le funzionalità non documentate non vanno date per assenti: prima si verifica l'implementazione attuale, poi si individuano le lacune effettive e si propongono modifiche coerenti con le tecnologie già in uso.

---

## 1. Vincoli architetturali

Salvo motivazioni tecniche documentate, lo stack esistente resta invariato:

| Area | Tecnologie |
|---|---|
| Linguaggio | Python 3.12 |
| API e validazione | FastAPI, Uvicorn, Pydantic v2 |
| Modelli | Ollama con `llama3.2` e `nomic-embed-text` |
| Retrieval | ChromaDB, `rank_bm25`, Reciprocal Rank Fusion |
| Agenti | Orchestrazione multi-agente round-robin |
| Logging | Loguru |
| Qualità | pytest, mypy, mutation testing |
| Infrastruttura | Docker, Docker Compose, supporto GPU NVIDIA opzionale |
| Processo | Git, GitHub, Makefile |

**Regole:**

- Non sostituire componenti funzionanti senza una motivazione concreta.
- Non introdurre Kubernetes, sistemi distribuiti, cloud obbligatorio o infrastrutture enterprise se i requisiti non li giustificano.
- PyTorch è predisposto ma non usato: **non** implementare una pipeline di training artificiale solo per dichiarare il progetto MLOps.
- Mantenere il funzionamento in locale e la possibilità di eseguire i test **senza** un server Ollama attivo.

---

## 2. Fase preliminare: audit del repository

Prima di modificare il codice:

1. Esaminare la struttura del repository, il codice applicativo, i test, il Dockerfile, Docker Compose, il Makefile e la documentazione.
2. Analizzare il diagramma Archify e confrontare i flussi documentati con quelli realmente implementati.
3. Distinguere le funzionalità già presenti, quelle parziali e quelle mancanti.
4. Eseguire, dove possibile, la suite di test e i controlli statici esistenti.
5. Documentare i risultati iniziali e gli eventuali problemi preesistenti.

**Output:** una matrice con *area, stato attuale, lacuna, intervento proposto, priorità, criterio di accettazione*.

> ⚠️ Non dichiarare superati test o audit che non siano stati effettivamente eseguiti.

---

## 3. Fase 1 — Versionamento e valutazione del sistema RAG

Implementare un sistema riproducibile per valutare la qualità del retrieval e della generazione.

**Requisiti:**

- [ ] Dataset di benchmark **versionato**: query, documenti di riferimento e, quando disponibili, risposte attese.
- [ ] Procedura riproducibile per eseguire il benchmark.
- [ ] Valutazione **separata** di retrieval vettoriale, BM25 e ricerca ibrida RRF.
- [ ] Metriche di retrieval: **Recall@k**, **MRR** e, con giudizi di rilevanza graduati, **nDCG@k**.
- [ ] Metriche di generazione: validità degli output strutturati, aderenza alle fonti, correttezza delle citazioni, pertinenza delle risposte.
- [ ] Separazione tra test deterministici e valutazioni che dipendono dalla generazione probabilistica degli LLM.
- [ ] Soglie configurabili per identificare le regressioni.
- [ ] Report confrontabile tra esecuzioni, con configurazione, versione del modello, parametri di retrieval e risultati.

> Non usare un LLM come unico giudice della qualità. Servono metriche deterministiche e verifiche basate su riferimenti attendibili; le valutazioni automatiche aggiuntive si usano solo quando appropriate.

**✅ Criterio di accettazione:** il benchmark si esegue in modo ripetibile e permette di confrontare due configurazioni senza modificare manualmente il codice.

---

## 4. Fase 2 — Affidabilità e versionamento dell'ingestion

Analizzare il flusso di ingestion dei documenti e rafforzarne la consistenza.

**Requisiti:**

- [ ] Identificatori stabili per documento, versione e chunk.
- [ ] I chunk obsoleti non devono restare recuperabili dopo l'aggiornamento di un documento.
- [ ] Gestione delle ingestion concorrenti sullo stesso documento.
- [ ] Gestione degli errori parziali durante la generazione degli embedding e la scrittura su ChromaDB.
- [ ] Verifica della compatibilità tra modello di embedding, dimensione dei vettori e collezione usata.
- [ ] Strategia di attivazione della nuova versione e di recupero in caso di errore.
- [ ] Nessuna ipotesi di transazioni atomiche di ChromaDB sull'intero aggiornamento: documentare e implementare una strategia compatibile con le sue garanzie effettive.
- [ ] Test per: documenti più lunghi o più corti dopo l'aggiornamento, duplicati, errori, retry e concorrenza.

**✅ Criterio di accettazione:** un aggiornamento fallito non lascia il sistema in uno stato ambiguo e non espone silenziosamente una combinazione incoerente di versioni.

---

## 5. Fase 3 — CI/CD automatizzata

Integrare GitHub Actions, se non esiste già una pipeline equivalente.

**La pipeline esegue:**

1. Installazione riproducibile delle dipendenze.
2. Linting e controlli di qualità del codice.
3. mypy.
4. Test unitari e di integrazione compatibili con l'ambiente CI.
5. Test di regressione del benchmark, quando disponibili.
6. Controlli sulle dipendenze e scansione delle vulnerabilità.
7. Build dell'immagine Docker.
8. Verifica dell'avvio e degli health check in un ambiente di test.

**Regole:**

- Job separati per i test che richiedono Ollama o una GPU: non devono bloccare i controlli deterministici di base quando le risorse mancano.
- Immagini pubblicate con tag riconducibili al commit e, dove possibile, digest immutabili.
- Nessuna pubblicazione automatica in produzione di un modello o di un'immagine che non abbia superato i controlli previsti.

**✅ Criterio di accettazione:** ogni pull request attiva automaticamente i controlli pertinenti e ne impedisce l'integrazione quando i requisiti obbligatori falliscono.

---

## 6. Fase 4 — Experiment tracking e tracciabilità

Valutare MLflow come strumento di experiment tracking e di registrazione degli artefatti.

**Per ogni esecuzione significativa registrare, ove applicabile:**

| Categoria | Dati |
|---|---|
| Identità | Identificativo dell'esecuzione, commit Git, versione dell'immagine |
| Ambiente | Versioni di Python, delle dipendenze e delle librerie principali |
| Modelli | Modello LLM e modello di embedding effettivamente usati; versioni o digest degli artefatti, quando disponibili |
| Dati | Versione del dataset e del benchmark |
| Configurazione | Prompt e configurazione degli agenti; parametri di chunking, retrieval e RRF |
| Risultati | Metriche di valutazione, esito dei test, riferimenti ai report |

**Linee guida:**

- Distinguere correttamente tra experiment tracking, model registry e registrazione delle configurazioni di un sistema RAG.
- Non è necessario registrare come modello ML proprietario un LLM preaddestrato solo perché lo si usa tramite Ollama.
- Se MLflow introduce una complessità sproporzionata, proporre prima un'alternativa minima basata su **manifest JSON versionati** e artefatti conservati in modo affidabile.

**✅ Criterio di accettazione:** si può ricostruire la configurazione di un'esecuzione e confrontarne i risultati con quelli di altre versioni.

---

## 7. Fase 5 — Observability e monitoring

Estendere il logging esistente con metriche operative e applicative. Valutare Prometheus e Grafana oppure una soluzione più leggera, se sufficiente.

**Da monitorare, come minimo:**

- Numero di richieste e tasso di errore per endpoint.
- Latenza per endpoint e per agente.
- Timeout e indisponibilità di Ollama.
- Durata di ingestion, embedding e retrieval.
- Numero di chunk recuperati e distribuzione degli score.
- Dimensione degli input e dei prompt.
- Token usati e costi stimati, quando le informazioni sono disponibili.
- Uso di CPU, memoria e GPU, dove accessibile.
- Errori di validazione, retry e richieste interrotte.

**Regole:**

- Mantenere il logging JSON di Loguru e il contesto `conversation_id`.
- Non registrare dati personali, segreti, prompt sensibili o documenti integrali senza una necessità documentata.
- Health check distinti per la disponibilità del processo (*liveness*) e per quella delle dipendenze esterne (*readiness*).
- Soglie e alert iniziali basati su valori configurabili, senza inventare SLA o requisiti di latenza non concordati.

**✅ Criterio di accettazione:** un errore o un rallentamento significativo è individuabile attraverso metriche e log correlabili.

---

## 8. Fase 6 — Security hardening

**Verificare e rafforzare:**

- [ ] Autenticazione e autorizzazione degli endpoint, se il servizio deve essere accessibile fuori da un ambiente locale fidato.
- [ ] Rate limiting e limiti alla dimensione delle richieste.
- [ ] Timeout e risorse consumate dalle richieste.
- [ ] Gestione dei segreti e delle configurazioni sensibili.
- [ ] Scansione delle dipendenze e delle immagini Docker.
- [ ] Restrizioni di rete per i servizi interni, ove appropriate.
- [ ] Protezione contro prompt injection e contenuti malevoli provenienti dai documenti RAG.
- [ ] Isolamento di dati e risultati tra utenti o conversazioni, quando richiesto.
- [ ] Logging, retention e gestione dei dati personali.
- [ ] Gestione degli errori senza esporre dettagli interni o segreti.

Il precedente audit di sicurezza resta il riferimento; ogni correzione va verificata con test riproducibili.

**✅ Criterio di accettazione:** ogni controllo dichiarato come implementato ha evidenze verificabili nel codice, nella configurazione o nei test.

---

## 9. Fase 7 — Deployment controllato e rollback

Definire una procedura di rilascio riproducibile con ambienti separati, almeno logicamente, per test e produzione.

**Prevedere:**

- [ ] Configurazioni esterne al codice.
- [ ] Versionamento delle immagini Docker.
- [ ] Health check e smoke test.
- [ ] Verifica della disponibilità delle dipendenze.
- [ ] Migrazione o aggiornamento controllato degli artefatti RAG.
- [ ] Ripristino della versione applicativa precedente.
- [ ] Backup e recupero dei dati persistenti.
- [ ] Documentazione dei comandi di rilascio e rollback.

> Kubernetes non è un requisito iniziale. Docker Compose resta sufficiente per il deployment locale e per molti scenari di studio.

**✅ Criterio di accettazione:** si può eseguire un rilascio di prova, verificarne il funzionamento e ripristinare la versione precedente seguendo una procedura documentata.

---

## 10. Aggiornamento della documentazione Archify

Aggiornare la mappa interattiva e i diagrammi Mermaid perché rappresentino fedelmente:

- Ingestion e versionamento dei documenti.
- Pipeline di benchmark e valutazione.
- CI/CD.
- Tracking degli esperimenti e degli artefatti.
- Deployment e rollback.
- Monitoring e feedback loop.
- Confini di sicurezza e dipendenze esterne.

I componenti **implementati** vanno distinti chiaramente da quelli **pianificati**. Le funzionalità ancora da sviluppare non vanno rappresentate come operative.

---

## 11. Metodo di esecuzione

Si procede per fasi incrementali. **Per ogni fase:**

1. Analizzare il codice interessato.
2. Presentare brevemente la modifica proposta e i file coinvolti.
3. Implementare una soluzione coerente con le convenzioni esistenti.
4. Aggiungere o aggiornare i test.
5. Eseguire i controlli pertinenti.
6. Documentare risultati, limiti ed eventuali problemi residui.
7. Aggiornare diagramma e README **solo** dopo aver verificato la modifica.

**Da evitare:** riscritture massive, duplicazioni di responsabilità, dipendenze inutili e modifiche incompatibili non motivate. Non rimuovere funzionalità esistenti e non modificare silenziosamente contratti API, schema dei dati o comportamento dell'orchestrazione.

Se emerge una decisione architetturale rilevante che il repository non risolve, esplicitare le alternative e scegliere l'opzione più semplice e reversibile, salvo che serva una conferma.

---

## 12. Deliverable finali

- Matrice iniziale delle lacune.
- Piano delle fasi con stato di avanzamento.
- Codice e configurazioni implementati.
- Test aggiunti e relativi risultati effettivi.
- Report di benchmark prima/dopo, quando i dati consentono il confronto.
- Workflow CI/CD.
- Configurazione di monitoring.
- Procedure di deployment e rollback.
- Documentazione aggiornata.
- Diagramma Archify aggiornato.
- Elenco delle funzionalità completate, parziali e ancora pianificate.

---

## Definizione di completamento

Il progetto si può considerare migliorato quando:

- il sistema è **verificabile automaticamente**;
- le configurazioni sono **tracciabili**;
- la qualità del retrieval e della generazione è **misurabile**;
- i rilasci sono **controllati**;
- i problemi operativi sono **osservabili**.

> Il sistema non va dichiarato "MLOps enterprise-ready" solo perché tutte le dipendenze sono installate o tutti i test esistenti passano. Il risultato si valuta rispetto alle funzionalità effettivamente implementate, ai requisiti dichiarati e alle evidenze raccolte.
