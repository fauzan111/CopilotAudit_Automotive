# CopilotAudit_Automotive

A guardrailed, GDPR-compliant diagnostic copilot for automotive technicians, with
EU AI Act Article 12 audit logging and risk classification built in from the start
rather than added after the fact.

Two things live in one codebase on purpose, because they share the same underlying
infrastructure:

1. **Diagnostic copilot** - a time-series degradation/anomaly forecaster (XGBoost) paired
   with a retrieval-augmented diagnostic assistant that answers technician questions
   grounded in fault reference documentation, with a guardrail that blocks ungrounded
   repair instructions.
2. **Compliance layer** - every query, retrieval, and generation step is logged with full
   provenance, each logical AI component is classified against EU AI Act risk tiers, and
   audit-ready reports can be generated on demand. Logs are stored in a self-hosted
   Postgres instance, not a vendor's cloud.

## Why both in one repo

A PII-stripping filter and a provenance-logging layer aren't two separate compliance
features, they're the same discipline applied to input and output. Building them once
and wiring both the forecaster and the RAG assistant through them is both the efficient
engineering choice and the more honest demonstration of what "defensible AI deployment"
actually requires in practice, not a bolted-on policy document.

## Architecture

```
FastAPI service
  |-- compliance/audit layer  (logs everything, tags EU AI Act risk tier)
  |-- guardrail layer         (faithfulness scoring, escalate-or-answer gate)
  |-- rag layer               (pgvector hybrid search + LangGraph agent)
  |-- forecasting layer       (XGBoost degradation/anomaly model, MLflow tracked)
  |-- pii layer               (strips VIN/GPS/driver-ID before anything is stored)
  |-- data layer              (NASA Li-ion battery dataset, public OBD-II DTC reference docs,
                                synthetic telemetry with injected PII for testing the filter)
```

## Data sources

- **NASA Li-ion battery degradation dataset** (B0005/B0006/B0007/B0018) - the standard
  public benchmark for degradation forecasting.
- **Public OBD-II diagnostic trouble code (DTC) reference documentation** - a legitimate,
  openly available stand-in for proprietary OEM fault manuals. The RAG layer doesn't care
  what text it indexes, so swapping in real manuals later is a data change, not a
  redesign.
- **Synthetic telemetry** with deliberately injected VIN/GPS/driver-ID fields, used to
  prove the PII filter actually catches and strips sensitive data, since the public
  datasets above contain none to test against.

## Status

- [x] Phase 0: project setup
- [x] Phase 1: data layer + PII filter
- [x] Phase 2: forecasting layer
- [x] Phase 3: RAG layer
- [x] Phase 4: guardrail/decision layer
- [x] Phase 5: compliance/audit layer
- [x] Phase 6: service layer + packaging
- [ ] Phase 7: documentation and framing

## Running locally

```bash
docker compose up -d        # starts Postgres (pgvector) + MLflow
cp .env.example .env         # already done if you're reading this from the repo
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt

# NASA battery data and DTC reference docs (gitignored, fetch/build first)
./scripts/download_nasa_battery_data.sh
python scripts/extract_battery_features.py
python scripts/seed_dtc_reference.py
python scripts/embed_dtc_reference.py   # requires Ollama for the generation step, see below

# Ollama, for local LLM generation (RAG layer)
ollama serve                 # if not already running as a service
ollama pull llama3.2         # one-time, ~2GB

# Windows only: mlflow prints unicode in run-completion messages that this
# platform's default console codec can't encode. The crash happens after
# the run already logged successfully, cosmetic, but avoid it with:
export PYTHONIOENCODING=utf-8   # or set it before any script that uses mlflow

# Compliance layer (Phase 5)
python -m src.compliance.risk_registry   # populate ai_system_registry (idempotent)
python -m src.compliance.report          # print a full audit-ready compliance report

# Service layer (Phase 6), running directly against the host (not the api container)
uvicorn src.api.main:app --reload --port 8000
```

### Running the full stack in Docker (Phase 6)

```bash
ollama serve                 # Ollama stays on the host, GPU access, not containerized
ollama pull llama3.2

docker compose up -d --build   # starts Postgres, MLflow, and the api service together
curl http://localhost:8000/health
```

The `api` service reaches Postgres and MLflow by their compose service names
(`db`, `mlflow`) over the internal Docker network, and reaches Ollama on the
host via `host.docker.internal`, see the `environment:` block for the `api`
service in `docker-compose.yml` and the `OLLAMA_BASE_URL` handling in
`src/rag/agent.py` / `src/guardrail/faithfulness.py`. The forecasting model
is loaded from MLflow's tracking server at startup (`src/forecasting/model.py`'s
`load_latest_model()`), not a separate file baked into the image, so training
at least once (`python -m src.forecasting.model`) before starting the api
service is required.

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness check |
| `/diagnose` | POST | `{"question": "..."}` -> guardrailed, audit-logged diagnostic answer |
| `/forecast` | POST | battery telemetry features -> predicted capacity fade % |
| `/compliance/report` | GET | full audit-ready markdown compliance report |
| `/compliance/registry` | GET | EU AI Act risk classification for each component |

## Known environment gotchas

Real problems hit during development, documented here rather than left as
tribal knowledge in commit history:

- **spaCy on Windows**: every model, even a blank pipeline, throws
  `OverflowError: Python int too large to convert to C ssize_t` in the
  compiled tokenizer (broken Cython wheel ABI for this Python/Windows
  combo). Worked around by using Presidio's `PatternRecognizer`/
  `AnonymizerEngine` directly instead of its spaCy-backed `AnalyzerEngine`,
  see `src/pii/filter.py`.
- **torch native-library conflicts**: `sentence-transformers`/`torch`
  (pulled in transitively by `presidio_analyzer`) fail to initialize
  (`c10.dll`, `WinError 1114`) alongside xgboost/scipy in the same process.
  Avoided entirely by using `fastembed` (ONNX Runtime) for embeddings
  instead, see `src/rag/retrieval.py`.
- **numpy/protobuf three-way conflict**: `onnx` (via fastembed) needs
  numpy>=2.0 and protobuf>=6.31.1; older spacy/thinc needed numpy<2.0;
  `mlflow-skinny` 2.17.0 needed protobuf<6. Resolved by upgrading to
  `spacy>=3.8` and `mlflow==2.22.5`, and removing `langchain`/
  `langchain-community` entirely, they were unused dead weight and the
  actual source of the numpy<2.0 pin.
- **Windows console encoding**: mlflow prints a unicode emoji in
  run-completion messages that this platform's default cp1252 stdout
  codec can't encode, crashing cosmetically after the run already logged
  successfully. Fix: `export PYTHONIOENCODING=utf-8` before running
  anything that uses mlflow.
- **MLflow artifact storage resolves locally per-client, not through the
  server, unless explicitly proxied**: `mlflow server --default-artifact-root
  /mlflow/mlruns` (a bare local path) makes every client, training script,
  API service, whichever machine or container it runs on, resolve model
  artifacts against its OWN local filesystem rather than fetching them over
  HTTP. This "worked" during Phase 2 because the training script happened
  to run on the same host that then also served the API, each silently
  recreating the same literal absolute path on its own disk. It broke the
  moment Phase 6's containerized `api` service, a different filesystem
  entirely, tried to load the same model and got
  `OSError: No such file or directory`. Fixed by setting
  `--default-artifact-root mlflow-artifacts:/ --artifacts-destination
  /mlflow/mlruns`, the proxy scheme that makes every client fetch artifact
  bytes over HTTP from the tracking server regardless of where it runs, see
  `docker-compose.yml`'s mlflow service. This only affects experiments
  created after the change, existing ones keep their original (broken)
  artifact location, which is why the stale `battery_degradation_forecaster`
  experiment was deleted and retrained rather than patched in place.

## Compliance layer

`src/compliance/audit_log.py` writes an append-only, hash-chained log of every
query/retrieval/generation/guardrail step (`AuditLogger`), and `verify_chain()`
walks the whole table confirming no row was altered after the fact, each row's
hash commits to its own content plus the previous row's hash, so tampering with
any row, or deleting one from the middle, breaks every hash after it. This was
adversarially tested, not just run once: rows were tampered with directly via
SQL and confirmed caught by `verify_chain()`, see `tests/test_compliance.py`.

`src/compliance/pipeline.py`'s `AuditedDiagnosticPipeline` wraps the Phase 4
guardrail pipeline and logs the full provenance trail for every interaction.
`src/compliance/risk_registry.py` records each AI component's EU AI Act risk
tier with its actual Annex III reasoning, not just a label. `src/compliance/
report.py` turns both into a single audit-ready markdown report.
`src/compliance/retention.py` reports rows past the Article 12 minimum
retention window, it deliberately does not auto-delete anything, see its
module docstring for why that would conflict with the hash chain's
tamper-evidence guarantee.
