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

Early scaffold. See the phase plan below.

- [ ] Phase 0: project setup
- [ ] Phase 1: data layer + PII filter
- [ ] Phase 2: forecasting layer
- [ ] Phase 3: RAG layer
- [ ] Phase 4: guardrail/decision layer
- [ ] Phase 5: compliance/audit layer
- [ ] Phase 6: service layer + packaging
- [ ] Phase 7: documentation and framing

## Running locally

```bash
docker compose up -d        # starts Postgres (pgvector) + MLflow
cp .env.example .env         # already done if you're reading this from the repo
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```
