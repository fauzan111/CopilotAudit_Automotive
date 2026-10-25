-- CopilotAudit_Automotive: shared schema for RAG storage + EU AI Act audit logging
-- This single database serves both the diagnostic copilot (RAG) and the compliance layer,
-- by design: the audit log needs to reference the same chunks the RAG layer retrieves from.

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- RAG layer: fault/diagnostic reference documents
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,          -- e.g. 'obd2_dtc_reference'
    doc_ref         TEXT NOT NULL,          -- e.g. DTC code 'P0300'
    title           TEXT,
    content         TEXT NOT NULL,
    embedding       VECTOR(384),            -- adjust to match embedding model dim
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_content_fts_idx
    ON documents USING gin (to_tsvector('english', content));

-- ============================================================
-- Telemetry: anonymized vehicle sensor data (post-PII-filter only)
-- ============================================================
CREATE TABLE IF NOT EXISTS telemetry_readings (
    id                  BIGSERIAL PRIMARY KEY,
    vehicle_hash        TEXT NOT NULL,       -- irreversible hash, never the real VIN
    reading_ts          TIMESTAMPTZ NOT NULL,
    regime_cluster       TEXT,                -- e.g. 'hot_highway', 'cold_urban'
    sensor_payload       JSONB NOT NULL,       -- battery/CAN-bus features, no PII
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- Compliance layer: full provenance log, EU AI Act Article 12
-- Every query -> retrieval -> generation step gets a row here.
-- Retained 6+ months per Article 12 minimum.
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id                      BIGSERIAL PRIMARY KEY,
    interaction_id          UUID NOT NULL,
    step                    TEXT NOT NULL,     -- 'query' | 'retrieval' | 'generation' | 'guardrail'
    model_name              TEXT,
    input_summary           TEXT,
    retrieved_doc_ids       BIGINT[],
    output_summary          TEXT,
    confidence              NUMERIC,
    faithfulness_score      NUMERIC,
    decision                TEXT,              -- 'answered' | 'escalated' | 'blocked'
    risk_tier               TEXT,              -- EU AI Act: 'minimal' | 'limited' | 'high_risk'
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audit_log_interaction_idx ON audit_log (interaction_id);
CREATE INDEX IF NOT EXISTS audit_log_created_idx ON audit_log (created_at);

-- ============================================================
-- Risk classification registry: one row per logical AI system/component
-- (not per interaction) -- this is the "inventory" the EU AI Act readiness
-- research flagged most enterprises don't even have.
-- ============================================================
CREATE TABLE IF NOT EXISTS ai_system_registry (
    id                  BIGSERIAL PRIMARY KEY,
    component_name      TEXT NOT NULL UNIQUE,   -- e.g. 'battery_degradation_forecaster'
    description          TEXT,
    risk_tier            TEXT NOT NULL,          -- 'minimal' | 'limited' | 'high_risk' | 'prohibited'
    classification_basis TEXT,                   -- short justification, Annex III reference if applicable
    last_reviewed        TIMESTAMPTZ NOT NULL DEFAULT now()
);
