"""FastAPI service layer: the single HTTP entry point wrapping every other
layer (compliance/audit -> guardrail -> rag -> forecasting), matching the
architecture diagram in the README.

Endpoints hit real external dependencies (Postgres, Ollama, MLflow), so
failures there are a genuine system boundary, not a hypothetical: this is
the one place in the codebase that catches those failures and turns them
into a 503 instead of letting a raw connection error reach the caller.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.compliance.pipeline import AuditedDiagnosticPipeline
from src.compliance.report import fetch_registry, generate_report
from src.forecasting.model import FEATURE_COLUMNS, load_latest_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = AuditedDiagnosticPipeline()
    try:
        app.state.forecast_model = load_latest_model()
    except RuntimeError:
        # Forecasting model hasn't been trained in this environment yet;
        # /forecast reports 503 rather than the app failing to start.
        app.state.forecast_model = None
    yield


app = FastAPI(title="CopilotAudit_Automotive", lifespan=lifespan)


class DiagnoseRequest(BaseModel):
    question: str


class DiagnoseResponse(BaseModel):
    question: str
    answer: str
    decision: str
    retrieved_doc_refs: list[str]
    grounded: bool | None


class ForecastRequest(BaseModel):
    discharge_cycle_index: int
    mean_voltage_v: float
    min_voltage_v: float
    mean_current_a: float
    mean_temperature_c: float
    max_temperature_c: float
    discharge_duration_s: float


class ForecastResponse(BaseModel):
    capacity_fade_pct: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    try:
        result = app.state.pipeline.run(request.question)
    except Exception as exc:  # Ollama/DB are both real external dependencies that can be down
        raise HTTPException(status_code=503, detail=f"diagnostic pipeline unavailable: {exc}") from exc

    return DiagnoseResponse(
        question=result.question,
        answer=result.answer,
        decision=result.decision,
        retrieved_doc_refs=[d.doc_ref for d in result.retrieved_docs],
        grounded=result.faithfulness.grounded if result.faithfulness else None,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    if app.state.forecast_model is None:
        raise HTTPException(status_code=503, detail="forecasting model not trained yet, run: python -m src.forecasting.model")

    row = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    prediction = float(app.state.forecast_model.predict(row)[0])
    return ForecastResponse(capacity_fade_pct=prediction)


@app.get("/compliance/report")
def compliance_report() -> dict:
    try:
        return {"report_markdown": generate_report()}
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc


@app.get("/compliance/registry")
def compliance_registry() -> list[dict]:
    try:
        return fetch_registry()
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
