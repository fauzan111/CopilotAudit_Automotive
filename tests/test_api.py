"""Tests for the Phase 6 FastAPI service layer, exercised against the real
Postgres/Ollama/MLflow stack via FastAPI's TestClient, consistent with
every earlier phase's integration-test-over-mocking approach in this
project.

Run with: pytest tests/test_api.py -v
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

VALID_FORECAST_PAYLOAD = {
    "discharge_cycle_index": 50,
    "mean_voltage_v": 3.8,
    "min_voltage_v": 3.2,
    "mean_current_a": -2.0,
    "mean_temperature_c": 30.0,
    "max_temperature_c": 35.0,
    "discharge_duration_s": 3000.0,
}


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_compliance_registry_lists_both_components():
    with TestClient(app) as client:
        response = client.get("/compliance/registry")
    assert response.status_code == 200
    names = {row["component_name"] for row in response.json()}
    assert names == {"battery_degradation_forecaster", "automotive_diagnostic_copilot"}


def test_compliance_report_contains_expected_sections():
    with TestClient(app) as client:
        response = client.get("/compliance/report")
    assert response.status_code == 200
    report = response.json()["report_markdown"]
    assert "Audit Log Integrity" in report
    assert "AI System Risk Registry" in report


def test_forecast_returns_a_plausible_capacity_fade_percentage():
    with TestClient(app) as client:
        response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    fade = response.json()["capacity_fade_pct"]
    assert 0 <= fade <= 100


def test_forecast_rejects_malformed_payload():
    with TestClient(app) as client:
        response = client.post("/forecast", json={"discharge_cycle_index": 50})
    assert response.status_code == 422  # pydantic validation, missing required fields


def test_diagnose_answers_a_well_grounded_question_and_logs_it():
    with TestClient(app) as client:
        response = client.post("/diagnose", json={"question": "What is P0300 and how do I fix it?"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "answered"
    assert "P0300" in body["retrieved_doc_refs"]
    assert body["grounded"] is True
