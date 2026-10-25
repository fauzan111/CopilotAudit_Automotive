"""Proves the PII filter actually strips what it claims to strip.

Run with: pytest tests/test_pii_filter.py -v
"""

from src.pii.filter import TelemetryPIIFilter


def test_strips_vin():
    f = TelemetryPIIFilter()
    text = "Vehicle VIN 1HGCM82633A004352 reported low battery voltage."
    result = f.filter(text)
    assert "1HGCM82633A004352" not in result.anonymized_text
    assert "VEHICLE_VIN" in result.entities_found
    assert result.vehicle_hash is not None


def test_same_vin_produces_same_hash():
    f = TelemetryPIIFilter()
    vin = "1HGCM82633A004352"
    r1 = f.filter(f"VIN {vin} check A")
    r2 = f.filter(f"VIN {vin} check B")
    assert r1.vehicle_hash == r2.vehicle_hash


def test_strips_gps_coordinates():
    f = TelemetryPIIFilter()
    text = "Last known position: 45.0703, 7.6869 near the service center."
    result = f.filter(text)
    assert "45.0703" not in result.anonymized_text
    assert "GPS_COORDINATES" in result.entities_found


def test_strips_driver_id():
    f = TelemetryPIIFilter()
    text = "Driver DRV-04812 flagged a dashboard warning light."
    result = f.filter(text)
    assert "DRV-04812" not in result.anonymized_text
    assert "DRIVER_ID" in result.entities_found


def test_strips_name_and_email():
    f = TelemetryPIIFilter()
    text = "Technician Marco Rossi (marco.rossi@fleetco.it) completed the inspection."
    result = f.filter(text)
    assert "Marco Rossi" not in result.anonymized_text
    assert "marco.rossi@fleetco.it" not in result.anonymized_text


def test_clean_text_passes_through_unchanged_entities():
    f = TelemetryPIIFilter()
    text = "Battery state of charge at 82 percent, no anomalies detected."
    result = f.filter(text)
    assert result.entities_found == []
    assert result.anonymized_text == text


def test_combined_record_strips_everything():
    f = TelemetryPIIFilter()
    text = (
        "Technician Giulia Bianchi (giulia.bianchi@fleetco.it) logged a check for "
        "vehicle VIN 1HGCM82633A004352, driver DRV-09911, at position 45.0703, 7.6869. "
        "Battery voltage slightly low during cold urban conditions."
    )
    result = f.filter(text)
    for leaked in [
        "Giulia Bianchi",
        "giulia.bianchi@fleetco.it",
        "1HGCM82633A004352",
        "DRV-09911",
        "45.0703, 7.6869",
    ]:
        assert leaked not in result.anonymized_text, f"leaked: {leaked}"
