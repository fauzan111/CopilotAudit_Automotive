"""Seed a curated set of standard OBD-II Diagnostic Trouble Codes (DTCs).

These are SAE J2012 standardized "generic" powertrain codes, public domain
by design (every OBD-II compliant vehicle sold since 1996 uses them), used
here as an honest stand-in for proprietary OEM repair manuals. The RAG layer
indexes whatever text it's given, so swapping this for real manuals later is
a data-ingestion change, not an architecture change.

Output: data/raw/dtc_reference.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "dtc_reference.jsonl"

# A representative sample of standard generic powertrain DTCs, focused on
# codes relevant to EV/hybrid battery and charging systems plus common
# general powertrain faults, since that pairs with the battery-degradation
# forecaster in Phase 2.
DTC_RECORDS = [
    {
        "code": "P0A80",
        "title": "Replace Hybrid Battery Pack",
        "description": (
            "Indicates the hybrid/EV traction battery pack has degraded below the "
            "manufacturer's usable capacity threshold and requires replacement. "
            "Triggered when the battery management system's state-of-health "
            "estimate falls under a calibrated limit across multiple drive cycles."
        ),
        "likely_causes": [
            "Cell-level capacity fade from age or cycle count",
            "Thermal stress from repeated fast-charging in hot climates",
            "One or more failed cell groups within the pack",
        ],
        "typical_fix": (
            "Confirm with a full battery health scan before replacement; partial "
            "pack repair is possible on some platforms if failure is isolated to "
            "a single module."
        ),
    },
    {
        "code": "P0A7F",
        "title": "Hybrid/EV Battery Performance",
        "description": (
            "Battery pack is operating but not meeting expected performance "
            "characteristics, such as reduced range or power delivery, without "
            "yet crossing the full replacement threshold."
        ),
        "likely_causes": [
            "Early-stage capacity fade",
            "Cold-weather performance derating misread as a fault",
            "Cooling system underperformance causing thermal throttling",
        ],
        "typical_fix": (
            "Check battery coolant levels and thermal management system first; "
            "re-evaluate after a full charge cycle in moderate ambient temperature."
        ),
    },
    {
        "code": "P0AA6",
        "title": "Hybrid Battery Voltage System Isolation Fault",
        "description": (
            "High-voltage isolation monitor has detected unexpected resistance "
            "between the battery pack's high-voltage circuit and vehicle chassis "
            "ground, a safety-critical fault."
        ),
        "likely_causes": [
            "Moisture intrusion into the high-voltage connector or battery enclosure",
            "Damaged high-voltage cable insulation",
            "Degraded connector seals after age or a prior collision repair",
        ],
        "typical_fix": (
            "Do not operate the vehicle until isolation resistance is confirmed "
            "safe; inspect all high-voltage connectors and enclosure seals before "
            "clearing the code."
        ),
    },
    {
        "code": "P0562",
        "title": "System Voltage Low",
        "description": (
            "The 12V auxiliary system voltage has dropped below the expected "
            "operating range, which can affect control module stability even on "
            "EV/hybrid platforms with a separate high-voltage pack."
        ),
        "likely_causes": [
            "Weak or failing 12V auxiliary battery",
            "DC-DC converter underperformance",
            "Parasitic draw from an accessory left active",
        ],
        "typical_fix": (
            "Load-test the 12V battery and verify DC-DC converter output voltage "
            "under load before replacing the battery."
        ),
    },
    {
        "code": "P0A1F",
        "title": "Charge Depleting Mode Malfunction",
        "description": (
            "Vehicle failed to properly transition into or maintain charge-depleting "
            "(electric-only) driving mode, commonly seen on plug-in hybrids."
        ),
        "likely_causes": [
            "Battery state-of-charge sensor miscalibration",
            "Software logic fault in the hybrid control unit",
            "Battery temperature outside the allowed range for EV-only mode",
        ],
        "typical_fix": (
            "Check for pending software updates to the hybrid control unit; "
            "verify battery temperature sensor readings against a second source."
        ),
    },
    {
        "code": "P0D54",
        "title": "Charger AC Voltage Low",
        "description": (
            "Onboard charger detected AC input voltage below the expected range "
            "during a charging session."
        ),
        "likely_causes": [
            "Weak or undersized external charging circuit",
            "Faulty charging cable or EVSE unit",
            "Voltage drop from a long extension or poor-quality outlet wiring",
        ],
        "typical_fix": (
            "Test the charging circuit with a known-good EVSE and a dedicated "
            "outlet before suspecting the onboard charger itself."
        ),
    },
    {
        "code": "P0300",
        "title": "Random/Multiple Cylinder Misfire Detected",
        "description": (
            "Engine control module detected misfires across multiple cylinders "
            "without a consistent single-cylinder pattern. Relevant on hybrid "
            "platforms with a combustion engine component."
        ),
        "likely_causes": [
            "Worn spark plugs or ignition coils",
            "Vacuum leak",
            "Low fuel pressure",
        ],
        "typical_fix": (
            "Check spark plug condition and ignition coil resistance first, "
            "they're the most common cause of random multi-cylinder misfires."
        ),
    },
    {
        "code": "P0171",
        "title": "System Too Lean (Bank 1)",
        "description": (
            "Engine control module detected more oxygen in the exhaust than "
            "expected for the current fuel trim, indicating a lean air-fuel mixture."
        ),
        "likely_causes": [
            "Vacuum leak in intake system",
            "Failing mass airflow sensor",
            "Weak fuel pump or clogged fuel filter",
        ],
        "typical_fix": (
            "Smoke-test the intake for vacuum leaks before replacing any sensors.",
        ),
    },
    {
        "code": "U0100",
        "title": "Lost Communication with ECM/PCM",
        "description": (
            "A module on the vehicle's communication bus (CAN-bus) lost contact "
            "with the engine/powertrain control module."
        ),
        "likely_causes": [
            "CAN-bus wiring fault or connector corrosion",
            "ECM/PCM power or ground fault",
            "Module failure",
        ],
        "typical_fix": (
            "Check battery voltage and ECM/PCM ground connections before "
            "condemning the module itself; intermittent bus faults are often "
            "wiring-related."
        ),
    },
    {
        "code": "P0A0F",
        "title": "Drive Motor A Performance",
        "description": (
            "Electric drive motor is not delivering the torque or RPM response "
            "expected for the commanded input."
        ),
        "likely_causes": [
            "Motor resolver/position sensor fault",
            "Inverter underperformance",
            "Internal motor winding degradation",
        ],
        "typical_fix": (
            "Scan inverter diagnostic data for resolver alignment errors before "
            "assuming motor hardware failure."
        ),
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in DTC_RECORDS:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(DTC_RECORDS)} DTC reference records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
