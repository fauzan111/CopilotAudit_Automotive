"""Generate synthetic vehicle telemetry with deliberately injected PII.

Public datasets (NASA battery data, OBD-II DTC references) contain no
sensitive fields, so there's nothing to prove the PII filter against. This
script fabricates realistic-looking technician log entries and raw telemetry
records with embedded VINs, GPS coordinates, driver IDs, names, and emails,
entirely synthetic, no real vehicles or people involved, so Phase 1's filter
has real input to be tested and demonstrated against.

Output: data/synthetic/telemetry_log.jsonl
"""

from __future__ import annotations

import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "telemetry_log.jsonl"

REGIMES = ["hot_highway", "cold_urban", "mild_mixed", "hot_urban", "cold_highway"]

FIRST_NAMES = ["Marco", "Giulia", "Alessandro", "Francesca", "Luca", "Chiara"]
LAST_NAMES = ["Rossi", "Bianchi", "Romano", "Ferrari", "Russo", "Colombo"]

# Turin-area coordinates, since that's where AI4I/CRF are based, kept
# illustrative only.
TURIN_LAT_RANGE = (45.02, 45.12)
TURIN_LON_RANGE = (7.60, 7.75)


def _random_vin() -> str:
    chars = [c for c in string.ascii_uppercase + string.digits if c not in "IOQ"]
    return "".join(random.choices(chars, k=17))


def _random_driver_id() -> str:
    return f"DRV-{random.randint(1000, 99999)}"


def _random_gps() -> str:
    lat = round(random.uniform(*TURIN_LAT_RANGE), 5)
    lon = round(random.uniform(*TURIN_LON_RANGE), 5)
    return f"{lat}, {lon}"


def _random_name_email() -> tuple[str, str]:
    first, last = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
    email = f"{first.lower()}.{last.lower()}@fleetco.it"
    return f"{first} {last}", email


def generate_record(ts: datetime) -> dict:
    vin = _random_vin()
    driver_id = _random_driver_id()
    gps = _random_gps()
    name, email = _random_name_email()
    regime = random.choice(REGIMES)

    note = (
        f"Technician {name} ({email}) logged a check for vehicle VIN {vin}, "
        f"driver {driver_id}, at position {gps}. "
        f"Battery voltage slightly low during {regime.replace('_', ' ')} conditions; "
        f"recommend re-check at next service interval."
    )

    return {
        "raw_log_text": note,
        "vin": vin,
        "driver_id": driver_id,
        "gps": gps,
        "regime_cluster": regime,
        "reading_ts": ts.isoformat(),
        "sensor_payload": {
            "battery_voltage": round(random.uniform(11.5, 12.9), 2),
            "battery_temp_c": round(random.uniform(-5, 45), 1),
            "state_of_charge_pct": round(random.uniform(15, 100), 1),
            "cycle_count": random.randint(50, 1200),
        },
    }


def main(n_records: int = 500) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(n_records):
            ts = start + timedelta(hours=i * 3)
            record = generate_record(ts)
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {n_records} synthetic records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
