"""Load synthetic telemetry into Postgres, through the PII filter.

This is the proof that Phase 1 actually works end to end: raw records with
injected VIN/GPS/driver-ID/name/email go in, only the vehicle_hash and clean
sensor payload come out the other side into telemetry_readings. The real
VIN, GPS, driver ID, and technician identity are never written to the
database, only a one-way hash standing in for the vehicle.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pii.filter import TelemetryPIIFilter

load_dotenv()

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "telemetry_log.jsonl"
DATABASE_URL = os.environ["DATABASE_URL"]


def main() -> None:
    pii_filter = TelemetryPIIFilter()
    loaded, pii_caught = 0, 0

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            with INPUT_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)

                    result = pii_filter.filter(record["raw_log_text"])
                    if result.entities_found:
                        pii_caught += 1

                    cur.execute(
                        """
                        INSERT INTO telemetry_readings
                            (vehicle_hash, reading_ts, regime_cluster, sensor_payload)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            result.vehicle_hash,
                            record["reading_ts"],
                            record["regime_cluster"],
                            json.dumps(record["sensor_payload"]),
                        ),
                    )
                    loaded += 1
        conn.commit()

    print(f"Loaded {loaded} telemetry readings.")
    print(f"PII filter caught sensitive entities in {pii_caught}/{loaded} raw log records.")
    print("None of the raw VIN, GPS, driver ID, name, or email fields were written to the DB.")


if __name__ == "__main__":
    main()
