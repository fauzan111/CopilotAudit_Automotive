"""Extract a tidy degradation feature table from the raw NASA .mat files.

NASA's PCoE Li-ion battery dataset (B0005, B0006, B0007, B0018) stores each
battery as a nested MATLAB struct: a top-level cycle array mixing charge,
discharge, and impedance operations, with discharge cycles carrying the
Capacity (Ahr) field, the degradation signal every published analysis of
this dataset forecasts.

This script walks the raw structs, keeps only discharge cycles, and engineers
one row per discharge cycle: cycle index, ambient temperature, summary stats
over the voltage/current/temperature traces, discharge duration, the measured
capacity, and capacity fade relative to that battery's first discharge. The
regime_cluster label mirrors the synthetic telemetry's labeling: buckets on
ambient temperature, matching the hot/cold urban/highway framing used
elsewhere in this project.

Output: data/processed/battery_degradation.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nasa_battery"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "battery_degradation.csv"

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]

# NASA's EOL criterion for this dataset: 30% fade from the ~2 Ah rated
# capacity (down to 1.4 Ah). Used downstream for RUL labeling in Phase 2's
# model, kept here as a documented constant rather than a magic number.
EOL_CAPACITY_AH = 1.4


def _temperature_regime(ambient_c: float) -> str:
    """Bucket ambient temperature the same way the synthetic telemetry
    generator buckets driving regimes, so both data sources share a
    vocabulary downstream.
    """
    if ambient_c >= 35:
        return "hot"
    if ambient_c <= 10:
        return "cold"
    return "mild"


def extract_battery(battery_id: str) -> pd.DataFrame:
    mat = sio.loadmat(RAW_DIR / f"{battery_id}.mat")
    cycle_struct = mat[battery_id][0, 0]["cycle"]

    rows = []
    discharge_index = 0
    first_capacity = None

    for i in range(cycle_struct.shape[1]):
        entry = cycle_struct[0, i]
        if entry["type"][0] != "discharge":
            continue

        data = entry["data"]
        voltage = data["Voltage_measured"][0, 0].flatten()
        current = data["Current_measured"][0, 0].flatten()
        temperature = data["Temperature_measured"][0, 0].flatten()
        time = data["Time"][0, 0].flatten()
        capacity = float(data["Capacity"][0, 0][0, 0])
        ambient_temp = float(entry["ambient_temperature"][0, 0])

        if first_capacity is None:
            first_capacity = capacity

        rows.append(
            {
                "battery_id": battery_id,
                "discharge_cycle_index": discharge_index,
                "ambient_temperature_c": ambient_temp,
                "regime_cluster": _temperature_regime(ambient_temp),
                "capacity_ah": capacity,
                "capacity_fade_pct": (first_capacity - capacity) / first_capacity * 100.0,
                "below_eol_threshold": capacity < EOL_CAPACITY_AH,
                "mean_voltage_v": float(np.mean(voltage)),
                "min_voltage_v": float(np.min(voltage)),
                "mean_current_a": float(np.mean(current)),
                "mean_temperature_c": float(np.mean(temperature)),
                "max_temperature_c": float(np.max(temperature)),
                "discharge_duration_s": float(np.max(time)),
            }
        )
        discharge_index += 1

    return pd.DataFrame(rows)


def main() -> None:
    frames = [extract_battery(bid) for bid in BATTERY_IDS]
    full = pd.concat(frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT_PATH, index=False)

    print(f"Extracted {len(full)} discharge-cycle records across {len(BATTERY_IDS)} batteries.")
    print(full.groupby("battery_id")["discharge_cycle_index"].max().add(1).rename("n_discharge_cycles"))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
