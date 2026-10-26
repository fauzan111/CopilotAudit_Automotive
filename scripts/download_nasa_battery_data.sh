#!/usr/bin/env bash
# Downloads the four NASA PCoE Li-ion battery .mat files used by
# extract_battery_features.py.
#
# NASA's own hosting (data.nasa.gov / c3.nasa.gov/dashlink) has the dataset
# listed but the direct download is currently unavailable ("Data is
# currently unavailable for download directly" as of the last check), so
# this pulls from a mirror repo that checked the original files straight
# into GitHub: anirudhkhatry/SOH-prediction-using-NASA-Dataset. Verified
# against NASA's documented struct schema (cycle/type/ambient_temperature/
# data with Voltage_measured, Current_measured, Capacity, etc.) before
# trusting it, file sizes and discharge-cycle counts match what's publicly
# known about this dataset (B0005/B0006/B0007 at 616 cycles/168 discharges,
# B0018 at 319 cycles/132 discharges).
#
# data/raw/ is gitignored, so this script is how a fresh clone gets the data.

set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw/nasa_battery"
mkdir -p "$TARGET_DIR"

BASE_URL="https://raw.githubusercontent.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset/master"

for battery in B0005 B0006 B0007 B0018; do
    echo "Downloading ${battery}.mat..."
    curl -sL -o "${TARGET_DIR}/${battery}.mat" "${BASE_URL}/${battery}.mat"
done

echo "Done. Files in ${TARGET_DIR}:"
ls -la "$TARGET_DIR"
