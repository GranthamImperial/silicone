#!/bin/bash
SR15_CSV="sr15_scenarios.csv"
OUTPUT_DIR="../output-examples"
if [ -f "$SR15_CSV" ]; then
    echo "${SR15_CSV} exists"
else
    echo "Downloading data to ${SR15_CSV}"
    python download_sr15_emissions.py ${SR15_CSV}
fi

mkdir -p "${OUTPUT_DIR}"
silicone-explore-quantiles "${SR15_CSV}" --output-dir "${OUTPUT_DIR}" --years "2010, 2015, 2020, 2030, 2050, 2100" --quantiles "0.0001, 0.05, 0.17, 0.33, 0.5, 0.67, 0.83, 0.95, 0.9999" --quantile-boxes 30 --quantile-decay-factor 0.7 --no-model-colours --legend-fraction 0.65
