#!/usr/bin/env python3
"""Apply regional scaling to national yields using suitability + climate adjustments.

Usage:
  python scripts/apply_regional_scaling.py \
      --input project_data/processed_data/master_data_fnn.csv \
      --output project_data/processed_data/master_data_fnn_scaled.csv

This script:
 - loads `config/crop_zone_suitability.json` (or the specialized file if present)
 - reads the master input CSV
 - computes zone averages for temperature and rainfall
 - applies scaling: final = national_yield * (0.7*suitability + 0.3*climate_factor) * random(0.95,1.05)
 - saves scaled output and preserves original yields in `Yield_kg_per_ha_original`
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / 'project_data' / 'processed_data' / 'master_data_fnn.csv'
DEFAULT_OUT = ROOT / 'project_data' / 'processed_data' / 'master_data_fnn_scaled.csv'
CFG = ROOT / 'config' / 'crop_zone_suitability.json'
CFG5 = ROOT / 'config' / 'crop_zone_suitability_5crops.json'


def load_config():
    if CFG.exists():
        path = CFG
    elif CFG5.exists():
        path = CFG5
    else:
        raise FileNotFoundError('No crop suitability config found.')
    with path.open() as f:
        return json.load(f)


def compute_climate_factor(row, zone_avgs):
    # climate_factor starts at 1.0; apply simple temperature & rainfall rules
    cz = row['Geopolitical_Zone'] if 'Geopolitical_Zone' in row else row.get('Zone', None)
    if cz not in zone_avgs:
        return 1.0
    avg_temp, avg_rain = zone_avgs[cz]
    # Avoid division by zero
    if avg_temp == 0:
        temp_dev = 0.0
    else:
        temp_dev = (row.get('Temperature_C', avg_temp) - avg_temp) / avg_temp
    if avg_rain == 0:
        rain_dev = 0.0
    else:
        rain_dev = (row.get('Rainfall_mm', avg_rain) - avg_rain) / avg_rain

    climate_factor = 1.0
    climate_factor *= (1.0 - 0.1 * min(abs(temp_dev), 1.0))
    if rain_dev > 0:
        climate_factor *= (1.0 + 0.15 * min(rain_dev, 0.5))
    else:
        climate_factor *= (1.0 + 0.2 * max(rain_dev, -0.5))
    return float(climate_factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(DEFAULT_IN))
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    cfg = load_config()
    suitability = cfg.get('suitability_factors', {})

    df = pd.read_csv(args.input)

    # normalize possible column names
    if 'Geopolitical_Zone' not in df.columns and 'Zone' in df.columns:
        df = df.rename(columns={'Zone': 'Geopolitical_Zone'})

    # compute zone averages for climate
    zone_avgs = {}
    grp = df.groupby('Geopolitical_Zone')
    for z, g in grp:
        avg_temp = g['Temperature_C'].mean() if 'Temperature_C' in g else 0.0
        avg_rain = g['Rainfall_mm'].mean() if 'Rainfall_mm' in g else 0.0
        zone_avgs[z] = (avg_temp, avg_rain)

    # preserve original
    if 'Yield_kg_per_ha_original' not in df.columns:
        df['Yield_kg_per_ha_original'] = df['Yield_kg_per_ha']

    scaled = []
    random.seed(42)
    for _, row in df.iterrows():
        crop = row['Crop']
        zone = row['Geopolitical_Zone']
        national_yield = row['Yield_kg_per_ha']
        suit = 1.0
        try:
            suit = suitability.get(crop, {}).get(zone, 1.0)
        except Exception:
            suit = 1.0

        climate_factor = compute_climate_factor(row, zone_avgs)
        base = 0.7 * suit + 0.3 * climate_factor
        rnd = random.uniform(0.95, 1.05)
        final = national_yield * base * rnd
        scaled.append(final)

    df['Yield_kg_per_ha_scaled'] = np.array(scaled)
    args_out = Path(args.output)
    args_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args_out, index=False)
    print(f'Scaled data written to {args_out}')


if __name__ == '__main__':
    main()
