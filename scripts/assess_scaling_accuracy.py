#!/usr/bin/env python3
"""Assess the effects of regional scaling: directional and magnitude checks.

Usage:
  python scripts/assess_scaling_accuracy.py --input project_data/processed_data/master_data_fnn_scaled.csv

Creates a small summary printed to stdout and saved as `project_data/processed_data/scaling_accuracy_summary.json`.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / 'project_data' / 'processed_data' / 'master_data_fnn_scaled.csv'
OUT = ROOT / 'project_data' / 'processed_data' / 'scaling_accuracy_summary.json'


def directional_accuracy(df):
    # For each crop, find the zone with highest mean before and after scaling
    res = {}
    for crop, g in df.groupby('Crop'):
        before = g.groupby('Geopolitical_Zone')['Yield_kg_per_ha_original'].mean()
        after = g.groupby('Geopolitical_Zone')['Yield_kg_per_ha_scaled'].mean()
        top_before = before.idxmax()
        top_after = after.idxmax()
        res[crop] = {
            'top_before': top_before,
            'top_after': top_after,
            'match': top_before == top_after
        }
    return res


def magnitude_stats(df):
    stats = {}
    for crop, g in df.groupby('Crop'):
        b_mean = g['Yield_kg_per_ha_original'].mean()
        a_mean = g['Yield_kg_per_ha_scaled'].mean()
        rmse = np.sqrt(((g['Yield_kg_per_ha_original'] - g['Yield_kg_per_ha_scaled']) ** 2).mean())
        stats[crop] = {
            'before_mean': float(b_mean),
            'after_mean': float(a_mean),
            'rmse': float(rmse),
            'std_before': float(g['Yield_kg_per_ha_original'].std()),
            'std_after': float(g['Yield_kg_per_ha_scaled'].std())
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(DEFAULT_IN))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if 'Yield_kg_per_ha_original' not in df.columns:
        raise RuntimeError('Input file must contain Yield_kg_per_ha_original column')

    dir_acc = directional_accuracy(df)
    mags = magnitude_stats(df)

    summary = {
        'directional': dir_acc,
        'magnitude': mags
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Scaling assessment written to', OUT)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
