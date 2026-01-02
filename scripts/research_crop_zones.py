"""Generate or inspect crop-zone suitability config based on project metadata.

Usage:
  python scripts/research_crop_zones.py

This script checks for an existing suitability config (`config/crop_zone_suitability_5crops.json`) and
copies it to `config/crop_zone_suitability.json` for general use. If no config exists it creates a
template using crops/zones from `project_data/processed_data/preprocessing_metadata.json`.
"""
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'project_data' / 'processed_data' / 'preprocessing_metadata.json'
EXISTING = ROOT / 'config' / 'crop_zone_suitability_5crops.json'
OUT = ROOT / 'config' / 'crop_zone_suitability.json'


def load_meta():
    if not META.exists():
        print(f"Metadata not found: {META}")
        sys.exit(1)
    with META.open() as f:
        return json.load(f)


def create_template(crops, zones):
    data = {
        "metadata": {
            "description": "Template suitability factors",
            "crops": crops,
            "zones": zones,
            "scale": "0.0 (not grown) to 1.5 (optimal conditions)"
        },
        "suitability_factors": {},
        "validation": {},
    }
    for c in crops:
        data['suitability_factors'][c] = {z: 1.0 for z in zones}
        data['suitability_factors'][c]['rationale'] = 'Auto-generated template; please update.'
    return data


def main():
    meta = load_meta()
    crops = meta.get('crops', [])
    zones = meta.get('zones', [])

    if EXISTING.exists():
        print(f"Found existing config: {EXISTING}. Copying to {OUT}")
        with EXISTING.open('r', encoding='utf-8') as fr, OUT.open('w', encoding='utf-8') as fw:
            fw.write(fr.read())
        print('Done.')
        return

    print('No existing specialized config found; creating template from metadata...')
    tpl = create_template(crops, zones)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(tpl, f, indent=2)
    print(f'Template written to {OUT}')


if __name__ == '__main__':
    main()
