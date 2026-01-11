import json
import pandas as pd

CSV = 'adm_crop_production_NG.csv'
REGIONS_FILE = 'config/regions_and_state.json'
OUT_SUMMARY = 'region_crop_coverage.csv'
OUT_FULL = 'region_crops_100pct.csv'

print('Loading data...')
df = pd.read_csv(CSV)
with open(REGIONS_FILE, 'r', encoding='utf-8') as f:
    regions = json.load(f)

# normalize
df['product_clean'] = df['product'].astype(str).str.strip().str.lower()
df['admin_1_clean'] = df['admin_1'].astype(str).str.strip()

def region_states_lookup():
    lookup = {}
    for r, states in regions.items():
        lookup[r] = set([s.strip() for s in states])
    return lookup

region_lookup = region_states_lookup()

rows = []
full_rows = []
all_products = df['product_clean'].dropna().unique()
for prod in sorted(all_products):
    prod_rows = df[df['product_clean'] == prod]
    for region, states in region_lookup.items():
        sub = prod_rows[prod_rows['admin_1_clean'].isin(states)]
        if sub.empty:
            rows.append({'product': prod, 'region': region, 'states_in_region': len(states), 'present_states': 0, 'min_year': None, 'max_year': None, 'years_count': 0, 'expected': 0, 'complete': 0, 'pct': 0.0})
            continue
        years = sorted(sub['planting_year'].dropna().unique().astype(int))
        miny, maxy = int(min(years)), int(max(years))
        year_list = list(range(miny, maxy+1))
        n_states_present = sub['admin_1_clean'].dropna().unique().tolist()
        expected = len(states) * len(year_list)
        pivot = sub.pivot_table(index=['admin_1_clean', 'planting_year'], columns='indicator', values='value', aggfunc='first')
        for needed in ['area','production','yield']:
            if needed not in pivot.columns:
                pivot[needed] = pd.NA
        complete_mask = pivot[['area','production','yield']].notnull().all(axis=1)
        complete = int(complete_mask.sum())
        pct = 100.0 * complete / expected if expected else 0.0
        rows.append({'product': prod, 'region': region, 'states_in_region': len(states), 'present_states': len(n_states_present), 'min_year': miny, 'max_year': maxy, 'years_count': len(year_list), 'expected': expected, 'complete': complete, 'pct': round(pct,2)})

# Save summary
out = pd.DataFrame(rows)
out.to_csv(OUT_SUMMARY, index=False)
print('Wrote', OUT_SUMMARY)

# For each region, list products with 100% (pct == 100)
full = out[out['pct'] == 100.0]
# pivot to one row per product with regions satisfied
if not full.empty:
    grouped = full.groupby('product')['region'].apply(list).reset_index()
    grouped.to_csv(OUT_FULL, index=False)
    print('Wrote', OUT_FULL)
else:
    # write empty with header
    pd.DataFrame(columns=['product','region']).to_csv(OUT_FULL, index=False)
    print('No product achieved 100% in any region; wrote empty', OUT_FULL)

print('\nTop rows of summary:')
print(out.head(12).to_string(index=False))
