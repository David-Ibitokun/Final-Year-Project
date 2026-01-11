import pandas as pd

CSV = 'adm_crop_production_NG.csv'
CROPS = ['cowpea','tomato','okra','cotton']
# map user terms to search tokens
SEARCH_TOKENS = {
    'cowpea': 'cowpea',
    'tomato': 'tomato',
    'okra': 'okra',
    'cotton': 'cotton'
}
OUT_CSV = 'coverage_selected_crops.csv'

print('Loading', CSV)
df = pd.read_csv(CSV)
# normalize product names for matching
df['product_clean'] = df['product'].astype(str).str.strip().str.lower()

rows = []
for key, token in SEARCH_TOKENS.items():
    mask = df['product_clean'].str.contains(token, na=False)
    sub = df[mask].copy()
    if sub.empty:
        rows.append({
            'query': key,
            'product_sample': '',
            'states': 0,
            'min_year': None,
            'max_year': None,
            'years_count': 0,
            'expected_combos': 0,
            'complete_combos': 0,
            'percent_complete': 0.0
        })
        continue

    sub_years = sub['planting_year'].dropna().unique().astype(int)
    miny = int(sub_years.min())
    maxy = int(sub_years.max())
    years = list(range(miny, maxy + 1))
    states = sub['admin_1'].dropna().unique().tolist()
    n_states = len(states)
    expected = n_states * len(years)

    pivot = sub.pivot_table(index=['admin_1', 'planting_year'], columns='indicator', values='value', aggfunc='first')
    needed = ['area', 'production', 'yield']
    # ensure columns exist
    for c in needed:
        if c not in pivot.columns:
            pivot[c] = pd.NA
    complete_mask = pivot[needed].notnull().all(axis=1)
    complete_count = int(complete_mask.sum())
    pct = 100.0 * complete_count / expected if expected else 0.0

    sample_product = sub['product'].mode().iat[0] if not sub['product'].mode().empty else sub['product'].iloc[0]

    rows.append({
        'query': key,
        'product_sample': sample_product,
        'states': n_states,
        'min_year': miny,
        'max_year': maxy,
        'years_count': len(years),
        'expected_combos': expected,
        'complete_combos': complete_count,
        'percent_complete': round(pct, 2)
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print('Wrote', OUT_CSV)
print(out.to_string(index=False))
