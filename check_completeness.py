import pandas as pd

fn = 'adm_crop_production_NG.csv'
df = pd.read_csv(fn)

crops = ['Rice','Maize','Cassava','Yams','Sorghum']
years = list(range(1999, 2024))

results = {}
all_missing_examples = []

for crop in crops:
    d = df[df['product'] == crop]
    states = sorted(d['admin_1'].dropna().unique())
    num_states = len(states)
    expected = num_states * len(years)

    # count (state,year) combos that have all three indicators
    grp = d.groupby(['admin_1','planting_year'])['indicator'].nunique().reset_index()
    grp['complete'] = grp['indicator'] >= 3
    present_complete = grp['complete'].sum()

    # find missing combos (either missing entirely or missing indicators)
    # build full grid of state x year
    import itertools
    full = pd.DataFrame([(s,y) for s,y in itertools.product(states, years)], columns=['admin_1','planting_year'])
    merged = full.merge(grp[['admin_1','planting_year','indicator','complete']], on=['admin_1','planting_year'], how='left')
    merged['indicator'] = merged['indicator'].fillna(0).astype(int)
    merged['complete'] = merged['complete'].fillna(False)
    missing = merged[~merged['complete']]
    missing_count = len(missing)

    # check for NaNs in numerical values for present combos
    # For combos present, check if any of area/production/yield are NaN
    combos = d.groupby(['admin_1','planting_year'])
    nan_issues = 0
    nan_examples = []
    for (s,y), g in combos:
        # need indicators present
        vals = g.set_index('indicator')['value'] if not g.empty else pd.Series()
        for ind in ['area','production','yield']:
            if ind not in vals.index or pd.isna(vals.get(ind, None)):
                nan_issues += 1
                if len(nan_examples) < 10:
                    nan_examples.append((crop,s,y,ind))
                break

    # record
    results[crop] = {
        'states_count': num_states,
        'expected_state_year_combos': expected,
        'complete_combos': int(present_complete),
        'percent_complete': round(100 * present_complete / expected, 2) if expected>0 else 0.0,
        'missing_combos': int(missing_count),
        'nan_issues': nan_issues,
        'missing_examples': missing.head(15).to_dict('records'),
        'nan_examples': nan_examples
    }
    # collect a few missing examples globally
    for r in missing.head(10).to_dict('records'):
        all_missing_examples.append((crop, r['admin_1'], int(r['planting_year']), int(r['indicator'])))

# Overall completeness across all 5 crops using union of states observed per crop
print('\nCOMPLETENESS REPORT FOR HARVESTSTAT-NG (per crop)')
for c,v in results.items():
    print(f"\nCrop: {c}")
    print(f" States with data: {v['states_count']}")
    print(f" Expected state-year combos: {v['expected_state_year_combos']}")
    print(f" Complete combos (all 3 indicators): {v['complete_combos']} ({v['percent_complete']}%)")
    print(f" Missing combos: {v['missing_combos']}")
    print(f" NaN issues in present combos: {v['nan_issues']}")
    if v['missing_examples']:
        ex = v['missing_examples'][:5]
        print(' Sample missing examples (up to 5):')
        for e in ex:
            print('  -', e)
    if v['nan_examples']:
        print(' Sample NaN issues (up to 5):')
        for ne in v['nan_examples']:
            print('  -', ne)

# Summary
total_expected = sum(v['expected_state_year_combos'] for v in results.values())
total_complete = sum(v['complete_combos'] for v in results.values())
print('\n\nSUMMARY ACROSS ALL 5 CROPS:')
print(f' Total expected combos: {total_expected}')
print(f' Total complete combos: {total_complete} ({round(100*total_complete/total_expected,2)}%)')

# Check for any rows with missing critical columns
critical_nans = df[df['product'].isin(crops)][['admin_1','planting_year','indicator','value']].isna().any(axis=1).sum()
print(f' Rows with any missing critical fields (admin_1/planting_year/indicator/value): {critical_nans}')

# Save a small report
import json
with open('harveststat_completeness_report.json','w') as f:
    json.dump({'results':results, 'summary':{'total_expected':total_expected,'total_complete':total_complete}}, f, indent=2)

print('\nDetailed JSON report saved to harveststat_completeness_report.json')
