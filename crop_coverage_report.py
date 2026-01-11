import pandas as pd
import itertools

df = pd.read_csv('adm_crop_production_NG.csv')
products = sorted(df['product'].unique())
report = []
for p in products:
    d = df[df['product']==p].copy()
    states = sorted(d['admin_1'].dropna().unique())
    if len(d)==0:
        continue
    min_y = int(d['planting_year'].min())
    max_y = int(d['planting_year'].max())
    years = list(range(min_y, max_y+1))
    expected = len(states)*len(years)
    grp = d.groupby(['admin_1','planting_year'])['indicator'].nunique().reset_index()
    grp['complete'] = grp['indicator']>=3
    # build full grid for this product
    full = pd.DataFrame(list(itertools.product(states, years)), columns=['admin_1','planting_year'])
    merged = full.merge(grp[['admin_1','planting_year','complete']], on=['admin_1','planting_year'], how='left')
    merged['complete'] = merged['complete'].fillna(False)
    complete_count = int(merged['complete'].sum())
    percent_complete = round(100*complete_count/expected,2) if expected>0 else 0.0
    # indicators counts
    ind_counts = d['indicator'].value_counts().to_dict()
    total_rows = len(d)
    report.append({
        'product':p,
        'states':len(states),
        'min_year':min_y,
        'max_year':max_y,
        'years':len(years),
        'expected_combos':expected,
        'complete_combos':complete_count,
        'percent_complete':percent_complete,
        'total_rows':total_rows,
        'indicator_counts':ind_counts
    })

# sort by percent_complete desc then states desc then years desc
report_sorted = sorted(report, key=lambda x:(-x['percent_complete'], -x['states'], -x['years']))
for r in report_sorted:
    print(f"{r['product']}: states={r['states']}, years={r['min_year']}-{r['max_year']} ({r['years']} yrs), rows={r['total_rows']}, complete={r['complete_combos']}/{r['expected_combos']} ({r['percent_complete']}%)")

# write CSV summary
pd.DataFrame(report_sorted).to_csv('crop_coverage_summary.csv', index=False)
print('\nSummary saved to crop_coverage_summary.csv')
