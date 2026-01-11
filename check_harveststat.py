import pandas as pd

df = pd.read_csv('adm_crop_production_NG.csv')

crops_of_interest = ['Rice', 'Maize', 'Cassava', 'Yams', 'Sorghum']
df_filtered = df[df['product'].isin(crops_of_interest)]

print('Data for your 5 crops:')
print(df_filtered.groupby(['product', 'indicator']).size().unstack(fill_value=0))

print('\n\nYear coverage per crop:')
for crop in crops_of_interest:
    crop_data = df_filtered[df_filtered['product'] == crop]
    if len(crop_data) > 0:
        years = crop_data['planting_year'].unique()
        print(f'{crop}: {years.min()}-{years.max()} ({len(years)} years)')
    else:
        print(f'{crop}: No data')

print('\n\nSample data for Rice in 2020:')
rice_2020 = df_filtered[(df_filtered['product'] == 'Rice') & (df_filtered['planting_year'] == 2020)]
print(rice_2020[['admin_1', 'indicator', 'value']].head(15))

print('\n\nComparing to your current FAOSTAT data:')
faostat = pd.read_csv('FAOSTAT_data_en_12-27-2025.csv')
print(f"FAOSTAT rows: {len(faostat)}")
print(f"FAOSTAT year range: {faostat['Year'].min()}-{faostat['Year'].max()}")
print(f"FAOSTAT unique items: {faostat['Item'].nunique()}")
print(f"FAOSTAT items: {sorted(faostat['Item'].unique())}")
