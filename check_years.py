import pandas as pd

df = pd.read_csv('adm_crop_production_NG.csv')
print('min_year', df['planting_year'].min())
print('max_year', df['planting_year'].max())
print('unique_products', sorted(df['product'].unique()))
