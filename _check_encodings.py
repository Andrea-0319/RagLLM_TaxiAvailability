import pandas as pd

df = pd.read_parquet(r'C:\Users\andre\Desktop\Progetto_Accenture\output\model_ready_all_months.parquet')

# Get borough -> borough_encoded mapping
borough_map = df[['borough', 'borough_encoded']].drop_duplicates().sort_values('borough_encoded')
print('borough -> borough_encoded:')
for _, row in borough_map.iterrows():
    b_name = str(row['borough'])
    b_enc = int(row['borough_encoded'])
    print(f'  {b_name:20s} -> {b_enc}')

print()

# Get service_zone -> service_zone_encoded mapping
sz_map = df[['service_zone', 'service_zone_encoded']].drop_duplicates().sort_values('service_zone_encoded')
print('service_zone -> service_zone_encoded:')
for _, row in sz_map.iterrows():
    sz_name = str(row['service_zone'])
    sz_enc = int(row['service_zone_encoded'])
    print(f'  {sz_name:20s} -> {sz_enc}')

print()

# Get defaults for features not provided by user
print('unique_taxi_types mode:', df['unique_taxi_types'].mode().values)
print('avg_trip_duration_min median:', round(df['avg_trip_duration_min'].median(), 2))
print('unique_taxi_types range:', df['unique_taxi_types'].min(), '-', df['unique_taxi_types'].max())
print('avg_trip_duration_min range:', round(df['avg_trip_duration_min'].min(), 2), '-', round(df['avg_trip_duration_min'].max(), 2))

# Get per-zone defaults
zone_defaults = df.groupby('PULocationID').agg(
    unique_taxi_types=('unique_taxi_types', 'median'),
    avg_trip_duration_min=('avg_trip_duration_min', 'median'),
    borough_encoded=('borough_encoded', 'first'),
    service_zone_encoded=('service_zone_encoded', 'first')
).reset_index()

print(f'\nPer-zone defaults: {len(zone_defaults)} zones')
print(zone_defaults.head(10))

# Save zone defaults for use in predictor
zone_defaults.to_csv(r'C:\Users\andre\Desktop\Progetto_Accenture\output\zone_defaults.csv', index=False)
print('\nZone defaults saved!')
