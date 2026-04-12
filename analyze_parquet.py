import pandas as pd

files = {
    'yellow': r'C:\Users\andre\Desktop\Progetto_Accenture\data\yellow_tripdata_2025-01.parquet',
    'green': r'C:\Users\andre\Desktop\Progetto_Accenture\data\green_tripdata_2025-01.parquet',
    'fhv': r'C:\Users\andre\Desktop\Progetto_Accenture\data\fhv_tripdata_2025-01.parquet',
    'fhvhv': r'C:\Users\andre\Desktop\Progetto_Accenture\data\fhvhv_tripdata_2025-01.parquet',
}

schemas = {}
for name, path in files.items():
    df = pd.read_parquet(path)
    schemas[name] = {
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'shape': df.shape,
        'nulls': df.isnull().sum().to_dict(),
    }
    sep = '=' * 80
    print(f'{sep}')
    print(f'  {name.upper()}')
    print(f'{sep}')
    print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
    print(f'\nColumns ({len(df.columns)}):')
    for i, col in enumerate(df.columns, 1):
        print(f'  {i}. {col}')
    print(f'\nDtypes:')
    for col, dtype in df.dtypes.items():
        print(f'  {col}: {dtype}')
    print(f'\nNull counts:')
    for col, null_count in df.isnull().sum().items():
        print(f'  {col}: {null_count}')
    print(f'\nFirst 3 rows:')
    print(df.head(3).to_string())
    print('\n')

# Comparison table
sep = '=' * 100
print(sep)
print('COLUMN COMPARISON TABLE')
print(sep)

all_columns = sorted(set().union(*[s['columns'] for s in schemas.values()]))
header = '{:<45} | {:<8} | {:<8} | {:<8} | {:<8}'.format('Column', 'yellow', 'green', 'fhv', 'fhvhv')
print(header)
print('-' * 100)
for col in all_columns:
    vals = [
        'YES' if col in schemas['yellow']['columns'] else '',
        'YES' if col in schemas['green']['columns'] else '',
        'YES' if col in schemas['fhv']['columns'] else '',
        'YES' if col in schemas['fhvhv']['columns'] else '',
    ]
    print('{:<45} | {:<8} | {:<8} | {:<8} | {:<8}'.format(col, vals[0], vals[1], vals[2], vals[3]))

common = set(schemas['yellow']['columns']) & set(schemas['green']['columns']) & set(schemas['fhv']['columns']) & set(schemas['fhvhv']['columns'])
print('\n\nColumns common to ALL 4 datasets ({}):'.format(len(common)))
for col in sorted(common):
    print('  - {}'.format(col))

yellow_green = set(schemas['yellow']['columns']) & set(schemas['green']['columns'])
print('\nColumns common to yellow + green ({}):'.format(len(yellow_green)))
for col in sorted(yellow_green):
    print('  - {}'.format(col))

fhv_fhvhv = set(schemas['fhv']['columns']) & set(schemas['fhvhv']['columns'])
print('\nColumns common to fhv + fhvhv ({}):'.format(len(fhv_fhvhv)))
for col in sorted(fhv_fhvhv):
    print('  - {}'.format(col))
