import pandas as pd
zl = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv')
zl.columns = ['LocationID', 'Borough', 'Zone', 'service_zone']
for kw in ['Grand Central', 'Murray Hill', 'Garment', 'Tudor City', 'Wall Street', 'Upper East', 'Upper West', 'East Village', 'West Village', 'Tribeca', 'Chinatown', 'Little Italy']:
    matches = zl[zl['Zone'].str.contains(kw, case=False, na=False)]
    if len(matches) > 0:
        print(f'{kw}:')
        for _, r in matches.iterrows():
            lid = int(r['LocationID'])
            zone = r['Zone']
            print(f'  {lid:3d} - {zone}')
    else:
        print(f'{kw}: not found')
