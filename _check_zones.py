import pandas as pd
zl = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv')
zl.columns = ['LocationID', 'Borough', 'Zone', 'service_zone']
for kw in ['JFK', 'LaGuardia', 'Newark', 'Airport', 'Grand Central', 'Penn']:
    matches = zl[zl['Zone'].str.contains(kw, case=False, na=False)]
    if len(matches) > 0:
        print(f'{kw}:')
        for _, r in matches.iterrows():
            lid = int(r['LocationID'])
            zone = r['Zone']
            borough = r['Borough']
            print(f'  {lid:3d} - {zone} ({borough})')
