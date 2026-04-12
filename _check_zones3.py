import pandas as pd
zl = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv')
zl.columns = ['LocationID', 'Borough', 'Zone', 'service_zone']
# Search for key zones
for kw in ['Wall', 'Grand', 'Central', 'Penn', 'Chelsea', 'SoHo', 'Harlem', 'Times', 'Midtown', 'Financial', 'Battery', 'Flatiron', 'Gramercy', 'Union Sq', 'Hell', 'Hudson Yards', 'Kips Bay', 'Stuyvesant', 'Lenox Hill', 'Carnegie Hill', 'Yorkville', 'Randalls', 'Roosevelt Island']:
    matches = zl[zl['Zone'].str.contains(kw, case=False, na=False)]
    if len(matches) > 0:
        print(f'{kw}:')
        for _, r in matches.iterrows():
            lid = int(r['LocationID'])
            zone = r['Zone']
            print(f'  {lid:3d} - {zone}')
