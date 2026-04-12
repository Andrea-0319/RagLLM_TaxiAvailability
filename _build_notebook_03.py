"""Generate notebook: Multi-Month Data Processing (Gen-Dic 2025)."""
import json

def md(lines):
    raw = lines.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in raw]}

def code(text):
    raw = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in raw]}

cells = []

cells.append(md("""# NYC Taxi Demand Prediction - Multi-Month Data Processing

## Obiettivo
Processare **tutti i mesi disponibili del 2025** (Gennaio-Dicembre) dai 4 dataset TLC:
- **Yellow Taxi** - Taxi tradizionali Manhattan
- **Green Taxi** - Taxi outer boroughs
- **FHV** - For-Hire Vehicles (Uber, Lyft, etc.)
- **HVFHV** - High Volume FHV (solo Uber/Lyft)

## Strategia di Processamento
1. **Scansione automatica** dei file disponibili nella cartella `data/`
2. **Processamento mese-per-mese** per gestire la memoria:
   - Carica un mese → pulisci → feature engineering → aggrega → salva → libera memoria
   - Questo evita di tenere tutti i dati grezzi in RAM contemporaneamente
3. **Concatenazione** di tutti i mesi aggregati
4. **Calcolo availability_index GLOBALE** sul dataset combinato (il max trip_count per zona e' il picco assoluto su tutti i mesi)
5. **Salvataggio** del dataset finale pronto per il ML

## Perche' availability_index globale?
Calcolare il max per zona su tutti i mesi combinati e' piu' corretto:
- Se una zona ha picco 5000 corse a Giugno e 3000 a Gennaio, il "100%" della sua capacita' e' 5000
- Questo rende le classi comparabili tra mesi diversi
- Un valore di 0.8 a Gennaio significa la stessa cosa di 0.8 a Giugno"""))

cells.append(md("## 1. Setup - Import e Configurazione"))

cells.append(code("""import warnings
import gc
import time
import os
import glob as glob_mod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

DATA_DIR = Path(r'C:\\Users\\andre\\Desktop\\Progetto_Accenture\\data')
OUTPUT_DIR = Path(r'C:\\Users\\andre\\Desktop\\Progetto_Accenture\\output')
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = OUTPUT_DIR / 'temp_monthly'
TEMP_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("NYC Taxi Demand Prediction - Multi-Month Processing")
print("=" * 70)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Temp directory: {TEMP_DIR}")"""))

cells.append(md("## 2. Taxi Zone Lookup"))

cells.append(code("""zones = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv')
print(f"Zone totali: {len(zones)}")
print(f"\\nDistribuzione per Borough:")
print(zones[zones['Borough'] != 'N/A']['Borough'].value_counts().to_string())
print(f"\\nDistribuzione per Service Zone:")
print(zones['service_zone'].value_counts().to_string())"""))

cells.append(md("## 3. Scansione File Disponibili"))

cells.append(code("""# Trova tutti i mesi disponibili per ogni tipo di taxi
def find_available_months(data_dir):
    \"\"\"Scansiona la directory data/ e trova tutti i mesi disponibili.\"\"\"
    types = {
        'yellow': 'yellow_tripdata_2025-*.parquet',
        'green': 'green_tripdata_2025-*.parquet',
        'fhv': 'fhv_tripdata_2025-*.parquet',
        'fhvhv': 'fhvhv_tripdata_2025-*.parquet'
    }

    months_by_type = {}
    all_months = set()

    for taxi_type, pattern in types.items():
        files = sorted(glob_mod.glob(str(data_dir / pattern)))
        months = set()
        for f in files:
            # Estrai MM dal filename
            stem = Path(f).stem
            month = stem.split('-')[-1]  # es. '2025-03' -> '03'
            months.add(month)
        months_by_type[taxi_type] = sorted(months)
        all_months.update(months)
        print(f"  {taxi_type:>6}: {len(files)} file trovati -> mesi {sorted(months)}")

    all_months = sorted(all_months)
    print(f"\\nMesi totali disponibili: {len(all_months)} -> {all_months}")

    # Verifica copertura
    print("\\nCopertura per mese:")
    for m in all_months:
        present = [t for t in types if m in months_by_type[t]]
        missing = [t for t in types if m not in months_by_type[t]]
        status = "OK" if len(present) == 4 else f"MANCANO: {missing}"
        print(f"  2025-{m}: {len(present)}/4 dataset [{status}]")

    return all_months, months_by_type

all_months, months_by_type = find_available_months(DATA_DIR)"""))

cells.append(md("## 4. Funzioni di Pulizia e Feature Engineering"))

cells.append(md("""Queste funzioni sono identiche a quelle usate in `01_eda_feature_engineering.py`.
Le ridefiniamo qui per avere un notebook completamente standalone e riproducibile."""))

cells.append(code("""def clean_dataset(df, dataset_type):
    \"\"\"
    Pulisce e standardizza un dataset TLC.

    Standardizza i nomi delle colonne e filtra:
    - PULocationID nel range 1-265
    - Durata corsa tra 0 e 24 ore

    Args:
        df: DataFrame grezzo
        dataset_type: 'yellow', 'green', 'fhv', o 'fhvhv'

    Returns:
        DataFrame con colonne: pickup_datetime, PULocationID, taxi_type, trip_duration_sec
    \"\"\"
    df = df.copy()

    # Standardizza nomi colonne
    if dataset_type == 'yellow':
        df = df.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'tpep_dropoff_datetime': 'dropoff_datetime'
        })
    elif dataset_type == 'green':
        df = df.rename(columns={
            'lpep_pickup_datetime': 'pickup_datetime',
            'lpep_dropoff_datetime': 'dropoff_datetime'
        })
    elif dataset_type == 'fhv':
        df = df.rename(columns={
            'PUlocationID': 'PULocationID',
            'DOlocationID': 'DOLocationID',
            'dropOff_datetime': 'dropoff_datetime'
        })

    df['taxi_type'] = dataset_type

    # Filtra location valida
    before = len(df)
    df = df[df['PULocationID'].notna()]
    df['PULocationID'] = df['PULocationID'].astype(int)
    df = df[df['PULocationID'].between(1, 265)]

    # Calcola durata corsa
    if 'dropoff_datetime' in df.columns:
        df['trip_duration_sec'] = (
            pd.to_datetime(df['dropoff_datetime']) - pd.to_datetime(df['pickup_datetime'])
        ).dt.total_seconds()
    elif 'trip_time' in df.columns:
        df['trip_duration_sec'] = df['trip_time'].astype(float)

    # Filtra durata valida (0 < durata < 24 ore)
    df = df[(df['trip_duration_sec'] > 0) & (df['trip_duration_sec'] < 86400)]

    removed = before - len(df)
    print(f"  {dataset_type:>6}: {before:>10,} -> {len(df):>10,} righe (rimosse {removed:>8,})")

    return df[['pickup_datetime', 'PULocationID', 'taxi_type', 'trip_duration_sec']].copy()


def add_features(df):
    \"\"\"
    Aggiunge feature temporali e geografiche a un dataset pulito.

    Feature create:
    - hour, minute, half_hour_bucket (0-47)
    - day_of_week (0=Lun, 6=Dom)
    - month
    - is_weekend, is_rush_hour, is_night
    - trip_duration_min
    - borough, service_zone, zone_name (da JOIN con zone lookup)
    \"\"\"
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Feature temporali
    df['hour'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute
    df['half_hour_bucket'] = df['hour'] * 2 + (df['minute'] // 30)
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month

    # Feature booleane
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    df['is_night'] = (df['hour'] >= 22) | (df['hour'] < 5)

    # Durata in minuti
    df['trip_duration_min'] = df['trip_duration_sec'] / 60

    # JOIN con zone lookup
    zone_info = zones.set_index('LocationID')
    df = df.join(zone_info, on='PULocationID', how='left')
    df = df.rename(columns={'Zone': 'zone_name', 'Borough': 'borough'})

    return df"""))

cells.append(md("## 5. Funzione di Processamento Mensile"))

cells.append(code("""def process_single_month(month_str, data_dir, zones_df):
    \"\"\"
    Processa un singolo mese di dati TLC.

    Strategia memoria:
    - Carica un mese alla volta
    - HVFHV: prova 100%, fallback 75% -> 50% -> 25% se MemoryError
    - Dopo aggregazione, libera tutta la memoria

    Args:
        month_str: stringa 'MM' del mese (es. '01', '02')
        data_dir: Path alla directory data/
        zones_df: DataFrame del zone lookup

    Returns:
        aggregated: DataFrame aggregato per il mese (senza availability_index)
        stats: dict con statistiche del processamento
    \"\"\"
    print(f"\\n{'='*60}")
    print(f"Processing: 2025-{month_str}")
    print(f"{'='*60}")

    stats = {'month': f'2025-{month_str}', 'fhvhv_sample_rate': 1.0}

    # --- Caricamento ---
    t0 = time.time()

    # Yellow
    yellow_path = data_dir / f'yellow_tripdata_2025-{month_str}.parquet'
    if yellow_path.exists():
        yellow = pd.read_parquet(yellow_path)
        print(f"  Yellow caricato: {len(yellow):,} righe")
    else:
        yellow = pd.DataFrame()

    # Green
    green_path = data_dir / f'green_tripdata_2025-{month_str}.parquet'
    if green_path.exists():
        green = pd.read_parquet(green_path)
        print(f"  Green caricato: {len(green):,} righe")
    else:
        green = pd.DataFrame()

    # FHV
    fhv_path = data_dir / f'fhv_tripdata_2025-{month_str}.parquet'
    if fhv_path.exists():
        fhv = pd.read_parquet(fhv_path)
        fhv_null_pct = fhv['PUlocationID'].isna().sum() / len(fhv) * 100
        print(f"  FHV caricato: {len(fhv):,} righe (PUlocationID missing: {fhv_null_pct:.1f}%)")
    else:
        fhv = pd.DataFrame()

    # HVFHV con fallback memoria
    fhvhv_path = data_dir / f'fhvhv_tripdata_2025-{month_str}.parquet'
    fhvhv = pd.DataFrame()
    fhvhv_cols = ['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_time', 'trip_miles']

    if fhvhv_path.exists():
        for sample_rate in [1.0, 0.75, 0.50, 0.25]:
            try:
                print(f"  HVFHV: tentando caricamento al {sample_rate*100:.0f}%...")
                table = pq.read_table(fhvhv_path, columns=fhvhv_cols)
                fhvhv = table.to_pandas()
                del table
                if sample_rate < 1.0:
                    fhvhv = fhvhv.sample(frac=sample_rate / 1.0, random_state=42).reset_index(drop=True)
                print(f"  HVFHV caricato: {len(fhvhv):,} righe ({sample_rate*100:.0f}%)")
                stats['fhvhv_sample_rate'] = sample_rate
                break
            except MemoryError:
                print(f"  HVFHV: MemoryError al {sample_rate*100:.0f}%, riprovo con meno...")
                continue
            except Exception as e:
                print(f"  HVFHV: Errore inatteso: {e}")
                break

    load_time = time.time() - t0
    stats['load_time'] = round(load_time, 1)

    # --- Pulizia ---
    print("\\n  Pulizia:")
    datasets = []
    if len(yellow) > 0:
        datasets.append(clean_dataset(yellow, 'yellow'))
    if len(green) > 0:
        datasets.append(clean_dataset(green, 'green'))
    if len(fhv) > 0:
        datasets.append(clean_dataset(fhv, 'fhv'))
    if len(fhvhv) > 0:
        datasets.append(clean_dataset(fhvhv, 'fhvhv'))

    # Libera memoria dai grezzi
    del yellow, green, fhv, fhvhv
    gc.collect()

    if not datasets:
        print("  NESSUN DATASET VALIDO!")
        return None, stats

    # --- Feature Engineering ---
    print("\\n  Feature engineering:")
    featured = []
    for ds in datasets:
        f = add_features(ds)
        featured.append(f)
        print(f"    {f['taxi_type'].iloc[0]:>6}: {len(f):,} righe")

    # --- Unione ---
    cols = [
        'pickup_datetime', 'PULocationID', 'taxi_type', 'trip_duration_sec',
        'hour', 'minute', 'half_hour_bucket', 'day_of_week', 'month',
        'is_weekend', 'is_rush_hour', 'is_night', 'trip_duration_min',
        'borough', 'service_zone', 'zone_name'
    ]

    all_trips = pd.concat([f[cols] for f in featured], ignore_index=True)
    total_raw = len(all_trips)
    stats['total_raw_trips'] = total_raw
    print(f"\\n  Dataset unito: {total_raw:,} righe")

    # --- Aggregazione ---
    print("  Aggregazione per (zona, fascia 30min, giorno, mese)...")
    aggregated = all_trips.groupby(
        ['PULocationID', 'half_hour_bucket', 'day_of_week', 'month']
    ).agg(
        trip_count=('pickup_datetime', 'count'),
        unique_taxi_types=('taxi_type', 'nunique'),
        avg_trip_duration_min=('trip_duration_min', 'mean'),
        median_trip_duration_min=('trip_duration_min', 'median'),
        std_trip_duration_min=('trip_duration_min', 'std'),
    ).reset_index()

    # JOIN con zone info
    zone_info = zones_df.set_index('LocationID')
    aggregated = aggregated.join(zone_info, on='PULocationID', how='left')
    aggregated = aggregated.rename(columns={'Zone': 'zone_name', 'Borough': 'borough'})

    # Feature derivate
    aggregated['is_weekend'] = aggregated['day_of_week'] >= 5
    aggregated['is_rush_hour'] = aggregated['half_hour_bucket'].apply(
        lambda b: (b // 2) in [7, 8, 9, 17, 18, 19]
    )
    aggregated['is_night'] = aggregated['half_hour_bucket'].apply(
        lambda b: (b // 2) >= 22 or (b // 2) < 5
    )

    stats['aggregated_rows'] = len(aggregated)
    stats['total_time'] = round(time.time() - t0, 1)
    print(f"  Aggregato: {len(aggregated):,} righe")
    print(f"  Tempo totale: {stats['total_time']:.1f}s")

    # Libera memoria
    del all_trips, datasets, featured
    gc.collect()

    return aggregated, stats"""))

cells.append(md("## 6. Processamento di Tutti i Mesi"))

cells.append(code("""# Processa ogni mese e salva l'aggregato temporaneo
all_stats = []
months_processed = []

for month_str in all_months:
    agg, stats = process_single_month(month_str, DATA_DIR, zones)

    if agg is not None:
        # Salva aggregato mensile come parquet temporaneo
        temp_path = TEMP_DIR / f'aggregated_2025-{month_str}.parquet'
        agg.to_parquet(temp_path, index=False)
        months_processed.append(month_str)
        print(f"  Salvato: {temp_path}")

    all_stats.append(stats)
    gc.collect()

print(f"\\n{'='*60}")
print(f"PROCESSAMENTO COMPLETATO")
print(f"{'='*60}")
print(f"Mesi processati: {len(months_processed)}/{len(all_months)}")
print(f"Mesi: {months_processed}")

# Riepilogo statistico
stats_df = pd.DataFrame(all_stats)
print(f"\\nRiepilogo per mese:")
print(stats_df[['month', 'total_raw_trips', 'aggregated_rows', 'fhvhv_sample_rate', 'total_time']].to_string(index=False))

total_raw = stats_df['total_raw_trips'].sum()
total_agg = stats_df['aggregated_rows'].sum()
print(f"\\nTOTALE corse grezze: {total_raw:,}")
print(f"TOTALE righe aggregate: {total_agg:,}")
print(f"Rapporto compressione: {total_raw/total_agg:.0f}:1")"""))

cells.append(md("## 7. Concatenazione e Availability Index Globale"))

cells.append(md("""Ora carichiamo tutti gli aggregati mensili, li concateniamo e calcoliamo
l'**availability_index GLOBALE** usando il massimo trip_count assoluto per zona su tutti i mesi."""))

cells.append(code("""# Carica e concatena tutti gli aggregati mensili
print("Caricamento aggregati mensili...")
monthly_dfs = []
for month_str in months_processed:
    temp_path = TEMP_DIR / f'aggregated_2025-{month_str}.parquet'
    mdf = pd.read_parquet(temp_path)
    monthly_dfs.append(mdf)
    print(f"  2025-{month_str}: {len(mdf):,} righe")

combined = pd.concat(monthly_dfs, ignore_index=True)
print(f"\\nDataset combinato: {len(combined):,} righe x {len(combined.columns)} colonne")
print(f"Memoria: {combined.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Distribuzione per mese
print(f"\\nRighe per mese:")
for m in sorted(combined['month'].unique()):
    cnt = (combined['month'] == m).sum()
    print(f"  Mese {int(m):2d}: {cnt:,} righe\")"""))

cells.append(code("""# Calcolo availability_index GLOBALE
print("\\n" + "=" * 60)
print("CALCOLO AVAILABILITY INDEX GLOBALE")
print("=" * 60)

# Max trip_count per zona su TUTTI i mesi combinati
max_per_zone = combined.groupby('PULocationID')['trip_count'].max().rename('max_trip_count_zone')
combined = combined.join(max_per_zone, on='PULocationID')

# Availability index
combined['availability_index'] = combined['trip_count'] / combined['max_trip_count_zone']

print(f"Max trip_count per zona calcolato su {len(max_per_zone)} zone")
print(f"Range availability_index: [{combined['availability_index'].min():.4f}, {combined['availability_index'].max():.4f}]")

# Classi di disponibilita'
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['Molto Difficile', 'Difficile', 'Medio', 'Facile', 'Molto Facile']
combined['availability_class'] = pd.cut(
    combined['availability_index'],
    bins=bins,
    labels=labels,
    include_lowest=True
)
class_mapping = {label: i for i, label in enumerate(labels)}
combined['availability_class_id'] = combined['availability_class'].map(class_mapping)

print("\\nDistribuzione delle Classi di Disponibilita':")
class_dist = combined['availability_class'].value_counts().sort_index()
for cls, count in class_dist.items():
    pct = count / len(combined) * 100
    bar = '#' * int(pct / 2)
    print(f"  {cls:>15}: {count:>7,} ({pct:>5.1f}%) {bar}")

# Visualizzazione classi
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(combined['availability_index'], bins=50, ax=axes[0], color='#4CAF50', edgecolor='black')
axes[0].set_title('Distribuzione Availability Index (Globale)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Availability Index (0-1)')
for b in [0.2, 0.4, 0.6, 0.8]:
    axes[0].axvline(b, color='gray', linestyle=':', alpha=0.7)

class_counts = combined['availability_class'].value_counts().sort_index()
class_colors = ['#D32F2F', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[1], palette=class_colors)
axes[1].set_title('Distribuzione Classi (Globale)', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)
for i, (cls, count) in enumerate(class_counts.items()):
    axes[1].text(i, count + max(class_counts.values)*0.01, f'{count:,}', ha='center', fontweight='bold', fontsize=9)

axes[2].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=class_colors)
axes[2].set_title('Proporzione Classi (Globale)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_multi_month_class_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\\n[Figura salvata: 09_multi_month_class_dist.png]")"""))

cells.append(md("## 8. EDA sul Dataset Multi-Mese"))

cells.append(md("""Analisi esplorativa rapida sul dataset combinato per verificare consistenza
e identificare pattern temporali su scala annuale."""))

cells.append(code("""# Pattern per mese
print("=== Pattern per Mese ===")
monthly_stats = combined.groupby('month').agg(
    total_trips=('trip_count', 'sum'),
    avg_trips=('trip_count', 'mean'),
    avg_avail=('availability_index', 'mean'),
    active_zones=('PULocationID', 'nunique')
).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

month_names = {1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mag', 6: 'Giu',
               7: 'Lug', 8: 'Ago', 9: 'Set', 10: 'Ott', 11: 'Nov', 12: 'Dic'}
monthly_stats['month_name'] = monthly_stats['month'].map(month_names)

sns.barplot(data=monthly_stats, x='month_name', y='total_trips', ax=axes[0, 0], palette='Blues')
axes[0, 0].set_title('Trip Count Totale per Mese', fontsize=14, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

sns.barplot(data=monthly_stats, x='month_name', y='avg_trips', ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title('Trip Count Medio per Combinazione', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.barplot(data=monthly_stats, x='month_name', y='avg_avail', ax=axes[1, 0], palette='Oranges')
axes[1, 0].set_title('Availability Index Medio per Mese', fontsize=14, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.barplot(data=monthly_stats, x='month_name', y='active_zones', ax=axes[1, 1], palette='Purples')
axes[1, 1].set_title('Zone Attive per Mese', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_monthly_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 10_monthly_patterns.png]")
print(monthly_stats.to_string(index=False))"""))

cells.append(code("""# Pattern orari (media su tutti i mesi)
print("\\n=== Pattern Orari (Media su Tutti i Mesi) ===")
hourly_avail = combined.groupby('half_hour_bucket').agg(
    avg_avail=('availability_index', 'mean'),
    total_trips=('trip_count', 'sum')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.lineplot(data=hourly_avail, x='half_hour_bucket', y='avg_avail',
             ax=axes[0], color='#4CAF50', marker='o', markersize=4)
axes[0].set_title('Disponibilita Media per Fascia Oraria', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fascia Oraria (0-47)')
axes[0].set_ylabel('Availability Index Medio')
axes[0].set_xticks(range(0, 48, 4))
axes[0].set_xticklabels([f"{i//2:02d}:{'00' if i%2==0 else '30'}" for i in range(0, 48, 4)], rotation=45)

sns.lineplot(data=hourly_avail, x='half_hour_bucket', y='total_trips',
             ax=axes[1], color='#2196F3', marker='o', markersize=4)
axes[1].set_title('Trip Count Totale per Fascia', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fascia Oraria (0-47)')
axes[1].set_ylabel('Trip Count Totale')
axes[1].set_xticks(range(0, 48, 4))
axes[1].set_xticklabels([f"{i//2:02d}:{'00' if i%2==0 else '30'}" for i in range(0, 48, 4)], rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '11_multi_month_hourly.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 11_multi_month_hourly.png]")"""))

cells.append(code("""# Pattern per giorno della settimana
print("\\n=== Pattern per Giorno ===")
dow_names = ['Lunedi', 'Martedi', 'Mercoledi', 'Giovedi', 'Venerdi', 'Sabato', 'Domenica']
dow_avail = combined.groupby('day_of_week').agg(
    avg_avail=('availability_index', 'mean'),
    total_trips=('trip_count', 'sum')
).reset_index()
dow_avail['day_name'] = dow_avail['day_of_week'].map(dict(enumerate(dow_names)))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(data=dow_avail, x='day_name', y='avg_avail', ax=axes[0], palette='Greens')
axes[0].set_title('Disponibilita Media per Giorno', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(data=dow_avail, x='day_name', y='total_trips', ax=axes[1], palette='Blues')
axes[1].set_title('Trip Count Totale per Giorno', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '12_multi_month_daily.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 12_multi_month_daily.png]")

# Heatmap: Giorno x Ora (media su tutti i mesi)
print("\\n=== Heatmap: Giorno x Ora ===")
dow_hour = combined.groupby(['day_of_week', 'half_hour_bucket'])['availability_index'].mean().unstack(fill_value=0)
dow_hour.index = [dow_names[i] for i in dow_hour.index]

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(dow_hour, cmap='RdYlGn', ax=ax, vmin=0, vmax=1,
            cbar_kws={'label': 'Availability Index Medio'})
ax.set_title('Disponibilita: Giorno x Fascia Oraria (Media su Tutti i Mesi)', fontsize=14, fontweight='bold')
ax.set_xlabel('Fascia Oraria (0-47)')
ax.set_ylabel('Giorno della Settimana')
ax.set_xticks(range(0, 48, 4))
ax.set_xticklabels([f"{i//2:02d}:{'00' if i%2==0 else '30'}" for i in range(0, 48, 4)], rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '13_multi_month_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 13_multi_month_heatmap.png]")"""))

cells.append(md("## 9. Preparazione Dataset Finale per il Modello"))

cells.append(code("""from sklearn.preprocessing import LabelEncoder

# Selezione feature finali
model_features = [
    'PULocationID',
    'half_hour_bucket',
    'day_of_week',
    'month',
    'unique_taxi_types',
    'avg_trip_duration_min',
    'is_weekend',
    'is_rush_hour',
    'is_night',
    'borough',
    'service_zone',
    'availability_class_id',
    'availability_index',
]

model_df = combined[model_features].copy()

# Codifica booleane come int
model_df['is_weekend'] = model_df['is_weekend'].astype(int)
model_df['is_rush_hour'] = model_df['is_rush_hour'].astype(int)
model_df['is_night'] = model_df['is_night'].astype(int)

# Label encoding per categoriche
le_borough = LabelEncoder()
model_df['borough_encoded'] = le_borough.fit_transform(model_df['borough'].fillna('Unknown'))

le_service = LabelEncoder()
model_df['service_zone_encoded'] = le_service.fit_transform(model_df['service_zone'].fillna('Unknown'))

print(f"Dataset per il modello: {len(model_df):,} righe x {len(model_df.columns)} colonne")
print(f"\\nFeature finali:")
for col in model_df.columns:
    dtype = str(model_df[col].dtype)
    nulls = model_df[col].isna().sum()
    unique = model_df[col].nunique()
    print(f"  {col:<25} | {dtype:<15} | nulls: {nulls:>5} | unique: {unique:>5}")

# Riepilogo distribuzione classi
print(f"\\nDistribuzione classi nel dataset finale:")
for cls_id in sorted(model_df['availability_class_id'].unique()):
    cnt = (model_df['availability_class_id'] == cls_id).sum()
    pct = cnt / len(model_df) * 100
    cls_name = ['Molto Difficile', 'Difficile', 'Medio', 'Facile', 'Molto Facile'][int(cls_id)]
    print(f"  {cls_id} - {cls_name:>15}: {cnt:>7,} ({pct:>5.1f}%)")"""))

cells.append(md("## 10. Salvataggio Finale"))

cells.append(code("""print("=" * 60)
print("SALVATAGGIO DATASET FINALE")
print("=" * 60)

# Dataset aggregato completo (con tutte le colonne originali)
combined.to_parquet(OUTPUT_DIR / 'aggregated_all_months.parquet', index=False)
print(f"Dataset aggregato completo: {OUTPUT_DIR / 'aggregated_all_months.parquet'}")
print(f"  Righe: {len(combined):,}")
print(f"  Colonne: {len(combined.columns)}")

# Dataset pronto per il modello
model_df.to_parquet(OUTPUT_DIR / 'model_ready_all_months.parquet', index=False)
print(f"\\nDataset ML pronto: {OUTPUT_DIR / 'model_ready_all_months.parquet'}")
print(f"  Righe: {len(model_df):,}")
print(f"  Colonne: {len(model_df.columns)}")
print(f"  Memoria: {model_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Label encoder
joblib.dump(le_borough, OUTPUT_DIR / 'le_borough_all.pkl')
joblib.dump(le_service, OUTPUT_DIR / 'le_service_all.pkl')
print(f"\\nLabel encoder salvati:")
print(f"  {OUTPUT_DIR / 'le_borough_all.pkl'}")
print(f"  {OUTPUT_DIR / 'le_service_all.pkl'}")

# Statistiche di processamento
stats_df.to_csv(OUTPUT_DIR / 'monthly_processing_stats.csv', index=False)
print(f"\\nStatistiche mensili: {OUTPUT_DIR / 'monthly_processing_stats.csv'}")

# Pulizia file temporanei
print(f"\\nPulizia file temporanei in {TEMP_DIR}...")
for f in TEMP_DIR.glob('*.parquet'):
    f.unlink()
TEMP_DIR.rmdir()
print("File temporanei eliminati.")

print(f"\\n{'='*60}")
print("PROCESSAMENTO MULTI-MESE COMPLETATO!")
print(f"{'='*60}")
print(f"Mesi processati: {len(months_processed)}")
print(f"Dataset finale: {len(model_df):,} righe")
print('Pronto per il training del modello!')"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython", "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'C:\Users\andre\Desktop\Progetto_Accenture\03_multi_month_processing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook 03 scritto!")
with open(r'C:\Users\andre\Desktop\Progetto_Accenture\03_multi_month_processing.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)
print("JSON valido!")
print(f"Celle totali: {len(cells)}")