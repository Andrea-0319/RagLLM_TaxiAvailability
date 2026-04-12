# -*- coding: utf-8 -*-
"""
NYC Taxi Demand Prediction — EDA & Feature Engineering
=======================================================

Analisi esplorativa e feature engineering sui 4 dataset TLC di Gennaio 2025:
- Yellow Taxi, Green Taxi, FHV, High-Volume FHV

Obiettivo: costruire un dataset aggregato con availability_index come target
per un modello di classificazione della disponibilità di taxi.

Granularità: fasce di 30 minuti
Target: availability_index = trip_count / max_trip_count_per_zone
Classi: 5 livelli da "Molto Difficile" a "Molto Facile"
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per salvataggio figure
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import warnings
import os
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

DATA_DIR = Path(r'C:\Users\andre\Desktop\Progetto_Accenture\data')
OUTPUT_DIR = Path(r'C:\Users\andre\Desktop\Progetto_Accenture\output')
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    'yellow': DATA_DIR / 'yellow_tripdata_2025-01.parquet',
    'green': DATA_DIR / 'green_tripdata_2025-01.parquet',
    'fhv': DATA_DIR / 'fhv_tripdata_2025-01.parquet',
    'fhvhv': DATA_DIR / 'fhvhv_tripdata_2025-01.parquet'
}

# Campionamento HVFHV per limiti di memoria del laptop
# Impostare a 1.0 quando si ha più RAM disponibile
SAMPLE_FHVHV = 0.1

print("=" * 70)
print("NYC Taxi Demand Prediction — EDA & Feature Engineering")
print("=" * 70)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"HVFHV sampling: {SAMPLE_FHVHV*100:.0f}%")

# =============================================================================
# 1. TAXI ZONE LOOKUP
# =============================================================================

print("\n" + "=" * 70)
print("1. TAXI ZONE LOOKUP")
print("=" * 70)

zones = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv')
print(f"Zone totali: {len(zones)}")
print(f"\nDistribuzione per Borough:")
print(zones[zones['Borough'] != 'N/A']['Borough'].value_counts().to_string())
print(f"\nDistribuzione per Service Zone:")
print(zones['service_zone'].value_counts().to_string())

# Visualizzazione zone
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
borough_counts = zones[zones['Borough'] != 'N/A']['Borough'].value_counts()
sns.barplot(x=borough_counts.index, y=borough_counts.values, ax=axes[0], palette='Set2')
axes[0].set_title('Numero di Zone per Borough', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Borough')
axes[0].set_ylabel('Numero Zone')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(borough_counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

sz_counts = zones[zones['service_zone'] != 'N/A']['service_zone'].value_counts()
sns.barplot(x=sz_counts.index, y=sz_counts.values, ax=axes[1], palette='Set2')
axes[1].set_title('Numero di Zone per Tipo di Servizio', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tipo Servizio')
axes[1].set_ylabel('Numero Zone')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(sz_counts.values):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_zone_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Figura salvata: 01_zone_distribution.png]")

# =============================================================================
# 2. CARICAMENTO E ANALISI INDIVIDUALE DEI DATASET
# =============================================================================

print("\n" + "=" * 70)
print("2. CARICAMENTO E ANALISI INDIVIDUALE")
print("=" * 70)

# --- YELLOW TAXI ---
print("\n--- Yellow Taxi ---")
yellow = pd.read_parquet(FILES['yellow'])
print(f"Shape: {yellow.shape[0]:,} righe x {yellow.shape[1]} colonne")
print(f"Memoria: {yellow.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print("\nSchema:")
for col in yellow.columns:
    dtype = str(yellow[col].dtype)
    nulls = yellow[col].isna().sum()
    null_pct = nulls / len(yellow) * 100
    unique = yellow[col].nunique()
    print(f"  {col:<30} | {dtype:<20} | nulls: {nulls:>8,} ({null_pct:>5.1f}%) | unique: {unique:>8,}")

# Distribuzione temporale Yellow
yellow['pickup_hour'] = pd.to_datetime(yellow['tpep_pickup_datetime']).dt.hour
yellow['pickup_dow'] = pd.to_datetime(yellow['tpep_pickup_datetime']).dt.dayofweek

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
hourly_y = yellow['pickup_hour'].value_counts().sort_index()
sns.barplot(x=hourly_y.index, y=hourly_y.values, ax=axes[0], color='#FFD700', edgecolor='black')
axes[0].set_title('Yellow — Corse per Ora', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(24))

dow_names = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
dow_y = yellow['pickup_dow'].value_counts().sort_index()
dow_y.index = [dow_names[i] for i in dow_y.index]
sns.barplot(x=dow_y.index, y=dow_y.values, ax=axes[1], color='#FFD700', edgecolor='black')
axes[1].set_title('Yellow — Corse per Giorno', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_yellow_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Figura salvata: 02_yellow_temporal.png]")
print(f"Ora di picco: {hourly_y.idxmax()}:00 ({hourly_y.max():,} corse)")

# --- GREEN TAXI ---
print("\n--- Green Taxi ---")
green = pd.read_parquet(FILES['green'])
print(f"Shape: {green.shape[0]:,} righe x {green.shape[1]} colonne")
print(f"Memoria: {green.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

green['pickup_hour'] = pd.to_datetime(green['lpep_pickup_datetime']).dt.hour
green['pickup_dow'] = pd.to_datetime(green['lpep_pickup_datetime']).dt.dayofweek

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
hourly_g = green['pickup_hour'].value_counts().sort_index()
sns.barplot(x=hourly_g.index, y=hourly_g.values, ax=axes[0], color='#3CB371', edgecolor='black')
axes[0].set_title('Green — Corse per Ora', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(24))

dow_g = green['pickup_dow'].value_counts().sort_index()
dow_g.index = [dow_names[i] for i in dow_g.index]
sns.barplot(x=dow_g.index, y=dow_g.values, ax=axes[1], color='#3CB371', edgecolor='black')
axes[1].set_title('Green — Corse per Giorno', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_green_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 03_green_temporal.png]")
print(f"Green ha solo {len(green):,} corse — molto meno di Yellow.")

# --- FHV ---
print("\n--- FHV (For-Hire Vehicle) ---")
fhv = pd.read_parquet(FILES['fhv'])
print(f"Shape: {fhv.shape[0]:,} righe x {fhv.shape[1]} colonne")
print(f"Memoria: {fhv.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\nSchema:")
for col in fhv.columns:
    dtype = str(fhv[col].dtype)
    nulls = fhv[col].isna().sum()
    null_pct = nulls / len(fhv) * 100
    print(f"  {col:<30} | {dtype:<20} | nulls: {nulls:>8,} ({null_pct:>5.1f}%)")

fhv_null_pct = fhv['PUlocationID'].isna().sum() / len(fhv) * 100
print(f"\nATTENZIONE: PUlocationID ha il {fhv_null_pct:.1f}% di missing!")
print(f"Righe con location valida: {(~fhv['PUlocationID'].isna()).sum():,}")
print("Decisione: Manteniamo SOLO righe con PUlocationID valido (1-265).")

# FHV temporale (solo con location valida)
fhv_valid = fhv[fhv['PUlocationID'].notna() & fhv['PUlocationID'].between(1, 265)].copy()
fhv_valid['pickup_hour'] = pd.to_datetime(fhv_valid['pickup_datetime']).dt.hour
fhv_valid['pickup_dow'] = pd.to_datetime(fhv_valid['pickup_datetime']).dt.dayofweek

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
hourly_f = fhv_valid['pickup_hour'].value_counts().sort_index()
sns.barplot(x=hourly_f.index, y=hourly_f.values, ax=axes[0], color='#FF6347', edgecolor='black')
axes[0].set_title('FHV — Corse per Ora (con location)', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(24))

dow_f = fhv_valid['pickup_dow'].value_counts().sort_index()
dow_f.index = [dow_names[i] for i in dow_f.index]
sns.barplot(x=dow_f.index, y=dow_f.values, ax=axes[1], color='#FF6347', edgecolor='black')
axes[1].set_title('FHV — Corse per Giorno (con location)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_fhv_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 04_fhv_temporal.png]")
print(f"FHV con location valida: {len(fhv_valid):,} righe ({100-fhv_null_pct:.1f}%)")

# --- HVFHV ---
print("\n--- HVFHV (High-Volume FHV) ---")
# Carichiamo solo 5 colonne per evitare MemoryError
fhvhv_cols = ['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_time', 'trip_miles']
print(f"Caricamento HVFHV (solo {len(fhvhv_cols)} colonne per memoria)...")
table = pq.read_table(FILES['fhvhv'], columns=fhvhv_cols)
fhvhv = table.to_pandas()

# Campionamento
if SAMPLE_FHVHV < 1.0:
    print(f"Campionamento: {SAMPLE_FHVHV*100:.0f}% dei dati...")
    fhvhv = fhvhv.sample(frac=SAMPLE_FHVHV, random_state=42).reset_index(drop=True)

print(f"Shape: {fhvhv.shape[0]:,} righe x {fhvhv.shape[1]} colonne")
print(f"Memoria: {fhvhv.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"Colonne caricate: {fhvhv_cols}")

print("\nSchema:")
for col in fhvhv.columns:
    dtype = str(fhvhv[col].dtype)
    nulls = fhvhv[col].isna().sum()
    print(f"  {col:<25} | {dtype:<20} | nulls: {nulls:>8,}")

print("\nNota: hvfhs_license_num non e' stato caricato per risparmiare memoria.")
print("Per analisi per compagnia, ricaricare con quella colonna aggiunta.")

# HVFHV temporale
fhvhv['pickup_hour'] = pd.to_datetime(fhvhv['pickup_datetime']).dt.hour
fhvhv['pickup_dow'] = pd.to_datetime(fhvhv['pickup_datetime']).dt.dayofweek

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
hourly_h = fhvhv['pickup_hour'].value_counts().sort_index()
sns.barplot(x=hourly_h.index, y=hourly_h.values, ax=axes[0], color='#8A2BE2', edgecolor='black')
axes[0].set_title('HVFHV — Corse per Ora', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(24))

dow_h = fhvhv['pickup_dow'].value_counts().sort_index()
dow_h.index = [dow_names[i] for i in dow_h.index]
sns.barplot(x=dow_h.index, y=dow_h.values, ax=axes[1], color='#8A2BE2', edgecolor='black')
axes[1].set_title('HVFHV — Corse per Giorno', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_fhvhv_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 05_fhvhv_temporal.png]")

# HVFHV durata e distanza
fhvhv['trip_duration_min'] = fhvhv['trip_time'] / 60

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
dur = fhvhv[fhvhv['trip_duration_min'] <= 60]['trip_duration_min']
sns.histplot(dur, bins=60, ax=axes[0], color='#8A2BE2', edgecolor='black')
axes[0].set_title('HVFHV — Durata Corse (0-60 min)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Durata (minuti)')
axes[0].axvline(dur.median(), color='red', linestyle='--', label=f'Mediana: {dur.median():.1f} min')
axes[0].legend()

miles = fhvhv[fhvhv['trip_miles'] <= 30]['trip_miles']
sns.histplot(miles, bins=60, ax=axes[1], color='#8A2BE2', edgecolor='black')
axes[1].set_title('HVFHV — Distanza Corse (0-30 mi)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Distanza (miglia)')
axes[1].axvline(miles.median(), color='red', linestyle='--', label=f'Mediana: {miles.median():.1f} mi')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_fhvhv_dur_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 06_fhvhv_dur_dist.png]")

# --- RIEPILOGO COMPARATIVO ---
print("\n--- Riepilogo Comparativo ---")
summary = pd.DataFrame({
    'Dataset': ['Yellow', 'Green', 'FHV', 'HVFHV'],
    'Righe': [len(yellow), len(green), len(fhv), len(fhvhv)],
    'Colonne': [len(yellow.columns), len(green.columns), len(fhv.columns), len(fhvhv.columns)],
    'Location_NULL_%': [
        yellow['PULocationID'].isna().sum() / len(yellow) * 100,
        green['PULocationID'].isna().sum() / len(green) * 100,
        fhv['PUlocationID'].isna().sum() / len(fhv) * 100,
        fhvhv['PULocationID'].isna().sum() / len(fhvhv) * 100
    ]
})
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(data=summary, x='Dataset', y='Righe', ax=axes[0],
            palette=['#FFD700', '#3CB371', '#FF6347', '#8A2BE2'])
axes[0].set_title('Volume Dati per Dataset', fontsize=14, fontweight='bold')
for i, v in enumerate(summary['Righe']):
    axes[0].text(i, v + max(summary['Righe'])*0.02, f'{v:,}', ha='center', fontweight='bold', fontsize=9)

sns.barplot(data=summary, x='Dataset', y='Location_NULL_%', ax=axes[1],
            palette=['#FFD700', '#3CB371', '#FF6347', '#8A2BE2'])
axes[1].set_title('% Location Mancanti', fontsize=14, fontweight='bold')
for i, v in enumerate(summary['Location_NULL_%']):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 07_comparison.png]")

# =============================================================================
# 3. PULIZIA E STANDARDIZZAZIONE
# =============================================================================

print("\n" + "=" * 70)
print("3. PULIZIA E STANDARDIZZAZIONE")
print("=" * 70)


def clean_dataset(df, dataset_type):
    """
    Pulisce e standardizza un dataset TLC.

    Standardizza i nomi delle colonne e filtra:
    - PULocationID nel range 1-265
    - Durata corsa tra 0 e 24 ore

    Args:
        df: DataFrame grezzo
        dataset_type: 'yellow', 'green', 'fhv', o 'fhvhv'

    Returns:
        DataFrame con colonne: pickup_datetime, PULocationID, taxi_type, trip_duration_sec
    """
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
    # fhvhv ha gia' i nomi corretti

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

    # Mantieni solo le colonne necessarie
    return df[['pickup_datetime', 'PULocationID', 'taxi_type', 'trip_duration_sec']].copy()


print("\nPulizia in corso...\n")
yellow_c = clean_dataset(yellow, 'yellow')
green_c = clean_dataset(green, 'green')
fhv_c = clean_dataset(fhv, 'fhv')
fhvhv_c = clean_dataset(fhvhv, 'fhvhv')

# Liberiamo memoria dai dataset grezzi
del yellow, green, fhv, fhvhv

total_clean = len(yellow_c) + len(green_c) + len(fhv_c) + len(fhvhv_c)
print(f"\nTOTALE pulito: {total_clean:,} righe")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("4. FEATURE ENGINEERING")
print("=" * 70)


def add_features(df):
    """
    Aggiunge feature temporali e geografiche a un dataset pulito.

    Feature create:
    - hour, minute, half_hour_bucket (0-47)
    - day_of_week (0=Lun, 6=Dom)
    - month
    - is_weekend, is_rush_hour, is_night
    - trip_duration_min
    - borough, service_zone, zone_name (da JOIN con zone lookup)
    """
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Feature temporali
    df['hour'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute
    # Fascia di 30 minuti: 0 = 00:00-00:29, 47 = 23:30-23:59
    df['half_hour_bucket'] = df['hour'] * 2 + (df['minute'] // 30)
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # 0=Lun, 6=Dom
    df['month'] = df['pickup_datetime'].dt.month

    # Feature booleane
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    df['is_night'] = (df['hour'] >= 22) | (df['hour'] < 5)

    # Durata in minuti
    df['trip_duration_min'] = df['trip_duration_sec'] / 60

    # JOIN con zone lookup per borough e service_zone
    zone_info = zones.set_index('LocationID')
    df = df.join(zone_info, on='PULocationID', how='left')
    df = df.rename(columns={'Zone': 'zone_name', 'Borough': 'borough'})

    return df


print("\nFeature engineering in corso...")
yellow_f = add_features(yellow_c)
green_f = add_features(green_c)
fhv_f = add_features(fhv_c)
fhvhv_f = add_features(fhvhv_c)

print(f"Completato. Colonne per dataset: {list(yellow_f.columns)}")
print(f"  Yellow: {len(yellow_f):,} righe")
print(f"  Green:  {len(green_f):,} righe")
print(f"  FHV:    {len(fhv_f):,} righe")
print(f"  HVFHV:  {len(fhvhv_f):,} righe")

# Esempio di riga con feature
print("\nEsempio riga Yellow con feature:")
print(yellow_f.head(1).T.to_string())

# =============================================================================
# 5. UNIONE DEI DATASET
# =============================================================================

print("\n" + "=" * 70)
print("5. UNIONE DEI DATASET")
print("=" * 70)

cols = [
    'pickup_datetime', 'PULocationID', 'taxi_type', 'trip_duration_sec',
    'hour', 'minute', 'half_hour_bucket', 'day_of_week', 'month',
    'is_weekend', 'is_rush_hour', 'is_night', 'trip_duration_min',
    'borough', 'service_zone', 'zone_name'
]

all_trips = pd.concat([
    yellow_f[cols], green_f[cols], fhv_f[cols], fhvhv_f[cols]
], ignore_index=True)

print(f"Dataset unito: {len(all_trips):,} righe x {len(all_trips.columns)} colonne")
print(f"Memoria: {all_trips.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\nDistribuzione per Tipo di Taxi:")
type_dist = all_trips['taxi_type'].value_counts()
for t, c in type_dist.items():
    pct = c / len(all_trips) * 100
    print(f"  {t:>8}: {c:>10,} ({pct:>5.1f}%)")

# Visualizzazione unione
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Pie chart
colors_type = {'yellow': '#FFD700', 'green': '#3CB371', 'fhv': '#FF6347', 'fhvhv': '#8A2BE2'}
axes[0].pie(type_dist.values, labels=[t.upper() for t in type_dist.index],
            autopct='%1.1f%%', colors=[colors_type[t] for t in type_dist.index])
axes[0].set_title('Distribuzione per Tipo', fontsize=14, fontweight='bold')

# Mix per ora
hourly_type = all_trips.groupby(['taxi_type', 'hour']).size().unstack(fill_value=0)
hourly_type = hourly_type.div(hourly_type.sum(axis=1), axis=0) * 100
hourly_type.T.plot(kind='bar', stacked=True, ax=axes[1], color=colors_type, width=0.8)
axes[1].set_title('Mix Taxi per Ora (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Ora')
axes[1].set_ylabel('%')
axes[1].legend(title='Tipo')

# Borough
borough_dist = all_trips[all_trips['borough'] != 'N/A']['borough'].value_counts()
sns.barplot(x=borough_dist.index, y=borough_dist.values, ax=axes[2], palette='Set2')
axes[2].set_title('Corse per Borough', fontsize=14, fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 08_combined.png]")

# =============================================================================
# 6. AGGREGAZIONE
# =============================================================================

print("\n" + "=" * 70)
print("6. AGGREGAZIONE")
print("=" * 70)

print("Aggregazione per (zona, fascia 30min, giorno, mese)...")

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
zone_info = zones.set_index('LocationID')
aggregated = aggregated.join(zone_info, on='PULocationID', how='left')
aggregated = aggregated.rename(columns={'Zone': 'zone_name', 'Borough': 'borough'})

# Feature aggiuntive
aggregated['is_weekend'] = aggregated['day_of_week'] >= 5
aggregated['is_rush_hour'] = aggregated['half_hour_bucket'].apply(
    lambda b: (b // 2) in [7, 8, 9, 17, 18, 19]
)
aggregated['is_night'] = aggregated['half_hour_bucket'].apply(
    lambda b: (b // 2) >= 22 or (b // 2) < 5
)

# Statistiche copertura
total_possible = 265 * 48 * 7 * 1  # zone x fasce x giorni x mesi (gennaio)
coverage = len(aggregated) / total_possible * 100

print(f"Dataset aggregato: {len(aggregated):,} combinazioni x {len(aggregated.columns)} colonne")
print(f"Copertura spazio feature: {coverage:.1f}%")
print(f"Combinazioni vuote (trip_count=0): {total_possible - len(aggregated):,} ({100-coverage:.1f}%)")

print("\nStatistiche descrittive:")
print(aggregated.describe().to_string())

# Distribuzione trip_count
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(aggregated['trip_count'], bins=100, ax=axes[0], color='#2196F3', edgecolor='black')
axes[0].set_title('Distribuzione Trip Count (lineare)', fontsize=14, fontweight='bold')
axes[0].axvline(aggregated['trip_count'].median(), color='red', linestyle='--',
                label=f"Mediana: {aggregated['trip_count'].median():.0f}")
axes[0].axvline(aggregated['trip_count'].mean(), color='green', linestyle='--',
                label=f"Media: {aggregated['trip_count'].mean():.0f}")
axes[0].legend()

sns.histplot(aggregated['trip_count'], bins=100, ax=axes[1], color='#2196F3', edgecolor='black')
axes[1].set_yscale('log')
axes[1].set_title('Distribuzione Trip Count (log)', fontsize=14, fontweight='bold')

sz_data = aggregated[aggregated['service_zone'] != 'N/A']
sns.boxplot(data=sz_data, x='service_zone', y='trip_count', ax=axes[2], palette='Set2')
axes[2].set_title('Trip Count per Tipo di Zona', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_trip_count_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 09_trip_count_dist.png]")

# =============================================================================
# 7. TARGET: AVAILABILITY INDEX
# =============================================================================

print("\n" + "=" * 70)
print("7. TARGET: AVAILABILITY INDEX")
print("=" * 70)

print("""
Concetto: availability_index misura quanto e' "facile" trovare un taxi
in una zona/ora/giorno, RELATIVAMENTE al picco storico di quella zona.

Formula: availability_index = trip_count(zona, ora, giorno) / max_trip_count(zona)

Perche' normalizzare per zona?
- Midtown Center puo' avere 5000 corse in una fascia
- Jamaica Bay ne ha al massimo 50
- 45 corse per Jamaica Bay e' ottimo (90% del picco)
- 45 corse per Midtown e' bassissimo (0.9% del picco)
""")

# Calcolo max trip_count per zona
max_per_zone = aggregated.groupby('PULocationID')['trip_count'].max().rename('max_trip_count_zone')
aggregated = aggregated.join(max_per_zone, on='PULocationID')

# Availability index
aggregated['availability_index'] = aggregated['trip_count'] / aggregated['max_trip_count_zone']

# Classi di disponibilita'
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['Molto Difficile', 'Difficile', 'Medio', 'Facile', 'Molto Facile']
aggregated['availability_class'] = pd.cut(
    aggregated['availability_index'],
    bins=bins,
    labels=labels,
    include_lowest=True
)
class_mapping = {label: i for i, label in enumerate(labels)}
aggregated['availability_class_id'] = aggregated['availability_class'].map(class_mapping)

print("Availability Index e classi create!\n")
print("Distribuzione delle Classi di Disponibilita':")
class_dist = aggregated['availability_class'].value_counts().sort_index()
for cls, count in class_dist.items():
    pct = count / len(aggregated) * 100
    bar = '#' * int(pct / 2)
    print(f"  {cls:>15}: {count:>6,} ({pct:>5.1f}%) {bar}")

# Visualizzazione classi
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(aggregated['availability_index'], bins=50, ax=axes[0], color='#4CAF50', edgecolor='black')
axes[0].set_title('Distribuzione Availability Index', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Availability Index (0-1)')
for b in [0.2, 0.4, 0.6, 0.8]:
    axes[0].axvline(b, color='gray', linestyle=':', alpha=0.7)

class_counts = aggregated['availability_class'].value_counts().sort_index()
class_colors = ['#D32F2F', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[1], palette=class_colors)
axes[1].set_title('Distribuzione Classi', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)
for i, (cls, count) in enumerate(class_counts.items()):
    axes[1].text(i, count + max(class_counts.values)*0.01, f'{count:,}', ha='center', fontweight='bold', fontsize=9)

axes[2].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=class_colors)
axes[2].set_title('Proporzione Classi', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_availability.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 10_availability.png]")

# =============================================================================
# 8. EDA SUL DATASET AGGREGATO
# =============================================================================

print("\n" + "=" * 70)
print("8. EDA SUL DATASET AGGREGATO")
print("=" * 70)

# --- Pattern orari ---
print("\n--- Pattern Orari ---")
hourly_avail = aggregated.groupby('half_hour_bucket').agg(
    avg_avail=('availability_index', 'mean'),
    total_trips=('trip_count', 'sum')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.lineplot(data=hourly_avail, x='half_hour_bucket', y='avg_avail',
             ax=axes[0], color='#4CAF50', marker='o', markersize=4)
axes[0].set_title('Disponibilita' + ' Media per Fascia Oraria', fontsize=14, fontweight='bold')
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
plt.savefig(OUTPUT_DIR / '11_hourly_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 11_hourly_patterns.png]")

peak_avail_idx = hourly_avail['avg_avail'].idxmax()
peak_trips_idx = hourly_avail['total_trips'].idxmax()
print(f"Fascia con disponibilita' media piu' alta: {hourly_avail.loc[peak_avail_idx, 'half_hour_bucket']//2:02d}:{'00' if hourly_avail.loc[peak_avail_idx, 'half_hour_bucket']%2==0 else '30'}")
print(f"Fascia con piu' corse totali: {hourly_avail.loc[peak_trips_idx, 'half_hour_bucket']//2:02d}:{'00' if hourly_avail.loc[peak_trips_idx, 'half_hour_bucket']%2==0 else '30'}")

# --- Pattern per giorno ---
print("\n--- Pattern per Giorno ---")
dow_names_full = ['Lunedi', 'Martedi', 'Mercoledi', 'Giovedi', 'Venerdi', 'Sabato', 'Domenica']
dow_avail = aggregated.groupby('day_of_week').agg(
    avg_avail=('availability_index', 'mean'),
    total_trips=('trip_count', 'sum')
).reset_index()
dow_avail['day_name'] = dow_avail['day_of_week'].map(dict(enumerate(dow_names_full)))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(data=dow_avail, x='day_name', y='avg_avail', ax=axes[0], palette='Greens')
axes[0].set_title('Disponibilita' + ' Media per Giorno', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(data=dow_avail, x='day_name', y='total_trips', ax=axes[1], palette='Blues')
axes[1].set_title('Trip Count Totale per Giorno', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '12_daily_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 12_daily_patterns.png]")

# --- Pattern per Borough ---
print("\n--- Pattern per Borough ---")
borough_avail = aggregated[aggregated['borough'] != 'N/A'].groupby('borough').agg(
    avg_avail=('availability_index', 'mean'),
    total_trips=('trip_count', 'sum'),
    avg_trip_count=('trip_count', 'mean'),
    num_zones=('PULocationID', 'nunique')
).reset_index().sort_values('total_trips', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.barplot(data=borough_avail, x='borough', y='avg_avail', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Disponibilita' + ' Media per Borough', fontsize=14, fontweight='bold')
sns.barplot(data=borough_avail, x='borough', y='total_trips', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Trip Count Totale per Borough', fontsize=14, fontweight='bold')
sns.barplot(data=borough_avail, x='borough', y='avg_trip_count', ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Trip Count Medio per Combinazione', fontsize=14, fontweight='bold')
sns.barplot(data=borough_avail, x='borough', y='num_zones', ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Zone Attive per Borough', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '13_borough_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 13_borough_patterns.png]")
print("\nStatistiche per Borough:")
print(borough_avail.to_string(index=False))

# --- Heatmap: Giorno x Ora ---
print("\n--- Heatmap: Giorno x Ora ---")
dow_hour = aggregated.groupby(['day_of_week', 'half_hour_bucket'])['availability_index'].mean().unstack(fill_value=0)
dow_hour.index = [dow_names_full[i] for i in dow_hour.index]

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(dow_hour, cmap='RdYlGn', ax=ax, vmin=0, vmax=1,
            cbar_kws={'label': 'Availability Index Medio'})
ax.set_title('Disponibilita' + ': Giorno x Fascia Oraria', fontsize=14, fontweight='bold')
ax.set_xlabel('Fascia Oraria (0-47)')
ax.set_ylabel('Giorno della Settimana')
ax.set_xticks(range(0, 48, 4))
ax.set_xticklabels([f"{i//2:02d}:{'00' if i%2==0 else '30'}" for i in range(0, 48, 4)], rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '14_heatmap_dow_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 14_heatmap_dow_hour.png]")

# --- Analisi zone top/bottom ---
print("\n--- Zone piu' attive e disponibilita' ---")
zone_avail = aggregated.groupby('PULocationID').agg(
    avg_avail=('availability_index', 'mean'),
    max_avail=('availability_index', 'max'),
    total_trips=('trip_count', 'sum'),
    active_slots=('trip_count', lambda x: (x > 0).sum()),
).join(zones.set_index('LocationID')[['Borough', 'Zone', 'service_zone']], on='PULocationID')

print("\nTop 10 Zone piu' Attive:")
top_active = zone_avail.sort_values('active_slots', ascending=False).head(10)
print(top_active[['Zone', 'Borough', 'service_zone', 'active_slots', 'avg_avail', 'total_trips']].to_string(index=False))

print("\nTop 10 Zone con Disponibilita' Media piu' Alta (min 50 slot attivi):")
top_avail = zone_avail[zone_avail['active_slots'] > 50].sort_values('avg_avail', ascending=False).head(10)
print(top_avail[['Zone', 'Borough', 'service_zone', 'active_slots', 'avg_avail', 'total_trips']].to_string(index=False))

print("\nTop 10 Zone con Disponibilita' Media piu' Bassa (min 50 slot attivi):")
low_avail = zone_avail[zone_avail['active_slots'] > 50].sort_values('avg_avail', ascending=True).head(10)
print(low_avail[['Zone', 'Borough', 'service_zone', 'active_slots', 'avg_avail', 'total_trips']].to_string(index=False))

# --- Correlazioni ---
print("\n--- Correlazioni ---")
corr_data = aggregated[[
    'PULocationID', 'half_hour_bucket', 'day_of_week', 'month',
    'trip_count', 'unique_taxi_types', 'avg_trip_duration_min',
    'availability_index', 'availability_class_id'
]].copy()
corr_data['is_weekend'] = aggregated['is_weekend'].astype(int)
corr_data['is_rush_hour'] = aggregated['is_rush_hour'].astype(int)
corr_data['is_night'] = aggregated['is_night'].astype(int)

print("\nCorrelazioni con availability_index:")
avail_corr = corr_data.corr()['availability_index'].sort_values(ascending=False)
for feat, corr in avail_corr.items():
    if feat != 'availability_index':
        strength = 'FORTE' if abs(corr) > 0.5 else 'MODERATA' if abs(corr) > 0.3 else 'DEBOLE'
        print(f"  [{strength:>8}] {feat:>30}: {corr:+.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Matrice di Correlazione', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '15_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 15_correlations.png]")

# --- Distribuzione unique_taxi_types ---
print("\n--- Tipi di Taxi per Combinazione ---")
type_counts = aggregated['unique_taxi_types'].value_counts().sort_index()
for n_types, count in type_counts.items():
    pct = count / len(aggregated) * 100
    bar = '#' * int(pct / 2)
    print(f"  {n_types} tipo/i: {count:>7,} ({pct:>5.1f}%) {bar}")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=type_counts.index.astype(str), y=type_counts.values, ax=ax,
            palette='Set2', edgecolor='black')
ax.set_title('Tipi di Taxi per Combinazione (Zona, Ora, Giorno)', fontsize=14, fontweight='bold')
ax.set_xlabel('Tipi di Taxi Presenti')
ax.set_ylabel('Numero Combinazioni')
for i, (n, c) in enumerate(type_counts.items()):
    ax.text(i, c + max(type_counts.values)*0.02, f'{c:,}', ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '16_taxi_types_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 16_taxi_types_dist.png]")

# --- Durata media per fascia oraria ---
print("\n--- Durata Media Corse per Fascia Oraria ---")
hourly_duration = aggregated.groupby('half_hour_bucket').agg(
    avg_duration=('avg_trip_duration_min', 'mean'),
    median_duration=('median_trip_duration_min', 'median')
).reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=hourly_duration, x='half_hour_bucket', y='avg_duration',
             ax=ax, color='#FF5722', linewidth=2, marker='o', markersize=4, label='Media')
sns.lineplot(data=hourly_duration, x='half_hour_bucket', y='median_duration',
             ax=ax, color='#2196F3', linewidth=2, marker='s', markersize=4, label='Mediana')
ax.set_title('Durata Media Corse per Fascia Oraria', fontsize=14, fontweight='bold')
ax.set_xlabel('Fascia Oraria (0-47)')
ax.set_ylabel('Durata (minuti)')
ax.set_xticks(range(0, 48, 4))
ax.set_xticklabels([f"{i//2:02d}:{'00' if i%2==0 else '30'}" for i in range(0, 48, 4)], rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '17_duration_hourly.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Figura salvata: 17_duration_hourly.png]")
print(f"Durata media: {aggregated['avg_trip_duration_min'].mean():.1f} min")
print(f"Durata mediana: {aggregated['median_trip_duration_min'].median():.1f} min")

# =============================================================================
# 9. PREPARAZIONE DATASET FINALE PER IL MODELLO
# =============================================================================

print("\n" + "=" * 70)
print("9. PREPARAZIONE DATASET FINALE PER IL MODELLO")
print("=" * 70)

from sklearn.preprocessing import LabelEncoder

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

model_df = aggregated[model_features].copy()

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
print(f"\nFeature finali:")
for col in model_df.columns:
    dtype = str(model_df[col].dtype)
    nulls = model_df[col].isna().sum()
    unique = model_df[col].nunique()
    print(f"  {col:<25} | {dtype:<15} | nulls: {nulls:>5} | unique: {unique:>5}")

# Riepilogo finale
print("\n" + "=" * 70)
print("RIEPILOGO FINALE")
print("=" * 70)
print(f"\nDataset grezzo originale:")
print(f"   Yellow:  {len(yellow_c):>10,} righe (dopo pulizia)")
print(f"   Green:   {len(green_c):>10,} righe (dopo pulizia)")
print(f"   FHV:     {len(fhv_c):>10,} righe (dopo pulizia)")
print(f"   HVFHV:   {len(fhvhv_c):>10,} righe (dopo pulizia + campionamento {SAMPLE_FHVHV*100:.0f}%)")
print(f"   TOTALE:  {total_clean:>10,} righe")
print(f"\nDopo aggregazione:")
print(f"   TOTALE:  {len(aggregated):>10,} combinazioni")
print(f"\nTarget:")
print(f"   availability_index: continuo [0, 1]")
print(f"   availability_class: 5 classi (Molto Difficile -> Molto Facile)")
print(f"\nFeature per il modello:")
print(f"   Numeriche: PULocationID, half_hour_bucket, day_of_week, month,")
print(f"              unique_taxi_types, avg_trip_duration_min")
print(f"   Booleane:  is_weekend, is_rush_hour, is_night")
print(f"   Categoricali: borough, service_zone (label encoded)")
print(f"\nMemoria dataset modello: {model_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# =============================================================================
# 10. SALVATAGGIO
# =============================================================================

print("\n" + "=" * 70)
print("10. SALVATAGGIO")
print("=" * 70)

# Dataset aggregato completo
aggregated.to_csv(OUTPUT_DIR / 'aggregated_dataset_jan2025.csv', index=False)
aggregated.to_parquet(OUTPUT_DIR / 'aggregated_dataset_jan2025.parquet', index=False)
print(f"Dataset aggregato salvato:")
print(f"  CSV:     {OUTPUT_DIR / 'aggregated_dataset_jan2025.csv'}")
print(f"  Parquet: {OUTPUT_DIR / 'aggregated_dataset_jan2025.parquet'}")

# Dataset pronto per il modello
model_df.to_csv(OUTPUT_DIR / 'model_ready_jan2025.csv', index=False)
model_df.to_parquet(OUTPUT_DIR / 'model_ready_jan2025.parquet', index=False)
print(f"\nDataset per il modello salvato:")
print(f"  CSV:     {OUTPUT_DIR / 'model_ready_jan2025.csv'}")
print(f"  Parquet: {OUTPUT_DIR / 'model_ready_jan2025.parquet'}")

# Label encoder
joblib.dump(le_borough, OUTPUT_DIR / 'le_borough.pkl')
joblib.dump(le_service, OUTPUT_DIR / 'le_service.pkl')
print(f"\nLabel encoder salvati:")
print(f"  {OUTPUT_DIR / 'le_borough.pkl'}")
print(f"  {OUTPUT_DIR / 'le_service.pkl'}")

print("\n" + "=" * 70)
print("DATASET PRONTO PER IL TRAINING DEL MODELLO!")
print("=" * 70)

# =============================================================================
# 11. FUNZIONE RIUTILIZZABILE PER MESI SUCCESSIVI
# =============================================================================

print("\n" + "=" * 70)
print("11. FUNZIONE RIUTILIZZABILE")
print("=" * 70)


def process_month(yellow_path, green_path, fhv_path, fhvhv_path, month_label="",
                  sample_fhvhv=SAMPLE_FHVHV):
    """
    Processa un mese completo di dati TLC e restituisce il dataset aggregato.

    Args:
        yellow_path: percorso file yellow_tripdata_YYYY-MM.parquet
        green_path: percorso file green_tripdata_YYYY-MM.parquet
        fhv_path: percorso file fhv_tripdata_YYYY-MM.parquet
        fhvhv_path: percorso file fhvhv_tripdata_YYYY-MM.parquet
        month_label: etichetta per il logging (es. 'Febbraio 2025')
        sample_fhvhv: frazione di HVFHV da campionare (0.1 = 10%)

    Returns:
        aggregated: DataFrame aggregato con availability_index e classi
        model_df: DataFrame pronto per il modello
    """
    print(f"\n{'='*60}")
    print(f"Processing: {month_label or 'Mese sconosciuto'}")
    print(f"{'='*60}")

    # Caricamento
    y = pd.read_parquet(yellow_path)
    g = pd.read_parquet(green_path)
    f = pd.read_parquet(fhv_path)

    # HVFHV con selezione colonne per memoria
    fhvhv_cols = ['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_time', 'trip_miles']
    h_table = pq.read_table(fhvhv_path, columns=fhvhv_cols)
    h = h_table.to_pandas()
    if sample_fhvhv < 1.0:
        h = h.sample(frac=sample_fhvhv, random_state=42).reset_index(drop=True)

    # Pulizia
    y_c = clean_dataset(y, 'yellow')
    g_c = clean_dataset(g, 'green')
    f_c = clean_dataset(f, 'fhv')
    h_c = clean_dataset(h, 'fhvhv')

    # Feature engineering
    y_f = add_features(y_c)
    g_f = add_features(g_c)
    f_f = add_features(f_c)
    h_f = add_features(h_c)

    # Unione
    cols = ['pickup_datetime', 'PULocationID', 'taxi_type', 'trip_duration_sec',
            'hour', 'minute', 'half_hour_bucket', 'day_of_week', 'month',
            'is_weekend', 'is_rush_hour', 'is_night', 'trip_duration_min',
            'borough', 'service_zone', 'zone_name']

    all_t = pd.concat([y_f[cols], g_f[cols], f_f[cols], h_f[cols]], ignore_index=True)

    # Aggregazione
    agg = all_t.groupby(
        ['PULocationID', 'half_hour_bucket', 'day_of_week', 'month']
    ).agg(
        trip_count=('pickup_datetime', 'count'),
        unique_taxi_types=('taxi_type', 'nunique'),
        avg_trip_duration_min=('trip_duration_min', 'mean'),
        median_trip_duration_min=('trip_duration_min', 'median'),
        std_trip_duration_min=('trip_duration_min', 'std'),
    ).reset_index()

    zone_info = zones.set_index('LocationID')
    agg = agg.join(zone_info, on='PULocationID', how='left')
    agg = agg.rename(columns={'Zone': 'zone_name', 'Borough': 'borough'})

    agg['is_weekend'] = agg['day_of_week'] >= 5
    agg['is_rush_hour'] = agg['half_hour_bucket'].apply(
        lambda b: (b // 2) in [7, 8, 9, 17, 18, 19]
    )
    agg['is_night'] = agg['half_hour_bucket'].apply(
        lambda b: (b // 2) >= 22 or (b // 2) < 5
    )

    # Availability index
    max_z = agg.groupby('PULocationID')['trip_count'].max().rename('max_trip_count_zone')
    agg = agg.join(max_z, on='PULocationID')
    agg['availability_index'] = agg['trip_count'] / agg['max_trip_count_zone']

    agg['availability_class'] = pd.cut(
        agg['availability_index'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Molto Difficile', 'Difficile', 'Medio', 'Facile', 'Molto Facile'],
        include_lowest=True
    )
    agg['availability_class_id'] = agg['availability_class'].map(class_mapping)

    # Dataset modello
    mdf = agg[[
        'PULocationID', 'half_hour_bucket', 'day_of_week', 'month',
        'unique_taxi_types', 'avg_trip_duration_min',
        'is_weekend', 'is_rush_hour', 'is_night',
        'borough', 'service_zone',
        'availability_class_id', 'availability_index'
    ]].copy()

    mdf['is_weekend'] = mdf['is_weekend'].astype(int)
    mdf['is_rush_hour'] = mdf['is_rush_hour'].astype(int)
    mdf['is_night'] = mdf['is_night'].astype(int)
    mdf['borough_encoded'] = le_borough.transform(mdf['borough'].fillna('Unknown'))
    mdf['service_zone_encoded'] = le_service.transform(mdf['service_zone'].fillna('Unknown'))

    print(f"\n{month_label}: {len(agg):,} combinazioni, {len(all_t):,} corse grezze")

    return agg, mdf


print("Funzione process_month() definita.")
print("Utilizzo:")
print("  agg, mdf = process_month(")
print("      yellow_path, green_path, fhv_path, fhvhv_path,")
print("      month_label='Febbraio 2025'")
print("  )")
