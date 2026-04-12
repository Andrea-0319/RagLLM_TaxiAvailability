"""Generate notebook: LightGBM Advanced Training with Learning Curves."""
import json

def md(lines):
    raw = lines.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in raw]}

def code(text):
    raw = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in raw]}

cells = []

cells.append(md("""# NYC Taxi Demand Prediction - LightGBM Advanced Training

## Obiettivo
Addestramento avanzato del modello **LightGBM** sul dataset multi-mese (Gen-Dic 2025) con:
1. **GridSearchCV** con 27 combinazioni di hyperparametri
2. **Learning Curve** per diagnosticare overfitting/underfitting
3. **Explainable AI** con SHAP (globale) + LIME (locale)
4. **Test di verifica** con scenari reali

## Perche' solo LightGBM?
Dal confronto precedente (Notebook 02), LightGBM ha dimostrato:
- **Migliore generalizzazione**: Test F1-Macro 0.5865 vs 0.5402 di RandomForest
- **Meno overfitting**: Gap CV-Test piu' contenuto rispetto a RF
- **Scalabilita'**: Performance eccellente con dataset piu' grandi
- **Velocita'**: Training time comparabile a RF ma con risultati superiori

## Strategia
1. Caricamento del dataset multi-mese aggregato
2. Split stratificato 80/20 (train/test)
3. SMOTE per bilanciamento classi
4. Standardizzazione (fit su train pre-SMOTE)
5. **GridSearchCV** con 27 combinazioni (3x3x3) su tutti i dati SMOTE
6. **Learning Curve** con 5 punti di training size
7. Retraining del modello migliore sui dati completi
8. SHAP + LIME per explainability
9. Test case con scenari reali"""))

cells.append(md("## 1. Setup - Import delle Librerie"))

cells.append(code("""import warnings
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    learning_curve, StratifiedShuffleSplit
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 120

print("Tutte le librerie importate con successo!")"""))

cells.append(md("## 2. Caricamento del Dataset Multi-Mese"))

cells.append(md("""Il dataset e' stato generato dal notebook `03_multi_month_processing.ipynb`.
Ogni riga rappresenta una combinazione unica di **zona + fascia oraria (30 min) + giorno della settimana + mese**.

Il dataset include tutti i mesi disponibili del 2025, con l'**availability_index calcolato globalmente**
(il max trip_count per zona e' il picco assoluto su tutti i mesi combinati)."""))

cells.append(code("""DATA_PATH = r'output/model_ready_all_months.parquet'
df = pd.read_parquet(DATA_PATH)
print(f"Shape del dataset: {df.shape[0]:,} righe, {df.shape[1]} colonne")
print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"\\nColonne disponibili: {list(df.columns)}")
print(f"\\nPrime 5 righe:")
df.head()"""))

cells.append(md("### 2.1 Distribuzione delle Classi Target"))

cells.append(md("""Analizziamo la distribuzione delle classi sul dataset multi-mese.
Con piu' mesi di dati, ci aspettiamo una distribuzione potenzialmente piu' bilanciata
rispetto al singolo mese di Gennaio."""))

cells.append(code("""CLASS_NAMES = {0: 'Molto Difficile', 1: 'Difficile', 2: 'Medio', 3: 'Facile', 4: 'Molto Facile'}
CLASS_COLORS = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']

counts = df['availability_class_id'].value_counts().sort_index()
labels = [CLASS_NAMES[i] for i in counts.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(labels, counts.values, color=CLASS_COLORS, edgecolor='white')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.01,
                 f'{val:,}', ha='center', fontweight='bold')
axes[0].set_title('Distribuzione Classi - Conteggio', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

pcts = counts / len(df) * 100
axes[1].pie(pcts.values, labels=labels, colors=CLASS_COLORS, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Distribuzione Classi - Percentuali', fontweight='bold')
plt.tight_layout()
plt.savefig('output/14_multi_month_class_dist.png', bbox_inches='tight', dpi=150)
plt.show()

print("Distribuzione dettagliata:")
for cls_id in sorted(counts.index):
    pct = counts[cls_id] / len(df) * 100
    print(f"  {CLASS_NAMES[cls_id]:20s}: {counts[cls_id]:,} ({pct:.1f}%)")"""))

cells.append(md("## 3. Preparazione dei Dati - Train/Test Split"))

cells.append(md("""Split stratificato 80/20. La stratificazione garantisce che la proporzione
delle classi nel train e nel test sia identica a quella del dataset originale.

**Nota**: Con GridSearchCV e 3-fold CV, la validazione e' gestita internamente.
Non serve un validation set separato."""))

cells.append(code("""FEATURE_COLS = [
    'PULocationID', 'half_hour_bucket', 'day_of_week', 'month',
    'unique_taxi_types', 'avg_trip_duration_min',
    'is_weekend', 'is_rush_hour', 'is_night',
    'borough_encoded', 'service_zone_encoded'
]

X = df[FEATURE_COLS].copy()
y = df['availability_class_id'].copy()

# Split stratificato: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} campioni ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]:,} campioni ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"Feature:      {X_train.shape[1]}")
print(f"Classi:       {y_train.nunique()}")"""))

cells.append(md("## 4. SMOTE - Synthetic Minority Over-sampling Technique"))

cells.append(md("""SMOTE crea esempi sintetici delle classi minoritarie interpolando tra i k-vicini
piu' prossimi nello spazio delle feature. Applicato **solo sul training set**."""))

cells.append(code("""print("=== Distribuzione PRIMA di SMOTE ===")
for cls_id in sorted(y_train.unique()):
    cnt = (y_train == cls_id).sum()
    pct = cnt / len(y_train) * 100
    print(f"  {CLASS_NAMES[cls_id]:20s}: {cnt:,} ({pct:.1f}%)")

# Calcolo k_neighbors: non puo' superare (min_class_count - 1)
min_class_count = y_train.value_counts().min()
k_neighbors = min(5, min_class_count - 1)
print(f"\\nk_neighbors per SMOTE: {k_neighbors}")

# Applicazione SMOTE
print(f"\\nApplicazione SMOTE in corso...")
t0 = time.time()
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
smote_time = time.time() - t0
print(f"SMOTE completato in {smote_time:.1f}s")

print(f"\\n=== Distribuzione DOPO SMOTE ===")
for cls_id in sorted(y_train_smote.unique()):
    cnt = (y_train_smote == cls_id).sum()
    pct = cnt / len(y_train_smote) * 100
    print(f"  {CLASS_NAMES[cls_id]:20s}: {cnt:,} ({pct:.1f}%)")

print(f"\\nEspansione: {len(y_train):,} -> {len(y_train_smote):,} campioni (+{(len(y_train_smote)/len(y_train)-1)*100:.0f}%)")"""))

cells.append(md("## 5. Standardizzazione delle Feature"))

cells.append(md("""Standardizzazione Z-score. Fit sul training originale (pre-SMOTE) per evitare data leakage."""))

cells.append(code("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_smote_scaled = scaler.transform(X_train_smote)

print(f"Shape training scalato: {X_train_scaled.shape}")
print(f"Shape test scalato:     {X_test_scaled.shape}")
print(f"Shape SMOTE scalato:    {X_train_smote_scaled.shape}")
print(f"\\nMedia delle feature (dovrebbe essere ~0): {X_train_scaled.mean(axis=0).round(2)}")
print(f"Std delle feature (dovrebbe essere ~1):   {X_train_scaled.std(axis=0).round(2)}")"""))

cells.append(md("## 6. GridSearchCV - LightGBM"))

cells.append(md("""### Strategia di Ricerca
**27 combinazioni** (3x3x3) x **3-fold** = **81 fit totali** su tutti i dati SMOTE.

### Hyperparametri

| Parametro | Valori | Descrizione |
|-----------|--------|-------------|
| `n_estimators` | [200, 300, 400] | Numero di boosting round |
| `learning_rate` | [0.05, 0.1, 0.15] | Tasso di apprendimento (shrinkage) |
| `num_leaves` | [63, 127, 255] | Foglie massime per albero (complessita') |

**Metrica**: F1-Macro (peso uguale a tutte le 5 classi)
**CV**: 3-fold stratificato"""))

cells.append(code("""# Configurazione Cross Validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print(f"Cross Validation: {cv.get_n_splits()} fold stratificati")
print(f"Dati per GridSearchCV: {X_train_smote_scaled.shape[0]:,} campioni (100% dei dati SMOTE)")
print(f"Combinazioni: 27 | Fit totali: 81")"""))

cells.append(code("""param_lgbm = {
    'n_estimators': [200, 300, 400],    # 200 (base) vs 300 (medio) vs 400 (massimo)
    'learning_rate': [0.05, 0.1, 0.15], # 0.05 (cauto) vs 0.1 (standard) vs 0.15 (aggressivo)
    'num_leaves': [63, 127, 255]        # 63 (conservativo) vs 127 (medio) vs 255 (espressivo)
}

gs_lgbm = GridSearchCV(
    lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    ),
    param_lgbm,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

print(f"\\nAvvio GridSearchCV per LightGBM...")
print(f"Combinazioni: {len(gs_lgbm.param_grid['n_estimators'])} x {len(gs_lgbm.param_grid['learning_rate'])} x {len(gs_lgbm.param_grid['num_leaves'])} = {len(gs_lgbm.param_grid['n_estimators']) * len(gs_lgbm.param_grid['learning_rate']) * len(gs_lgbm.param_grid['num_leaves'])}")
print(f"Fit totali: 27 combinazioni x 3 folds = 81")
t0 = time.time()
gs_lgbm.fit(X_train_smote_scaled, y_train_smote)
elapsed = time.time() - t0
print(f"\\nCompletato in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Miglior F1-Macro (CV): {gs_lgbm.best_score_:.4f}")
print(f"Migliori hyperparametri: {gs_lgbm.best_params_}")"""))

cells.append(md("### 6.1 Analisi dei Risultati GridSearch"))

cells.append(md("""Visualizziamo i risultati di tutte le combinazioni testate per capire
quali hyperparametri funzionano meglio."""))

cells.append(code("""# Tutti i risultati del GridSearch
gs_results_df = pd.DataFrame(gs_lgbm.cv_results_)
gs_results_df = gs_results_df.sort_values('rank_test_score')

print("Top 10 combinazioni:")
top10 = gs_results_df[['rank_test_score', 'mean_test_score', 'std_test_score',
                        'param_n_estimators', 'param_learning_rate', 'param_num_leaves']].head(10)
for i, row in top10.iterrows():
    n_est = int(row['param_n_estimators'])
    lr = float(row['param_learning_rate'])
    leaves = int(row['param_num_leaves'])
    print(f"  #{int(row['rank_test_score']):2d} | F1={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f}) | "
          f"n_est={n_est:4d} | lr={lr:.2f} | leaves={leaves:4d}")"""))

cells.append(code("""# Heatmap: learning_rate x num_leaves per ogni n_estimators
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, n_est in enumerate([200, 300, 400]):
    subset = gs_results_df[gs_results_df['param_n_estimators'] == float(n_est)]
    pivot = subset.pivot_table(
        values='mean_test_score',
        index='param_learning_rate',
        columns='param_num_leaves'
    )

    im = axes[idx].imshow(pivot.values, cmap='viridis', aspect='auto', vmin=gs_results_df['mean_test_score'].min(), vmax=gs_results_df['mean_test_score'].max())
    axes[idx].set_xticks(range(len(pivot.columns)))
    axes[idx].set_xticklabels([str(c) for c in pivot.columns])
    axes[idx].set_yticks(range(len(pivot.index)))
    axes[idx].set_yticklabels([str(i) for i in pivot.index])
    axes[idx].set_xlabel('num_leaves')
    axes[idx].set_ylabel('learning_rate')
    axes[idx].set_title(f'n_estimators={n_est}', fontweight='bold')

    # Annota valori
    for ii in range(len(pivot.index)):
        for jj in range(len(pivot.columns)):
            axes[idx].text(jj, ii, f'{pivot.values[ii, jj]:.4f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if pivot.values[ii, jj] < 0.6 else 'black')

    plt.colorbar(im, ax=axes[idx], fraction=0.046, label='F1-Macro')

plt.suptitle('GridSearchCV - F1-Macro per Combinazione', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/15_gs_heatmap.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("## 7. Learning Curve - Diagnosi Overfitting/Underfitting"))

cells.append(md("""### Cos'e' una Learning Curve?
Una learning curve mostra l'andamento del punteggio di training e validation al crescere
dei dati di training. E' lo strumento principale per diagnosticare:

- **Overfitting (alta varianza)**: Gap ampio tra training e validation score.
  Il modello memorizza i dati di training ma non generalizza.
  *Soluzione*: piu' dati, regolarizzazione, meno feature.

- **Underfitting (alto bias)**: Entrambi i punteggi sono bassi.
  Il modello e' troppo semplice per catturare i pattern nei dati.
  *Soluzione*: modello piu' complesso, piu' feature, meno regolarizzazione.

- **Good fit**: Training e validation score convergono a un valore alto.

### Configurazione
Calcoliamo la learning curve su **5 livelli di training size** usando 3-fold CV.
Per velocita', usiamo il miglior modello trovato dal GridSearchCV."""))

cells.append(code("""# Best model dal GridSearch
best_model_gs = gs_lgbm.best_estimator_
print(f"Modello per learning curve: {best_model_gs}")
print(f"\\nCalcolo learning curve con 5 livelli di training size, 3-fold CV...")

# Livelli di training size
train_sizes = np.linspace(0.1, 1.0, 5)
print(f"Training sizes: {train_sizes}")

t0 = time.time()
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_model_gs,
    X_train_smote_scaled, y_train_smote,
    train_sizes=train_sizes,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)
lc_time = time.time() - t0
print(f"Learning curve calcolata in {lc_time:.0f}s ({lc_time/60:.1f} min)")"""))

cells.append(code("""# Plot della learning curve
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Training e Validation score
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

axes[0].plot(train_sizes_abs, train_mean, 'o-', color='#d32f2f', linewidth=2, markersize=8, label='Training F1-Macro')
axes[0].fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#d32f2f')
axes[0].plot(train_sizes_abs, val_mean, 'o-', color='#1976d2', linewidth=2, markersize=8, label='Validation F1-Macro')
axes[0].fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#1976d2')
axes[0].set_xlabel('Numero di campioni di training', fontweight='bold')
axes[0].set_ylabel('F1-Macro', fontweight='bold')
axes[0].set_title('Learning Curve - LightGBM', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1)

# Annota valori
for i, (tm, vm, ts) in enumerate(zip(train_mean, val_mean, train_sizes_abs)):
    axes[0].annotate(f'T={tm:.3f}', (ts, tm), textcoords="offset points",
                     xytext=(10, 10), fontsize=9, color='#d32f2f', fontweight='bold')
    axes[0].annotate(f'V={vm:.3f}', (ts, vm), textcoords="offset points",
                     xytext=(10, -15), fontsize=9, color='#1976d2', fontweight='bold')

# Plot 2: Gap Training-Validation
gap = train_mean - val_mean
gap_std = np.sqrt(train_std**2 + val_std**2)
axes[1].plot(train_sizes_abs, gap, 'o-', color='#f57c00', linewidth=2, markersize=8)
axes[1].fill_between(train_sizes_abs, gap - gap_std, gap + gap_std, alpha=0.15, color='#f57c00')
axes[1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Gap accettabile (0.05)')
axes[1].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Gap critico (0.10)')
axes[1].set_xlabel('Numero di campioni di training', fontweight='bold')
axes[1].set_ylabel('Gap (Train - Validation)', fontweight='bold')
axes[1].set_title('Gap Training-Validation', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Diagnosi
final_gap = gap[-1]
if final_gap > 0.10:
    diagnosis = "OVERFITTING - Il modello memorizza i dati di training"
elif final_gap > 0.05:
    diagnosis = "LEGGERO OVERFITTING - Marginale, accettabile"
elif train_mean[-1] < 0.7:
    diagnosis = "UNDERFITTING - Il modello e' troppo semplice"
else:
    diagnosis = "GOOD FIT - Il modello generalizza bene"

axes[1].text(0.05, 0.95, f'Diagnosi: {diagnosis}', transform=axes[1].transAxes,
             fontsize=11, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('output/16_learning_curve.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"\\n{'='*60}")
print(f"DIAGNOSI LEARNING CURVE")
print(f"{'='*60}")
print(f"Training F1-Macro (max): {train_mean.max():.4f}")
print(f"Validation F1-Macro (max): {val_mean.max():.4f}")
print(f"Gap finale: {final_gap:.4f}")
print(f"Diagnosi: {diagnosis}")
print(f"\\nDettaglio per training size:")
for ts, tm, vm in zip(train_sizes_abs, train_mean, val_mean):
    print(f"  {ts:>7,} campioni: Train={tm:.4f} | Val={vm:.4f} | Gap={tm-vm:.4f}")"""))

cells.append(md("## 8. Retraining del Modello Finale"))

cells.append(md("""Retrain del modello con gli hyperparametri ottimali su tutti i dati SMOTE."""))

cells.append(code("""best_params = gs_lgbm.best_params_
cv_f1 = gs_lgbm.best_score_

print(f"Hyperparametri ottimali: {best_params}")
print(f"F1-Macro in CV: {cv_f1:.4f}")

# Creazione e addestramento del modello finale
best_model = lgb.LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    force_col_wise=True,
    **best_params
)

print(f"\\nRetraining su {X_train_smote_scaled.shape[0]:,} campioni SMOTE...")
t0 = time.time()
best_model.fit(X_train_smote_scaled, y_train_smote)
train_time = time.time() - t0
print(f"Training completato in {train_time:.1f}s")

# Valutazione sul test set
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
f1w = f1_score(y_test, y_pred, average='weighted')
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')

print(f"\\n{'='*60}")
print(f"RISULTATI FINALI - LightGBM (retrained su dati completi)")
print(f"{'='*60}")
print(f"Accuracy:    {acc:.4f}")
print(f"F1-Macro:    {f1:.4f}")
print(f"F1-Weighted: {f1w:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Tempo train: {train_time:.1f}s")"""))

cells.append(md("## 9. Classification Report"))

cells.append(code("""print(f"\\nClassification Report - LightGBM (retrained)")
print("="*70)
report = classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in range(5)], output_dict=True)
print(classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in range(5)]))

print("\\nAnalisi per classe:")
for i in range(5):
    cls_name = CLASS_NAMES[i]
    p = report[cls_name]['precision']
    r = report[cls_name]['recall']
    f = report[cls_name]['f1-score']
    s = int(report[cls_name]['support'])
    status = "OK" if f >= 0.55 else "DA MIGLIORARE"
    print(f"  {cls_name:20s} | P={p:.3f} R={r:.3f} F1={f:.3f} | Support: {s:,} [{status}]")"""))

cells.append(md("## 10. Matrice di Confusione"))

cells.append(code("""cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Matrice assoluta
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[CLASS_NAMES[i] for i in range(5)])
disp.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
axes[0].set_title('Matrice di Confusione (valori assoluti)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Matrice normalizzata
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=[CLASS_NAMES[i] for i in range(5)])
disp2.plot(ax=axes[1], cmap='Greens', values_format='.2f', colorbar=False)
axes[1].set_title('Matrice di Confusione (recall per classe)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('output/17_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

print("\\nRecall per classe:")
for i in range(5):
    recall_cls = cm_norm[i, i]
    correct = cm[i, i]
    total = cm[i].sum()
    status = "OK" if recall_cls >= 0.55 else "DA MIGLIORARE"
    print(f"  {CLASS_NAMES[i]:20s}: {recall_cls:.2%} ({correct}/{total}) [{status}]")"""))

cells.append(md("## 11. Feature Importance"))

cells.append(code("""importances = best_model.feature_importances_
feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors_imp, edgecolor='white')
for bar, val in zip(bars, feat_imp['importance']):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')
ax.set_xlabel('Importanza (gain)')
ax.set_title('Feature Importance - LightGBM', fontweight='bold')
plt.tight_layout()
plt.savefig('output/18_feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()

print("\\nFeature Importance ordinata:")
for _, row in feat_imp.nlargest(len(FEATURE_COLS), 'importance').iterrows():
    bar = '#' * int(row['importance'] * 50)
    print(f"  {row['feature']:30s}: {row['importance']:.4f} {bar}")"""))

cells.append(md("## 12. SHAP - Explainable AI Globale"))

cells.append(md("""### TreeExplainer per LightGBM
SHAP calcola il contributo marginale di ogni feature usando la teoria dei giochi cooperativi.
TreeExplainer e' ottimizzato per modelli ad albero e calcola SHAP esattamente."""))

cells.append(code("""print("Calcolo SHAP values con TreeExplainer...")
shap_sample = min(1000, len(X_test))

# Campionamento stratificato per SHAP
X_shap_arr, _, y_shap_arr, _ = train_test_split(
    X_test_scaled, y_test.values,
    train_size=shap_sample, random_state=42, stratify=y_test
)
X_shap = pd.DataFrame(X_shap_arr, columns=FEATURE_COLS)
y_shap = pd.Series(y_shap_arr)

print(f"Campioni per SHAP: {X_shap.shape[0]:,}")
print(f"Modello: LightGBM - {best_params}")

start = time.time()
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)
elapsed = time.time() - start
print(f"SHAP calcolato in {elapsed:.1f}s")

# Gestione formato multi-class
if isinstance(shap_values, list):
    print(f"Formato SHAP: lista di {len(shap_values)} array (uno per classe)")
    print(f"Shape per classe: {shap_values[0].shape}")
elif shap_values.ndim == 3:
    print(f"Formato SHAP: array 3D {shap_values.shape} (samples, features, classes)")
else:
    print(f"Formato SHAP: array 2D {shap_values.shape}")"""))

cells.append(md("### SHAP Summary Plot (Beeswarm)"))

cells.append(code("""# Classe mediana (2 = "Medio") come riferimento
if isinstance(shap_values, list):
    shap_for_plot = shap_values[2]
elif shap_values.ndim == 3:
    shap_for_plot = shap_values[:, :, 2]
else:
    shap_for_plot = shap_values

shap.summary_plot(shap_for_plot, X_shap, feature_names=FEATURE_COLS, show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig('output/19_shap_summary.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("### SHAP Bar Plot"))

cells.append(code("""shap.summary_plot(shap_for_plot, X_shap, feature_names=FEATURE_COLS, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('output/20_shap_bar.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("### SHAP Dependence Plot - Top 3 Feature"))

cells.append(code("""top3 = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances}).nlargest(3, 'importance')['feature'].tolist()
print(f"Top 3 feature: {top3}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, feat in enumerate(top3):
    feat_idx = FEATURE_COLS.index(feat)
    x_vals = X_shap.iloc[:, feat_idx].values

    if isinstance(shap_values, list):
        y_vals = np.mean([sv[:, feat_idx] for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        y_vals = shap_values[:, feat_idx, :].mean(axis=1)
    else:
        y_vals = shap_for_plot[:, feat_idx]

    sc = axes[idx].scatter(x_vals, y_vals, alpha=0.3, s=10, c=x_vals, cmap='viridis')
    axes[idx].set_xlabel(feat, fontweight='bold')
    axes[idx].set_ylabel('SHAP value')
    axes[idx].set_title(f'Effetto di {feat}', fontweight='bold')
    plt.colorbar(sc, ax=axes[idx], fraction=0.046, label='Valore feature')

plt.tight_layout()
plt.savefig('output/21_shap_dependence.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("## 13. LIME - Explainable AI Locale"))

cells.append(md("""LIME spiega singole previsioni perturbando l'input e osservando i cambiamenti.
Mostriamo 3 esempi: classe 0 (Molto Difficile), 2 (Medio), 4 (Molto Facile)."""))

cells.append(code("""explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=FEATURE_COLS,
    class_names=[CLASS_NAMES[i] for i in range(5)],
    mode='classification',
    random_state=42
)
print("Explainer LIME configurato!")
print(f"Training data per LIME: {X_train_scaled.shape[0]:,} campioni")"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(20, 8))
lime_success = 0

for i, target_class in enumerate([0, 2, 4]):
    mask = y_test == target_class
    indices = X_test[mask].index
    if len(indices) == 0:
        axes[i].text(0.5, 0.5, f'Nessun campione per {CLASS_NAMES[target_class]}',
                     ha='center', va='center', transform=axes[i].transAxes)
        continue

    idx = indices[0]
    row_idx = list(X_test.index).index(idx)
    instance = X_test_scaled[row_idx]

    try:
        exp = explainer_lime.explain_instance(
            instance, best_model.predict_proba, num_features=6, top_labels=1
        )
        pred_class = best_model.predict(instance.reshape(1, -1))[0]
        probs = best_model.predict_proba(instance.reshape(1, -1))[0]

        print(f"\\nLIME - Classe reale: {CLASS_NAMES[target_class]} -> Predetta: {CLASS_NAMES[pred_class]} (conf: {probs[pred_class]:.1%})")
        lime_list = exp.as_list(label=pred_class)
        for feat, weight in lime_list:
            direction = "+" if weight > 0 else "-"
            print(f"  {direction} {feat}: {weight:+.4f}")

        axes[i].barh(
            [f[0] for f in lime_list],
            [f[1] for f in lime_list],
            color=['#d32f2f' if v < 0 else '#388e3c' for _, v in lime_list],
            edgecolor='white'
        )
        axes[i].set_title(f'LIME: {CLASS_NAMES[target_class]} -> {CLASS_NAMES[pred_class]}', fontweight='bold')
        axes[i].set_xlabel('Peso nella decisione')
        axes[i].axvline(x=0, color='black', linewidth=0.5)
        lime_success += 1

    except Exception as e:
        axes[i].text(0.5, 0.5, f'LIME failed\\n{str(e)[:50]}',
                     ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
        axes[i].set_title(f'LIME: {CLASS_NAMES[target_class]} (errore)', fontweight='bold')
        print(f"  Errore LIME per {CLASS_NAMES[target_class]}: {e}")

plt.tight_layout()
plt.savefig('output/22_lime.png', bbox_inches='tight', dpi=150)
plt.show()
print(f"\\nLIME: {lime_success}/3 spiegazioni generate con successo")"""))

cells.append(md("## 14. Test Case - Scenari Reali"))

cells.append(md("""Simuliamo 5 scenari realistici per verificare il modello in condizioni operative."""))

cells.append(code("""zone_lookup = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv')
zone_lookup.columns = ['LocationID', 'Borough', 'Zone', 'service_zone']

test_cases = [
    (236, 8, 0, 0, "Manhattan Midtown, Lunedi 08:00 - Rush hour mattutino"),
    (236, 3, 0, 0, "Manhattan Midtown, Lunedi 03:00 - Notte fonda"),
    (7, 8, 0, 0, "Bronx Fordham, Lunedi 08:00 - Rush hour residenziale"),
    (138, 10, 0, 2, "JFK Airport, Mercoledi 10:00 - Fascia centrale"),
    (230, 20, 0, 6, "Times Square, Sabato 20:00 - Weekend serale"),
]

def prepare_input(zone_id, hour, minute, dow, df_orig):
    \"\"\"Prepara le feature per una previsione.\"\"\"
    bucket = hour * 2 + (1 if minute >= 30 else 0)
    mask = (df_orig['PULocationID'] == zone_id) & (df_orig['half_hour_bucket'] == bucket) & (df_orig['day_of_week'] == dow)
    subset = df_orig[mask]

    if len(subset) > 0:
        return {
            'PULocationID': zone_id,
            'half_hour_bucket': bucket,
            'day_of_week': dow,
            'month': subset['month'].mode()[0] if len(subset['month'].mode()) > 0 else 1,
            'unique_taxi_types': subset['unique_taxi_types'].iloc[0],
            'avg_trip_duration_min': subset['avg_trip_duration_min'].iloc[0],
            'is_weekend': 1 if dow >= 5 else 0,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            'is_night': 1 if (hour >= 23 or hour <= 5) else 0,
            'borough_encoded': subset['borough_encoded'].iloc[0],
            'service_zone_encoded': subset['service_zone_encoded'].iloc[0]
        }
    else:
        return {
            'PULocationID': zone_id,
            'half_hour_bucket': bucket,
            'day_of_week': dow,
            'month': df_orig['month'].mode()[0] if len(df_orig['month'].mode()) > 0 else 1,
            'unique_taxi_types': df_orig['unique_taxi_types'].mode()[0],
            'avg_trip_duration_min': df_orig['avg_trip_duration_min'].median(),
            'is_weekend': 1 if dow >= 5 else 0,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            'is_night': 1 if (hour >= 23 or hour <= 5) else 0,
            'borough_encoded': df_orig['borough_encoded'].mode()[0],
            'service_zone_encoded': df_orig['service_zone_encoded'].mode()[0]
        }

print(f"\\n{'='*85}")
print(f"TEST CASE - LightGBM (Hyperparametri: {best_params})")
print(f"{'='*85}")

for zone_id, hour, minute, dow, desc in test_cases:
    zi = zone_lookup[zone_lookup['LocationID'] == zone_id]
    zone_name = zi['Zone'].values[0] if len(zi) > 0 else f"Zone {zone_id}"
    borough = zi['Borough'].values[0] if len(zi) > 0 else "?"

    features = pd.DataFrame([prepare_input(zone_id, hour, minute, dow, df)])
    features_scaled = scaler.transform(features)
    pred = best_model.predict(features_scaled)[0]
    probs = best_model.predict_proba(features_scaled)[0]

    print(f"\\n{'-'*85}")
    print(f"  SCENARIO: {desc}")
    print(f"  Zona: {zone_name} ({borough}) - ID: {zone_id}")
    print(f"  PREDIZIONE: {CLASS_NAMES[pred]} (confidenza: {probs[pred]:.1%})")
    print(f"  Distribuzione probabilita':")
    for i in range(5):
        bar = '#' * int(probs[i] * 30)
        marker = " <--" if i == pred else ""
        print(f"    {CLASS_NAMES[i]:20s}: {probs[i]:5.1%} {bar}{marker}")"""))

cells.append(md("## 15. Confronto con il Modello Precedente (Gennaio 2025)"))

cells.append(md("""Confrontiamo le performance del modello addestrato su tutti i mesi
con quello addestrato solo su Gennaio 2025."""))

cells.append(code("""# Carica risultati precedenti
try:
    with open('output/model_comparison_results.json', 'r') as fp:
        prev_results = json.load(fp)

    # Trova il risultato LightGBM
    prev_lgbm = None
    for r in prev_results:
        if r['model'] == 'LightGBM':
            prev_lgbm = r
            break

    if prev_lgbm:
        print(f"{'='*70}")
        print(f"CONFRONTO: Gennaio 2025 vs Tutti i Mesi 2025")
        print(f"{'='*70}")
        print(f"{'Metrica':<20} {'Gen 2025':>12} {'Tutti i Mesi':>14} {'Delta':>10}")
        print(f"{'-'*60}")

        metrics = {
            'cv_f1_macro': 'CV F1-Macro',
            'test_accuracy': 'Test Accuracy',
            'test_f1_macro': 'Test F1-Macro',
            'test_f1_weighted': 'Test F1-Weighted',
            'test_precision': 'Test Precision',
            'test_recall': 'Test Recall'
        }

        current_metrics = {
            'cv_f1_macro': cv_f1,
            'test_accuracy': acc,
            'test_f1_macro': f1,
            'test_f1_weighted': f1w,
            'test_precision': prec,
            'test_recall': rec
        }

        for key, label in metrics.items():
            prev_val = prev_lgbm[key]
            curr_val = current_metrics[key]
            delta = curr_val - prev_val
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            print(f"{label:<20} {prev_val:>12.4f} {curr_val:>14.4f} {delta_str:>10}")

        print(f"\\nHyperparametri Gen 2025: {prev_lgbm['best_params']}")
        print(f"Hyperparametri Tutti i Mesi: {best_params}")
    else:
        print("Risultati precedenti non trovati per LightGBM.")
except FileNotFoundError:
    print("File model_comparison_results.json non trovato. Confronto non disponibile.")"""))

cells.append(md("## 16. Salvataggio del Modello e dei Risultati"))

cells.append(code("""# 1. Salvataggio artefatti completi
artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_cols': FEATURE_COLS,
    'class_names': CLASS_NAMES,
    'model_name': 'LightGBM',
    'best_params': {k: str(v) for k, v in best_params.items()},
    'metrics': {
        'accuracy': round(float(acc), 4),
        'f1_macro': round(float(f1), 4),
        'f1_weighted': round(float(f1w), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'cv_f1_macro': round(float(cv_f1), 4),
    },
    'dataset_info': {
        'total_rows': len(df),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'smote_rows': len(X_train_smote),
        'months': 'all_2025'
    }
}

joblib.dump(artifacts, 'output/ml_model_artifacts_all_months.pkl')
print(f"Artefatti completi salvati: output/ml_model_artifacts_all_months.pkl")

# 2. Salvataggio modello leggero
joblib.dump(best_model, 'output/best_model_all_months.pkl')
print(f"Modello leggero salvato: output/best_model_all_months.pkl")

# 3. Salvataggio metriche
metrics_summary = {
    'model': 'LightGBM',
    'dataset': 'all_months_2025',
    'cv_f1_macro': round(float(cv_f1), 4),
    'test_accuracy': round(float(acc), 4),
    'test_f1_macro': round(float(f1), 4),
    'test_f1_weighted': round(float(f1w), 4),
    'test_precision': round(float(prec), 4),
    'test_recall': round(float(rec), 4),
    'best_params': {k: str(v) for k, v in best_params.items()},
    'train_time_seconds': round(train_time, 1),
    'learning_curve_diagnosis': diagnosis
}

with open('output/lightgbm_advanced_results.json', 'w') as fp:
    json.dump(metrics_summary, fp, indent=2)
print(f"Metriche salvate: output/lightgbm_advanced_results.json")

print(f"\\n{'='*60}")
print(f"NOTEBOOK COMPLETATO")
print(f"{'='*60}")
print(f"Modello: LightGBM")
print(f"Hyperparametri: {best_params}")
print(f"Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
print(f"Learning Curve: {diagnosis}")
print(f"File salvati nella cartella output/")"""))

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

with open(r'C:\Users\andre\Desktop\Progetto_Accenture\04_lightgbm_advanced_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook 04 scritto!")
with open(r'C:\Users\andre\Desktop\Progetto_Accenture\04_lightgbm_advanced_training.ipynb', 'r') as f:
    json.load(f)
print("JSON valido!")
print(f"Celle totali: {len(cells)}")
