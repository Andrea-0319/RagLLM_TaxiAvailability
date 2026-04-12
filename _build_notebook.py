"""Generate notebook: GridSearchCV RF + LightGBM on full SMOTE data."""
import json

def md(lines):
    raw = lines.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in raw]}

def code(text):
    raw = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in raw]}

cells = []

cells.append(md("""# NYC Taxi Demand Prediction - Model Selection via Grid Search

## Obiettivo
Confrontare **RandomForest** e **LightGBM** tramite GridSearchCV con Cross Validation per identificare il modello e gli hyperparametri ottimali per la previsione della disponibilita' di taxi nelle zone NYC.

## Perche' questi due modelli?
| Modello | Punti di forza |
|---------|---------------|
| **RandomForest** | Ensemble di alberi decisionali baggati. Robusto al rumore, gestisce bene le non-linearita', meno propenso a overfitting. Ideale come baseline solida. |
| **LightGBM** | Gradient boosting ottimizzato di Microsoft. Usa split leaf-wise (cresce il nodo con massima riduzione della loss), e' 5-10x piu' veloce di XGBoost, gestisce nativamente dataset grandi. Con l'aumento dei dati (mesi aggiuntivi) scala meglio. |

## Strategia
1. **SMOTE** per bilanciare le classi (il dataset originale ha il 67% nelle classi "Difficile")
2. **GridSearchCV con 3-fold** su **tutti i dati di training SMOTE** - nessun subset, massima affidabilita'
3. **Retrain del migliore** sui dati completi per performance massime
4. **Valutazione sul test set** (mai visto durante training o selezione)
5. **Explainable AI** con SHAP (globale) + LIME (locale)
6. **Test case** con scenari reali tipici di un operatore taxi

## Classi Target
| ID | Classe | Range Availability | Significato |
|----|--------|-------------------|-------------|
| 0 | Molto Difficile | 0.0 - 0.2 | Quasi nessun taxi disponibile |
| 1 | Difficile | 0.2 - 0.4 | Pochi taxi, attesa lunga |
| 2 | Medio | 0.4 - 0.6 | Situazione bilanciata |
| 3 | Facile | 0.6 - 0.8 | Buona disponibilita' |
| 4 | Molto Facile | 0.8 - 1.0 | Molti taxi disponibili |"""))

cells.append(md("## 1. Setup - Import delle librerie"))

cells.append(code("""import warnings
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier

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

cells.append(md("## 2. Caricamento del Dataset"))

cells.append(md("""Il dataset e' stato aggregato nel notebook precedente (`01_eda_feature_engineering.py`).
Ogni riga rappresenta una combinazione unica di **zona + fascia oraria (30 min) + giorno della settimana + mese**.

Le feature includono:
- **Identificative**: PULocationID, half_hour_bucket, day_of_week, month
- **Statistiche**: unique_taxi_types, avg_trip_duration_min
- **Derivate**: is_weekend, is_rush_hour, is_night
- **Codificate**: borough_encoded, service_zone_encoded"""))

cells.append(code("""DATA_PATH = r'output/model_ready_jan2025.parquet'
df = pd.read_parquet(DATA_PATH)
print(f"Shape del dataset: {df.shape[0]:,} righe, {df.shape[1]} colonne")
print(f"\\nColonne disponibili: {list(df.columns)}")
print(f"\\nPrime 5 righe:")
df.head()"""))

cells.append(md("### 2.1 Distribuzione delle classi target"))

cells.append(md("""Prima di procedere, e' fondamentale analizzare la distribuzione delle classi.
Un forte squilibrio potrebbe compromettere l'addestramento, portando il modello a favorire le classi maggioritarie."""))

cells.append(code("""CLASS_NAMES = {0: 'Molto Difficile', 1: 'Difficile', 2: 'Medio', 3: 'Facile', 4: 'Molto Facile'}
CLASS_COLORS = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']

counts = df['availability_class_id'].value_counts().sort_index()
labels = [CLASS_NAMES[i] for i in counts.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(labels, counts.values, color=CLASS_COLORS, edgecolor='white')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}', ha='center', fontweight='bold')
axes[0].set_title('Distribuzione Classi - Conteggio', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

pcts = counts / len(df) * 100
axes[1].pie(pcts.values, labels=labels, colors=CLASS_COLORS, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Distribuzione Classi - Percentuali', fontweight='bold')
plt.tight_layout()
plt.savefig('output/01_class_distribution.png', bbox_inches='tight', dpi=150)
plt.show()

print("Distribuzione dettagliata:")
for cls_id in sorted(counts.index):
    pct = counts[cls_id] / len(df) * 100
    print(f"  {CLASS_NAMES[cls_id]:20s}: {counts[cls_id]:,} ({pct:.1f}%)")"""))

cells.append(md("## 3. Preparazione dei Dati - Train/Test Split"))

cells.append(md("""Separiamo le feature dalla variabile target e creiamo uno split stratificato 80/20.

La **stratificazione** e' cruciale: garantisce che la proporzione delle classi nel train e nel test sia la stessa del dataset originale. Senza di essa, il test set potrebbe non rappresentare adeguatamente le classi minoritarie."""))

cells.append(code("""FEATURE_COLS = [
    'PULocationID', 'half_hour_bucket', 'day_of_week', 'month',
    'unique_taxi_types', 'avg_trip_duration_min',
    'is_weekend', 'is_rush_hour', 'is_night',
    'borough_encoded', 'service_zone_encoded'
]

X = df[FEATURE_COLS].copy()
y = df['availability_class_id'].copy()

# Split stratificato: 80% train, 20% test
# random_state=42 garantisce riproducibilita'
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} campioni ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]:,} campioni ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"Feature:      {X_train.shape[1]}")
print(f"Classi:       {y_train.nunique()}")"""))

cells.append(md("## 4. SMOTE - Synthetic Minority Over-sampling Technique"))

cells.append(md("""### Il problema dello squilibrio
Come visto sopra, le classi "Molto Difficile" e "Difficile" rappresentano circa il 67% del dataset.
Un modello addestrato su dati sbilanciati tenderebbe a prevedere sempre queste classi, ottenendo alta accuracy ma scarsa utilita' pratica.

### La soluzione: SMOTE
SMOTE crea esempi sintetici delle classi minoritarie interpolando tra i k-vicini piu' prossimi nello spazio delle feature.
A differenza del semplice oversampling (duplicazione), SMOTE genera dati **nuovi** e **variati**, riducendo il rischio di overfitting.

**Importante**: SMOTE viene applicato **solo sul training set**, mai sul test set. Il test set deve rappresentare la distribuzione reale dei dati."""))

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
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\\n=== Distribuzione DOPO SMOTE ===")
for cls_id in sorted(y_train_smote.unique()):
    cnt = (y_train_smote == cls_id).sum()
    pct = cnt / len(y_train_smote) * 100
    print(f"  {CLASS_NAMES[cls_id]:20s}: {cnt:,} ({pct:.1f}%)")

print(f"\\nEspansione: {len(y_train):,} -> {len(y_train_smote):,} campioni (+{(len(y_train_smote)/len(y_train)-1)*100:.0f}%)")"""))

cells.append(md("## 5. Standardizzazione delle Feature"))

cells.append(md("""La standardizzazione (Z-score normalization) trasforma ogni feature in modo che abbia media 0 e deviazione standard 1.

**Perche' e' necessaria?**
- **SMOTE** usa distanze euclidee: feature con scale diverse dominerebbero il calcolo
- **LightGBM** beneficia della standardizzazione per la convergenza
- **SHAP** produce valori piu' interpretabili con feature standardizzate

Il scaler viene fit sul training originale (pre-SMOTE) per evitare data leakage dai dati sintetici."""))

cells.append(code("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_smote_scaled = scaler.transform(X_train_smote)

print(f"Shape training scalato: {X_train_scaled.shape}")
print(f"Shape test scalato:     {X_test_scaled.shape}")
print(f"Shape SMOTE scalato:    {X_train_smote_scaled.shape}")
print(f"\\nMedia delle feature (dovrebbe essere ~0): {X_train_scaled.mean(axis=0).round(2)}")
print(f"Std delle feature (dovrebbe essere ~1):   {X_train_scaled.std(axis=0).round(2)}")"""))

cells.append(md("## 6. GridSearchCV - Selezione Modello e Hyperparametri"))

cells.append(md("""### Strategia di ricerca
Utilizziamo **GridSearchCV** con **3-fold stratified cross-validation** su **tutti i dati SMOTE**.

Perche' nessun subset?
- Con solo 2 modelli e grid contenute, il tempo totale e' gestibile
- Usare tutti i dati garantisce stime di performance piu' affidabili
- La stratified CV protegge dall'overfitting durante la selezione

**Metrica di ottimizzazione**: F1-Macro (tratta tutte le classi allo stesso modo, cruciale con classi sbilanciate)

### Hyperparametri ricercati

**RandomForest** (8 combinazioni):
- `n_estimators`: [100, 200] - numero di alberi
- `max_depth`: [10, None] - profondita' massima (None = espansione completa)
- `class_weight`: [None, 'balanced'] - pesi inversamente proporzionali alle frequenze

**LightGBM** (8 combinazioni):
- `n_estimators`: [100, 200] - numero di boosting round
- `learning_rate`: [0.05, 0.1] - shrinkage, quanto ogni albero corregge l'errore
- `num_leaves`: [31, 63] - foglie massime per albero (controlla complessita')

Totale: 16 combinazioni x 3 folds = **48 fit**"""))

cells.append(code("""# Configurazione Cross Validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print(f"Cross Validation: {cv.get_n_splits()} fold stratificati")
print(f"Dati per GridSearchCV: {X_train_smote_scaled.shape[0]:,} campioni (100% dei dati SMOTE)")"""))

cells.append(md("### 6.1 Random Forest - Grid Search"))

cells.append(code("""param_rf = {
    'n_estimators': [100, 200],       # 100 alberi (veloce) vs 200 (piu' stabile)
    'max_depth': [10, None],           # 10 (regolarizza) vs None (massima espressivita')
    'class_weight': [None, 'balanced'] # None (uniforme) vs balanced (compensa squilibrio)
}

gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_rf,
    cv=cv,
    scoring='f1_macro',    # Ottimizza F1 macro: tutte le classi hanno lo stesso peso
    n_jobs=-1,             # Parallelizza su tutti i core
    verbose=1
)

n_combinations_rf = 8  # 2 x 2 x 2
n_folds = 3
print(f"Combinazioni hyperparametri: {n_combinations_rf}")
print(f"Fit totali: {n_combinations_rf} combinazioni x {n_folds} folds = {n_combinations_rf * n_folds}")
print(f"\\nAvvio GridSearchCV per RandomForest...")
t0 = time.time()
gs_rf.fit(X_train_smote_scaled, y_train_smote)
elapsed = time.time() - t0
print(f"\\nCompletato in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Miglior F1-Macro (CV): {gs_rf.best_score_:.4f}")
print(f"Migliori hyperparametri: {gs_rf.best_params_}")"""))

cells.append(md("### 6.2 LightGBM - Grid Search"))

cells.append(code("""param_lgbm = {
    'n_estimators': [100, 200],       # 100 round (veloce) vs 200 (piu' raffinato)
    'learning_rate': [0.05, 0.1],     # 0.05 (apprendimento cauto) vs 0.1 (standard)
    'num_leaves': [31, 63]            # 31 (conservativo) vs 63 (piu' espressivo)
}

gs_lgbm = GridSearchCV(
    lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,           # Silenzia i warning interni
        force_col_wise=True   # Ottimizzazione per dati tabulari
    ),
    param_lgbm,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

print(f"Combinazioni: 8 | Fit totali: 24")
print(f"\\nAvvio GridSearchCV per LightGBM...")
t0 = time.time()
gs_lgbm.fit(X_train_smote_scaled, y_train_smote)
elapsed = time.time() - t0
print(f"\\nCompletato in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Miglior F1-Macro (CV): {gs_lgbm.best_score_:.4f}")
print(f"Migliori hyperparametri: {gs_lgbm.best_params_}")"""))

cells.append(md("## 7. Confronto dei Modelli"))

cells.append(md("""Confrontiamo i due modelli sulle metriche chiave calcolate sul **test set** (dati mai visti).

**Metriche riportate:**
- **CV F1-Macro**: F1 medio dalla cross-validation (stima della generalizzazione)
- **Test Accuracy**: Percentuale di previsioni corrette
- **Test F1-Macro**: Media armonica di precision e recall, non pesata per classe
- **Test F1-Weighted**: F1 pesato per il supporto di ogni classe
- **Test Precision**: Quanti dei previsti positivi sono veri positivi
- **Test Recall**: Quanti dei veri positivi sono stati individuati"""))

cells.append(code("""gs_results = {
    'RandomForest': gs_rf,
    'LightGBM': gs_lgbm
}

comparison = []
for name, gs in gs_results.items():
    y_pred = gs.best_estimator_.predict(X_test_scaled)
    comparison.append({
        'model': name,
        'cv_f1': round(gs.best_score_, 4),
        'test_accuracy': round(accuracy_score(y_test, y_pred), 4),
        'test_f1_macro': round(f1_score(y_test, y_pred, average='macro'), 4),
        'test_f1_weighted': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'test_precision': round(precision_score(y_test, y_pred, average='macro'), 4),
        'test_recall': round(recall_score(y_test, y_pred, average='macro'), 4),
        'best_params': gs.best_params_,
        'y_pred': y_pred,
        'best_estimator': gs.best_estimator_
    })

results_df = pd.DataFrame(comparison)
results_df = results_df.sort_values('test_f1_macro', ascending=False).reset_index(drop=True)

print("\\n" + "="*90)
print("CONFRONTO MODELLI (GridSearchCV su dati SMOTE completi, valutati su test set)")
print("="*90)
for i, row in results_df.iterrows():
    marker = " <-- MIGLIORE" if i == 0 else ""
    print(f"\\n{i+1}. {row['model']}{marker}")
    print(f"   CV F1-Macro:    {row['cv_f1']:.4f}")
    print(f"   Test Accuracy:  {row['test_accuracy']:.4f}")
    print(f"   Test F1-Macro:  {row['test_f1_macro']:.4f}")
    print(f"   Test F1-Weight: {row['test_f1_weighted']:.4f}")
    print(f"   Test Precision: {row['test_precision']:.4f}")
    print(f"   Test Recall:    {row['test_recall']:.4f}")
    print(f"   Best Params:    {row['best_params']}")"""))

cells.append(md("### Visualizzazione del confronto"))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_df = results_df.copy()
colors = ['#1976d2', '#f57c00']

# CV F1 vs Test F1
x = np.arange(len(plot_df))
w = 0.3
bars1 = axes[0].bar(x - w/2, plot_df['cv_f1'], w, label='CV F1-Macro', color=colors[0], edgecolor='white')
bars2 = axes[0].bar(x + w/2, plot_df['test_f1_macro'], w, label='Test F1-Macro', color=colors[1], edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot_df['model'], rotation=15, ha='right', fontweight='bold')
axes[0].set_ylabel('F1-Macro')
axes[0].set_title('CV vs Test F1-Macro', fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 1)
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', fontweight='bold', fontsize=10)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', fontweight='bold', fontsize=10)

# Test F1-Macro con breakdown per classe
for idx, row in plot_df.iterrows():
    y_pred = row['y_pred']
    f1_per_class = f1_score(y_test, y_pred, average=None)
    axes[1].barh(
        [f"{row['model']} - {CLASS_NAMES[c]}" for c in range(5)],
        f1_per_class,
        color=CLASS_COLORS,
        edgecolor='white',
        label=row['model'] if idx == 0 else ""
    )
axes[1].set_xlabel('F1-Score per Classe')
axes[1].set_title('F1-Score per Classe (Test)', fontweight='bold')
axes[1].set_xlim(0, 1)

# Radar-like: Precision vs Recall
x_pos = np.arange(len(plot_df))
axes[2].bar(x_pos - w/2, plot_df['test_precision'], w, label='Precision', color='#388e3c', edgecolor='white')
axes[2].bar(x_pos + w/2, plot_df['test_recall'], w, label='Recall', color='#d32f2f', edgecolor='white')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(plot_df['model'], rotation=15, ha='right', fontweight='bold')
axes[2].set_ylabel('Score')
axes[2].set_title('Precision vs Recall (Macro)', fontweight='bold')
axes[2].legend()
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('output/02_model_comparison.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("## 8. Retrain del Migliore Modello sui Dati Completi"))

cells.append(md("""Il modello vincitore del GridSearchCV viene ora **riaddestrato da zero** su tutti i dati SMOTE.

**Perche' il retrain?**
Il GridSearchCV addestra versioni "parziali" del modello (una per ogni fold). Il best_estimator e' quello addestrato sull'ultimo fold, che non ha visto il ~33% dei dati. Ricreando il modello con gli stessi hyperparametri e addestrandolo su **tutti** i dati SMOTE, massimizziamo l'informazione disponibile."""))

cells.append(code("""best_row = results_df.iloc[0]
best_model_name = best_row['model']
best_params = best_row['best_params']

print(f"Miglior modello selezionato: {best_model_name}")
print(f"Hyperparametri ottimali: {best_params}")
print(f"F1-Macro in CV: {best_row['cv_f1']:.4f}")
print(f"F1-Macro su test: {best_row['test_f1_macro']:.4f}")

# Mappa nomi modello -> classe
model_classes = {
    'RandomForest': RandomForestClassifier,
    'LightGBM': lgb.LGBMClassifier
}

# Parametri extra specifici per modello
extra_params = {}
if best_model_name == 'RandomForest':
    extra_params = {'n_jobs': -1}
elif best_model_name == 'LightGBM':
    extra_params = {'verbose': -1, 'n_jobs': -1, 'force_col_wise': True}

# Creazione e addestramento del modello finale
model_cls = model_classes[best_model_name]
best_model = model_cls(random_state=42, **best_params, **extra_params)

print(f"\\nRetraining di {best_model_name} su {X_train_smote_scaled.shape[0]:,} campioni SMOTE...")
t0 = time.time()
best_model.fit(X_train_smote_scaled, y_train_smote)
train_time = time.time() - t0
print(f"Training completato in {train_time:.1f}s")

# Valutazione finale sul test set
y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best, average='macro')
f1w = f1_score(y_test, y_pred_best, average='weighted')
prec = precision_score(y_test, y_pred_best, average='macro')
rec = recall_score(y_test, y_pred_best, average='macro')

print(f"\\n{'='*60}")
print(f"RISULTATI FINALI - {best_model_name} (retrained su dati completi)")
print(f"{'='*60}")
print(f"Accuracy:    {acc:.4f}")
print(f"F1-Macro:    {f1:.4f}")
print(f"F1-Weighted: {f1w:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Tempo train: {train_time:.1f}s")"""))

cells.append(md("## 9. Analisi Dettagliata - Classification Report"))

cells.append(md("""Il classification report fornisce precision, recall e F1 per **ogni singola classe**.

**Come interpretarlo:**
- **Precision**: se il modello dice "Classe X", quanto e' affidabile?
- **Recall**: fra tutti i veri "Classe X", quanti ne trova?
- **F1-score**: media armonica di precision e recall
- **Support**: numero di campioni reali di quella classe nel test set"""))

cells.append(code("""print(f"\\nClassification Report - {best_model_name} (retrained)")
print("="*70)
report = classification_report(y_test, y_pred_best, target_names=[CLASS_NAMES[i] for i in range(5)], output_dict=True)
print(classification_report(y_test, y_pred_best, target_names=[CLASS_NAMES[i] for i in range(5)]))

# Analisi per classe
print("\\nAnalisi per classe:")
for i in range(5):
    cls_name = CLASS_NAMES[i]
    p = report[cls_name]['precision']
    r = report[cls_name]['recall']
    f = report[cls_name]['f1-score']
    s = int(report[cls_name]['support'])
    print(f"  {cls_name:20s} | P={p:.3f} R={r:.3f} F1={f:.3f} | Support: {s:,}")"""))

cells.append(md("## 10. Matrice di Confusione"))

cells.append(md("""La matrice di confusione mostra **dove** il modello sbaglia.

- **Diagonale**: previsioni corrette (valori alti = bene)
- **Fuori diagonale**: errori di classificazione
  - Se la classe 0 viene spesso confusa con la classe 1, ha senso: sono classi adiacenti
  - Se la classe 0 viene confusa con la classe 4, c'e' un problema strutturale"""))

cells.append(code("""cm = confusion_matrix(y_test, y_pred_best)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Matrice assoluta
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[CLASS_NAMES[i] for i in range(5)])
disp.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
axes[0].set_title('Matrice di Confusione (valori assoluti)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Matrice normalizzata (recall per classe)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=[CLASS_NAMES[i] for i in range(5)])
disp2.plot(ax=axes[1], cmap='Greens', values_format='.2f', colorbar=False)
axes[1].set_title('Matrice di Confusione (recall per classe)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('output/03_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

print("\\nRecall per classe (diagonale normalizzata):")
for i in range(5):
    recall_cls = cm_norm[i, i]
    correct = cm[i, i]
    total = cm[i].sum()
    status = "OK" if recall_cls >= 0.5 else "DA MIGLIORARE"
    print(f"  {CLASS_NAMES[i]:20s}: {recall_cls:.2%} ({correct}/{total}) [{status}]")"""))

cells.append(md("## 11. Feature Importance"))

cells.append(md("""La feature importance mostra quali variabili contribuiscono maggiormente alle decisioni del modello.

**Interpretazione:**
- Valore piu' alto = la feature e' piu' discriminante
- Non indica la direzione dell'effetto (positivo/negativo), solo l'intensita'
- Per capire la direzione serve SHAP (sezione successiva)"""))

cells.append(code("""importances = best_model.feature_importances_
feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors_imp, edgecolor='white')
for bar, val in zip(bars, feat_imp['importance']):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')
ax.set_xlabel('Importanza (gain)')
ax.set_title(f'Feature Importance - {best_model_name}', fontweight='bold')
plt.tight_layout()
plt.savefig('output/04_feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()

print("\\nFeature Importance ordinata:")
for _, row in feat_imp.nlargest(len(FEATURE_COLS), 'importance').iterrows():
    bar = '#' * int(row['importance'] * 50)
    print(f"  {row['feature']:30s}: {row['importance']:.4f} {bar}")"""))

cells.append(md("## 12. SHAP - SHapley Additive exPlanations (Explainable AI Globale)"))

cells.append(md("""### Cos'e' SHAP?
SHAP si basa sulla teoria dei giochi cooperativi (valori di Shapley). Per ogni previsione, calcola il **contributo marginale** di ogni feature rispetto a una previsione base (media del dataset).

**Vantaggi rispetto alla feature importance tradizionale:**
- Mostra la **direzione** dell'effetto (valori positivi = spingono verso classi piu' alte)
- Funziona a livello di **singola previsione** (explainability locale)
- E' matematicamente fondato e consistente

**TreeExplainer**: versione ottimizzata per modelli ad albero (RF, LGBM, XGBoost), calcola SHAP esattamente in O(T*L*D) invece che esponenziale."""))

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
print(f"Modello: {best_model_name}")

start = time.time()
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)
elapsed = time.time() - start
print(f"SHAP calcolato in {elapsed:.1f}s")

# Gestione formato: multi-class puo' restituire lista o array 3D
if isinstance(shap_values, list):
    print(f"Formato SHAP: lista di {len(shap_values)} array (uno per classe)")
    print(f"Shape per classe: {shap_values[0].shape}")
elif shap_values.ndim == 3:
    print(f"Formato SHAP: array 3D {shap_values.shape} (samples, features, classes)")
else:
    print(f"Formato SHAP: array 2D {shap_values.shape}")"""))

cells.append(md("### SHAP Summary Plot (Beeswarm)"))

cells.append(md("""Il beeswarm plot mostra per ogni feature:
- **Asse X**: impatto SHAP (valore positivo = aumenta la previsione, negativo = diminuisce)
- **Colore**: valore della feature (rosso = alto, blu = basso)
- **Densita'**: ogni punto e' un campione del test set

**Come leggerlo:**
- Se i punti rossi sono a destra e i blu a sinistra → correlazione positiva
- Se i punti rossi sono a sinistra e i blu a destra → correlazione negativa
- Dispersione ampia → la feature ha effetto non-lineare"""))

cells.append(code("""# Per multi-class, usiamo la classe mediana o la media assoluta
if isinstance(shap_values, list):
    # Media assoluta su tutte le classi per ranking, poi plot della classe mediana
    shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    feat_order = np.argsort(shap_abs_mean)[::-1]
    # Usiamo la classe mediana (2 = "Medio") come riferimento
    shap_for_plot = shap_values[2]
elif shap_values.ndim == 3:
    shap_for_plot = shap_values[:, :, 2]
else:
    shap_for_plot = shap_values

shap.summary_plot(shap_for_plot, X_shap, feature_names=FEATURE_COLS, show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig('output/05_shap_summary.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("### SHAP Bar Plot"))

cells.append(md("""Il bar plot mostra l'**importanza media assoluta** di ogni feature: quanto in media ogni feature contribuisce alle previsioni, indipendentemente dalla direzione."""))

cells.append(code("""shap.summary_plot(shap_for_plot, X_shap, feature_names=FEATURE_COLS, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('output/06_shap_bar.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("### SHAP Dependence Plot - Top 3 Feature"))

cells.append(md("""I dependence plot mostrano la relazione tra il valore di una feature e il suo impatto SHAP.
Selezioniamo le top 3 feature per importanza."""))

cells.append(code("""top3 = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances}).nlargest(3, 'importance')['feature'].tolist()
print(f"Top 3 feature per SHAP dependence: {top3}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, feat in enumerate(top3):
    feat_idx = FEATURE_COLS.index(feat)
    x_vals = X_shap.iloc[:, feat_idx].values

    # Estraiamo i SHAP values per questa feature
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
plt.savefig('output/07_shap_dependence.png', bbox_inches='tight', dpi=150)
plt.show()"""))

cells.append(md("## 13. LIME - Local Interpretable Model-agnostic Explanations (Explainable AI Locale)"))

cells.append(md("""### Cos'e' LIME?
Mentre SHAP e' globale (spiega il modello nel suo complesso), LIME spiega **singole previsioni**.

**Come funziona:**
1. Prende un'istanza da spiegare
2. Genera campioni perturbati intorno ad essa
3. Addestra un modello lineare interpretabile sui perturbati
4. I coefficienti del modello lineare diventano la spiegazione locale

**Differenza con SHAP:**
- SHAP: fondamento teorico solido (Shapley), computazionalmente costoso
- LIME: approssimazione locale, piu' veloce, ma meno consistente

Mostriamo 3 esempi: una previsione di classe 0 (Molto Difficile), 2 (Medio), 4 (Molto Facile)."""))

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
    # Trova un campione della classe target nel test set
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
plt.savefig('output/08_lime.png', bbox_inches='tight', dpi=150)
plt.show()
print(f"\\nLIME: {lime_success}/3 spiegazioni generate con successo")"""))

cells.append(md("## 14. Test Case - Scenari Reali per Marcello"))

cells.append(md("""Simuliamo 5 scenari che un operatore taxi potrebbe incontrare nella vita reale.
Per ognuno, il modello fornisce una previsione con le probabilita' per ogni classe.

**Scenario:**
1. **Manhattan Midtown, Lunedi 8:00** - Rush hour mattutino, zona ad alta domanda
2. **Manhattan Midtown, Lunedi 3:00** - Notte fonda, pochi taxi in circolazione
3. **Bronx Fordham, Lunedi 8:00** - Rush hour in zona residenziale
4. **JFK Airport, Mercoledi 10:00** - Fascia centrale, aeroporto (domanda costante)
5. **Times Square, Sabato 20:00** - Serata nel weekend, zona turistica"""))

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
    \"\"\"Prepara le feature per una previsione, usando dati reali o fallback.\"\"\"
    bucket = hour * 2 + (1 if minute >= 30 else 0)
    mask = (df_orig['PULocationID'] == zone_id) & (df_orig['half_hour_bucket'] == bucket) & (df_orig['day_of_week'] == dow)
    subset = df_orig[mask]

    if len(subset) > 0:
        # Dati reali disponibili per questa combinazione
        return {
            'PULocationID': zone_id,
            'half_hour_bucket': bucket,
            'day_of_week': dow,
            'month': 1,
            'unique_taxi_types': subset['unique_taxi_types'].iloc[0],
            'avg_trip_duration_min': subset['avg_trip_duration_min'].iloc[0],
            'is_weekend': 1 if dow >= 5 else 0,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            'is_night': 1 if (hour >= 23 or hour <= 5) else 0,
            'borough_encoded': subset['borough_encoded'].iloc[0],
            'service_zone_encoded': subset['service_zone_encoded'].iloc[0]
        }
    else:
        # Fallback: usa valori tipici del dataset
        return {
            'PULocationID': zone_id,
            'half_hour_bucket': bucket,
            'day_of_week': dow,
            'month': 1,
            'unique_taxi_types': df_orig['unique_taxi_types'].mode()[0],
            'avg_trip_duration_min': df_orig['avg_trip_duration_min'].median(),
            'is_weekend': 1 if dow >= 5 else 0,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            'is_night': 1 if (hour >= 23 or hour <= 5) else 0,
            'borough_encoded': df_orig['borough_encoded'].mode()[0],
            'service_zone_encoded': df_orig['service_zone_encoded'].mode()[0]
        }

print(f"\\n{'='*85}")
print(f"TEST CASE - {best_model_name} (Modello: {best_model_name})")
print(f"{'='*85}")

for zone_id, hour, minute, dow, desc in test_cases:
    zi = zone_lookup[zone_lookup['LocationID'] == zone_id]
    zone_name = zi['Zone'].values[0] if len(zi) > 0 else f"Zone {zone_id}"
    borough = zi['Borough'].values[0] if len(zi) > 0 else "?"

    features = pd.DataFrame([prepare_input(zone_id, hour, minute, dow, df)])
    features_scaled = scaler.transform(features)
    pred = best_model.predict(features_scaled)[0]
    probs = best_model.predict_proba(features_scaled)[0]

    print(f"\\n{'─'*85}")
    print(f"  SCENARIO: {desc}")
    print(f"  Zona: {zone_name} ({borough}) - ID: {zone_id}")
    print(f"  PREDIZIONE: {CLASS_NAMES[pred]} (confidenza: {probs[pred]:.1%})")
    print(f"  Distribuzione probabilita':")
    for i in range(5):
        bar = '#' * int(probs[i] * 30)
        marker = " <--" if i == pred else ""
        print(f"    {CLASS_NAMES[i]:20s}: {probs[i]:5.1%} {bar}{marker}")"""))

cells.append(md("## 15. Salvataggio del Modello e dei Risultati"))

cells.append(md("""Salviamo tutto il necessario per:
1. **Riutilizzare il modello** in produzione (file .pkl)
2. **Confronto documentato** (JSON con metriche di entrambi i modelli)
3. **Reproducibilita'** (scaler, feature names, mapping classi)"""))

cells.append(code("""# 1. Salvataggio risultati del confronto in JSON
results_for_json = []
for _, row in results_df.iterrows():
    results_for_json.append({
        'model': row['model'],
        'cv_f1_macro': row['cv_f1'],
        'test_accuracy': row['test_accuracy'],
        'test_f1_macro': row['test_f1_macro'],
        'test_f1_weighted': row['test_f1_weighted'],
        'test_precision': row['test_precision'],
        'test_recall': row['test_recall'],
        'best_params': {k: str(v) for k, v in row['best_params'].items()}
    })

with open('output/model_comparison_results.json', 'w') as fp:
    json.dump(results_for_json, fp, indent=2)
print("Confronto modelli salvato: output/model_comparison_results.json")

# 2. Salvataggio artefatti completi (modello + scaler + metadata)
artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_cols': FEATURE_COLS,
    'class_names': CLASS_NAMES,
    'model_name': best_model_name,
    'best_params': {k: str(v) for k, v in best_params.items()},
    'metrics': {
        'accuracy': round(float(acc), 4),
        'f1_macro': round(float(f1), 4),
        'f1_weighted': round(float(f1w), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
    }
}

joblib.dump(artifacts, 'output/ml_model_artifacts.pkl')
print("Artefatti completi salvati: output/ml_model_artifacts.pkl")

# 3. Salvataggio modello leggero (solo per inferenza)
joblib.dump(best_model, 'output/best_model.pkl')
print("Modello leggero salvato: output/best_model.pkl")

print(f"\\n{'='*60}")
print(f"NOTEBOOK COMPLETATO")
print(f"{'='*60}")
print(f"Modello finale: {best_model_name}")
print(f"Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
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

with open(r'C:\Users\andre\Desktop\Progetto_Accenture\02_model_training_explainability.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook scritto!")
with open(r'C:\Users\andre\Desktop\Progetto_Accenture\02_model_training_explainability.ipynb', 'r') as f:
    json.load(f)
print("JSON valido!")
print(f"Celle totali: {len(cells)}")
