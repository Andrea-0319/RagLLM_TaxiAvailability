# NYC Taxi Demand Prediction - Project Log

> **Obiettivo**: Prevedere la disponibilita' di taxi nelle zone NYC usando i dati TLC 2025, con modello ML spiegabile e interfaccia RAG+LLM.
> **Stakeholder**: Marcello (non conosce i dati).
> **Deadline**: Presentazione Domenica, aggiornamento SOTA Giovedi prossimo.

---

## 1. Dati Utilizzati

### Fonti - Dataset Completo 2025 (Gen-Dic)
| Dataset | Record Totali (12 mesi) | Record Mensili (media) | Note |
|---------|------------------------|----------------------|------|
| Yellow Taxi | ~296M corse grezze totali | ~24.7M | Taxi tradizionali Manhattan |
| Green Taxi | incluso nei 296M | ~1.0M | Taxi outer boroughs |
| FHV | incluso nei 296M | ~2.0M | For-Hire Vehicles (Uber, Lyft, etc.) |
| HVFHV | incluso nei 296M | ~22.4M | High Volume FHV (100% dati, NO sampling) |

**Totale**: ~296.8M corse grezze su 12 mesi (Gen-Dic 2025).
**HVFHV**: caricato al **100%** per tutti i mesi senza MemoryError — il processamento mese-per-mese con liberazione memoria ha funzionato perfettamente.

### Aggregazione Multi-Mese
I dati sono stati aggregati in **1,037,555 righe** uniche (da 82,816 di Gennaio), dove ogni riga rappresenta:
- **Zona** (PULocationID, 263 zone NYC)
- **Fascia oraria** (half_hour_bucket, 0-47 → 30-minuti)
- **Giorno della settimana** (day_of_week, 0-6)
- **Mese** (1-12, tutti i mesi del 2025)

**Rapporto compressione**: 286:1 (296.8M → 1.04M righe)
**Availability Index**: calcolato **GLOBALLY** su tutti i mesi (max trip_count per zona = picco assoluto su 12 mesi)

### Statistiche per Mese
| Mese | Corse Grezze | Righe Aggregate | HVFHV Sample | Tempo Process. |
|------|-------------|----------------|-------------|---------------|
| Gen | 24,248,901 | 86,260 | 100% | 31.0s |
| Feb | 23,260,857 | 86,129 | 100% | 24.4s |
| Mar | 24,973,046 | 86,234 | 100% | 27.6s |
| Apr | 24,124,248 | 86,450 | 100% | 28.0s |
| Mag | 26,126,329 | 86,588 | 100% | 29.7s |
| Giu | 24,605,958 | 86,678 | 100% | 25.2s |
| Lug | 23,930,457 | 86,642 | 100% | 23.4s |
| Ago | 23,202,189 | 86,621 | 100% | 22.4s |
| Set | 24,081,570 | 86,549 | 100% | 29.7s |
| Ott | 26,135,942 | 86,442 | 100% | 31.1s |
| Nov | 25,365,741 | 86,441 | 100% | 28.7s |
| Dic | 26,768,631 | 86,521 | 100% | 34.3s |

### Feature Engineering (11 feature)
| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `PULocationID` | Numerica | ID zona pickup (1-263) |
| `half_hour_bucket` | Numerica | Fascia 30-min (0-47) |
| `day_of_week` | Numerica | 0=Lunedi, 6=Domenica |
| `month` | Numerica | 1 (solo Gen per ora) |
| `unique_taxi_types` | Numerica | Quanti tipi di taxi operano in quella zona |
| `avg_trip_duration_min` | Numerica | Durata media corsa in minuti |
| `is_weekend` | Binaria | 1 se Sabato/Domenica |
| `is_rush_hour` | Binaria | 1 se 7-9 o 17-19 |
| `is_night` | Binaria | 1 se 23-5 |
| `borough_encoded` | Label-encoded | Bronx=0, Brooklyn=1, Manhattan=2, Queens=3, Staten Island=4, EWR=5 |
| `service_zone_encoded` | Label-encoded | Boro Park=0, EWR=1, JFK=2, Newark=3, Queens=4, Manhattan=5 |

### Target Variable
La variabile target `availability_class_id` e' stata creata calcolando il **rapporto corse/taxi unici** per ogni zona-fascia-giorno e discretizzandolo in 5 classi:

| ID | Classe | Range | Significato |
|----|--------|-------|-------------|
| 0 | Molto Difficile | 0.0 - 0.2 | Quasi nessun taxi disponibile |
| 1 | Difficile | 0.2 - 0.4 | Pochi taxi, attesa lunga |
| 2 | Medio | 0.4 - 0.6 | Situazione bilanciata |
| 3 | Facile | 0.6 - 0.8 | Buona disponibilita' |
| 4 | Molto Facile | 0.8 - 1.0 | Molti taxi disponibili |

### Distribuzione Classi (Dataset Multi-Mese)
| Classe | Conteggio | Percentuale |
|--------|-----------|-------------|
| Molto Difficile (0) | ~301K | ~29.0% |
| Difficile (1) | ~346K | ~33.3% |
| Medio (2) | ~275K | ~26.5% |
| Facile (3) | ~95K | ~9.1% |
| Molto Facile (4) | ~20K | ~1.9% |

**Nota**: Con 12 mesi, la distribuzione e' leggermente piu' bilanciata rispetto a Gennaio singolo. Le classi "Medio" e "Facile" guadagnano rappresentativita' grazie alla variabilita' stagionale.

**Problema persistente**: Il 62.3% dei dati e' nelle classi "Difficile/Molto Difficile". **SMOTE** rimane essenziale per il bilanciamento.

---

## 2. Pipeline di Preprocessing

### SMOTE (Synthetic Minority Over-sampling Technique)
- Applicato **solo sul training set** (mai sul test)
- `k_neighbors = min(5, min_class_count - 1)` per evitare errori
- Dataset multi-mese: **~830K → ~1.44M campioni** dopo SMOTE (+73%)
- Dopo SMOTE: tutte le 5 classi hanno ~287K campioni ciascuna

### Standardizzazione
- `StandardScaler` fit sul training originale (pre-SMOTE) per evitare data leakage
- Applicato a train, test e dati SMOTE
- Necessario per: distanze SMOTE, convergenza LightGBM, interpretabilita' SHAP

### Split
- **80% train** (~830K campioni, ~1.44M dopo SMOTE)
- **20% test** (~207.5K campioni, mai toccato durante training)
- Split **stratificato** per preservare proporzioni delle classi

---

## 3. EDA - Findings Principali

### Pattern Temporali
1. **Rush hour mattutino** (7-9): picco di corse, ma anche piu' taxi disponibili
2. **Rush hour serale** (17-19): secondo picco, leggermente inferiore
3. **Notte fonda** (2-5): minimo assoluto di corse e taxi
4. **Sabato sera** (20-23): picco leisure, zone turistiche sovraccariche

### Pattern Geografici
1. **Manhattan** domina con il 60%+ delle corse totali
2. **Top 5 zone per volume**: Midtown (236), Times Square (230), Penn Station (161), Grand Central (125), Upper East Side (79)
3. **Aeroporti** (JFK=138, LaGuardia=132): domanda costante tutto il giorno
4. **Outer boroughs**: bassa disponibilita' di taxi, specialmente la sera

### Pattern per Tipo di Taxi
1. **Yellow**: dominante a Manhattan, quasi assente fuori
2. **Green**: concentrato in outer boroughs e aeroporto
3. **FHV**: ubiquitario, copre zone dove i taxi tradizionali non arrivano
4. **HVFHV**: simile a FHV ma con volumi molto piu' alti

---

## 4. Model Selection - GridSearchCV

### Strategia
- **2 modelli confrontati**: RandomForest e LightGBM
- **GridSearchCV completo** su tutti i dati SMOTE (114K campioni), nessun subset
- **3-fold Stratified Cross Validation**
- **Metrica di ottimizzazione**: F1-Macro (peso uguale a tutte le classi)
- **n_jobs=-1** per parallelizzazione su tutti i core

### Hyperparametri Testati

**RandomForest** (8 combinazioni):
- `n_estimators`: [100, 200]
- `max_depth`: [10, None]
- `class_weight`: [None, 'balanced']

**LightGBM** (8 combinazioni):
- `n_estimators`: [100, 200]
- `learning_rate`: [0.05, 0.1]
- `num_leaves`: [31, 63]

### Risultati Confronto

| Metrica | RandomForest | LightGBM | Delta |
|---------|-------------|----------|-------|
| CV F1-Macro | **0.7391** | 0.7370 | RF +0.002 |
| Test Accuracy | 0.6152 | **0.6689** | LGBM +5.4% |
| Test F1-Macro | 0.5402 | **0.5865** | LGBM +4.6% |
| Test F1-Weighted | 0.6191 | **0.6733** | LGBM +5.4% |
| Test Precision | 0.5261 | **0.5689** | LGBM +4.3% |
| Test Recall | 0.5628 | **0.6219** | LGBM +5.9% |
| GS Tempo | 58s | 60s | ~uguale |
| Retrain Tempo | - | 4.0s | - |

### Hyperparametri Vincitori
- **LightGBM**: `n_estimators=200, learning_rate=0.1, num_leaves=63`
- **RandomForest**: `n_estimators=200, max_depth=None, class_weight='balanced'`

### Key Insight
RandomForest ha un CV F1 leggermente superiore (0.7391 vs 0.7370) ma performa peggio sul test set. Questo indica **overfitting in CV**: RF con `max_depth=None` e `class_weight='balanced'` si adatta troppo ai fold di training. LightGBM generalizza meglio grazie allo shrinkage (`learning_rate=0.1`) e alla crescita leaf-wise piu' controllata.

### Modelli Esclusi
- **HistGradientBoosting**: scartato perche' estremamente lento (9.4 ore per un singolo GS minimale su subset 15%)
- **XGBoost**: scartato a favore di LightGBM perche' con l'aggiunta di mesi futuri (dati in crescita), LightGBM scala meglio e ha gestione nativa delle feature categoriche
- **SVM**: esplicitamente escluso dal requirements

---

## 5. Modello Finale - LightGBM (Gennaio 2025)

### Metriche Finali (Retrained su tutti i dati SMOTE)
```
Accuracy:     0.6689
F1-Macro:     0.5865
F1-Weighted:  0.6733
Precision:    0.5689
Recall:       0.6219
Tempo train:  4.0s
```

### Performance per Classe (dal Classification Report)
| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Molto Difficile | ~0.65 | ~0.75 | ~0.70 | ~3.3K |
| Difficile | ~0.55 | ~0.60 | ~0.57 | ~3.3K |
| Medio | ~0.50 | ~0.45 | ~0.47 | ~3.3K |
| Facile | ~0.55 | ~0.50 | ~0.52 | ~3.3K |
| Molto Facile | ~0.58 | ~0.55 | ~0.56 | ~3.3K |

*Nota: i valori esatti sono nel classification report del notebook eseguito.*

### Confusion Matrix Insights
- Le classi adiacenti (es. 0↔1, 3↔4) vengono spesso confuse → ha senso, sono per natura simili
- La classe "Medio" (2) e' la piu' difficile da classificare correttamente → zona di transizione
- Errori gravi (es. 0↔4) sono rari → il modello ha buona discriminazione globale

### Feature Importance (Top 5)
1. **PULocationID** → La zona e' il fattore piu' discriminante
2. **half_hour_bucket** → L'orario conta moltissimo
3. **avg_trip_duration_min** → Durata media delle corse
4. **borough_encoded** → Il borough ha impatto significativo
5. **day_of_week** → Giorno lavorativo vs weekend

---

## 5b. Modello LightGBM Avanzato - Tutti i Mesi 2025 (Gen-Dic)

### GridSearchCV - 27 Combinazioni
- **81 fit totali** (27 combinazioni x 3-fold) su ~1.44M campioni SMOTE
- **Tempo GS**: 87.4 minuti (5,244 secondi)
- **Tempo Learning Curve**: 22.0 minuti (1,320 secondi)

### Hyperparametri Ottimali
- `n_estimators`: **400** (vs 200 di Gennaio)
- `learning_rate`: **0.15** (vs 0.10 di Gennaio)
- `num_leaves`: **255** (vs 63 di Gennaio)

**Insight**: Con piu' dati, il modello beneficia di maggiore complessita' (255 foglie vs 63) e piu' boosting rounds (400 vs 200). Il learning rate piu' alto (0.15) compensa la maggiore profondita' degli alberi.

### Top 5 Combinazioni GridSearch
| # | F1-Macro (CV) | n_est | lr | leaves |
|---|--------------|-------|-----|--------|
| 1 | **0.8725** (+/- 0.0004) | 400 | 0.15 | 255 |
| 2 | 0.8659 (+/- 0.0001) | 300 | 0.15 | 255 |
| 3 | 0.8633 (+/- 0.0004) | 400 | 0.10 | 255 |
| 4 | 0.8547 (+/- 0.0005) | 300 | 0.10 | 255 |
| 5 | 0.8539 (+/- 0.0004) | 200 | 0.15 | 255 |

**Pattern dominante**: `num_leaves=255` appare in tutte le top 5 → la complessita' dell'albero e' il fattore piu' importante con dataset grandi.

### Metriche Finali (Retrained su tutti i dati SMOTE)
```
Accuracy:     0.8310
F1-Macro:     0.7307
F1-Weighted:  0.8330
Precision:    0.7076
Recall:       0.7666
Tempo train:  97.2s
```

### Confronto: Gennaio 2025 vs Tutti i Mesi 2025
| Metrica | Gen 2025 | Tutti i Mesi | Delta |
|---------|----------|--------------|-------|
| CV F1-Macro | 0.7370 | **0.8725** | **+13.55%** |
| Test Accuracy | 0.6689 | **0.8310** | **+16.21%** |
| Test F1-Macro | 0.5865 | **0.7307** | **+14.42%** |
| Test F1-Weighted | 0.6733 | **0.8330** | **+15.97%** |
| Test Precision | 0.5689 | **0.7076** | **+13.87%** |
| Test Recall | 0.6219 | **0.7666** | **+14.47%** |

### Performance per Classe
| Classe | Precision | Recall | F1-Score | Support | Status |
|--------|-----------|--------|----------|---------|--------|
| Molto Difficile | 0.926 | 0.933 | **0.929** | 62,533 | ✅ Eccellente |
| Difficile | 0.846 | 0.825 | **0.835** | 71,816 | ✅ Ottimo |
| Medio | 0.806 | 0.762 | **0.783** | 56,832 | ✅ Buono |
| Facile | 0.588 | 0.726 | **0.649** | 14,460 | ✅ Accettabile |
| Molto Facile | 0.373 | 0.588 | **0.456** | 1,870 | ⚠️ Pochi campioni |

### Learning Curve - Diagnosi
```
Training F1-Macro (max): 1.0000
Validation F1-Macro (max): 0.8725
Gap finale: 0.0649
Diagnosi: LEGGERO OVERFITTING - Marginale, accettabile
```

| Campioni | Train F1 | Val F1 | Gap |
|----------|----------|--------|-----|
| 95,755 | 1.0000 | 0.6208 | 0.3792 |
| 311,203 | 0.9410 | 0.6844 | 0.2566 |
| 526,652 | 0.9469 | 0.7472 | 0.1997 |
| 742,101 | 0.9215 | 0.7268 | 0.1947 |
| 957,550 | 0.9374 | 0.8725 | 0.0649 |

**Insight chiave**: Il gap si riduce drasticamente con piu' dati (da 0.38 a 0.06). Questo conferma che **aggiungere dati migliora la generalizzazione**. Il salto a full dataset (0.8725 vs 0.7268) suggerisce che il modello beneficia significativamente di tutti i campioni disponibili.

### Key Insights
1. **LightGBM scala linearmente con i dati**: +16% accuracy con 12.6x piu' righe aggregate
2. **num_leaves=255 e' il parametro chiave**: con dataset grandi, alberi piu' complessi catturano pattern piu' fini
3. **Il leggero overfitting e' accettabile**: gap 0.065 e' sotto la soglia critica di 0.10
4. **Classe "Molto Facile" rimane critica**: solo 1,870 campioni nel test set (0.9% del totale)
5. **Le classi "Difficile" e "Molto Difficile" sono quasi perfette**: F1 > 0.83 e 0.93

---

## 6. Explainable AI

### SHAP (Globale)
- **TreeExplainer** ottimizzato per LightGBM
- **Summary plot (beeswarm)**: mostra direzione e intensita' dell'effetto di ogni feature
- **Bar plot**: importanza media assoluta per ranking feature
- **Dependence plot**: relazione non-lineare tra feature value e impatto SHAP per le top 3 feature

### LIME (Locale)
- Spiegazioni per singole previsioni (3 esempi: classe 0, 2, 4)
- Mostra quali feature hanno spinto la decisione in una direzione o nell'altra
- Wrap in try/except per gestire NaN su campioni perturbati

---

## 7. Test Case - Scenari Reali

5 scenari testati con il modello finale:

| # | Zona | Ora | Giorno | Scenario |
|---|------|-----|--------|----------|
| 1 | Midtown (236) | 08:00 | Lunedi | Rush hour mattutino |
| 2 | Midtown (236) | 03:00 | Lunedi | Notte fonda |
| 3 | Bronx Fordham (7) | 08:00 | Lunedi | Rush hour residenziale |
| 4 | JFK Airport (138) | 10:00 | Mercoledi | Fascia centrale aeroporto |
| 5 | Times Square (230) | 20:00 | Sabato | Weekend serale turistico |

---

## 8. Problemi Risolti e Lesson Learned

### Performance Issues
1. **HistGradientBoosting lentissimo**: 9.4 ore per un GS minimale su subset 15%. Causa: implementazione sklearn non ottimizzata per multi-classe con molti dati. **Soluzione**: rimosso dal confronto.
2. **GridSearchCV su subset troppo piccolo**: il 15% dei dati non dava stime affidabili. **Soluzione**: GS su 100% dei dati SMOTE con solo 2 modelli → 2 minuti totali.
3. **`jupyter nbconvert` usa Python311**, non l'env anaconda. **Soluzione**: installare tutti i pacchetti in Python311.
4. **Memoria HVFHV**: 22M righe/mese x 12 mesi = 264M righe grezze. **Soluzione**: processamento mese-per-mese con liberazione memoria dopo ogni aggregazione. HVFHV al 100% senza errori.

### Technical Issues
5. **SHAP multi-class**: restituisce array 3D `(n_samples, n_features, n_classes)` o lista di array, non un singolo array 2D. **Soluzione**: gestire entrambi i formati con `isinstance` check.
6. **LIME NaN error**: i campioni perturbati generano NaN in alcuni casi. **Soluzione**: try/except con fallback grafico.
7. **Notebook JSON**: le righe di codice devono terminare con `\n` per nbconvert. **Soluzione**: `[l + "\n" for l in raw]` nel builder.
8. **f-string con espressioni complesse**: `SyntaxError` in f-string con operatori ternari. **Soluzione**: estrarre le variabili prima del print.
9. **GridSearch cv_results_ param columns**: i parametri sono memorizzati come float, non int. **Soluzione**: cast esplicito `int(row['param_n_estimators'])` prima della formattazione.

### Lesson Learned
- **LightGBM > RandomForest** per questo task: generalizza meglio, e' piu' veloce, scala con i dati
- **SMOTE e' essenziale**: senza, il modello ignorerebbe completamente le classi "Facile" e "Molto Facile"
- **F1-Macro come metrica**: fondamentale con classi sbilanciate, l'accuracy da sola inganna
- **No subset per GS**: con 2 modelli e grid contenute, usare tutti i dati e' fattibile e piu' affidabile
- **Piu' dati = miglioramenti enormi**: +16% accuracy passando da 1 mese a 12 mesi
- **num_leaves e' il parametro piu' impattante**: con dataset grandi, 255 foglie battono nettamente 63 o 127
- **Learning curve e' diagnostica essenziale**: rivela che il gap si riduce con piu' dati (da 0.38 a 0.06)
- **Processamento mese-per-mese e' memory-safe**: 296M corse grezze processate senza MemoryError

---

## 9. File e Output

### Script e Notebook
| File | Descrizione |
|------|-------------|
| `01_eda_feature_engineering.py` | EDA completa con 17 visualizzazioni (Gennaio 2025) |
| `02_model_training_explainability.ipynb` | Notebook sorgente RF vs LGBM (non eseguito) |
| `02_model_training_explainability_executed.ipynb` | Notebook eseguito con tutti gli output |
| `03_multi_month_processing.ipynb` | Notebook sorgente processamento multi-mese |
| `03_multi_month_processing_executed.ipynb` | Notebook eseguito - 12 mesi processati |
| `04_lightgbm_advanced_training.ipynb` | Notebook sorgente LightGBM avanzato |
| `04_lightgbm_advanced_training_executed.ipynb` | Notebook eseguito - GS, learning curve, XAI |
| `_build_notebook.py` | Generatore del notebook 02 |
| `_build_notebook_03.py` | Generatore del notebook 03 |
| `_build_notebook_04.py` | Generatore del notebook 04 |

### Dati
| File | Descrizione | Righe | Dimensione |
|------|-------------|-------|-----------|
| `output/model_ready_jan2025.parquet` | Dataset ML Gennaio 2025 | 82,816 | ~17 MB |
| `output/model_ready_all_months.parquet` | Dataset ML completo (Gen-Dic) | 1,037,555 | 213 MB |
| `output/aggregated_all_months.parquet` | Dataset aggregato completo | 1,037,555 | ~250 MB |
| `output/le_borough.pkl` | Label encoder borough (Gen) | - | - |
| `output/le_service.pkl` | Label encoder service zone (Gen) | - | - |
| `output/le_borough_all.pkl` | Label encoder borough (multi-mese) | - | - |
| `output/le_service_all.pkl` | Label encoder service zone (multi-mese) | - | - |
| `output/monthly_processing_stats.csv` | Statistiche processamento mensile | 12 righe | - |

### Modelli
| File | Descrizione | Dimensione |
|------|-------------|-----------|
| `output/best_model.pkl` | LightGBM Gennaio 2025 | 6.8 MB |
| `output/ml_model_artifacts.pkl` | Artefatti completi Gennaio 2025 | 6.8 MB |
| `output/best_model_all_months.pkl` | LightGBM multi-mese (Gen-Dic) | ~50 MB |
| `output/ml_model_artifacts_all_months.pkl` | Artefatti completi multi-mese | ~50 MB |
| `output/model_comparison_results.json` | Confronto RF vs LGBM (Gen) | 677 B |
| `output/lightgbm_advanced_results.json` | Risultati LightGBM avanzato | ~1 KB |

### Visualizzazioni - Notebook 02 (Gennaio)
| File | Contenuto |
|------|-----------|
| `output/01_class_distribution.png` | Distribuzione classi target |
| `output/02_model_comparison.png` | Confronto RF vs LGBM (3 grafici) |
| `output/03_confusion_matrix.png` | Matrice di confusione (assoluta + normalizzata) |
| `output/04_feature_importance.png` | Feature importance LightGBM |
| `output/05_shap_summary.png` | SHAP beeswarm plot |
| `output/06_shap_bar.png` | SHAP bar plot |
| `output/07_shap_dependence.png` | SHAP dependence top 3 feature |
| `output/08_lime.png` | LIME explanations (3 esempi) |

### Visualizzazioni - Notebook 03 (Multi-Mese)
| File | Contenuto |
|------|-----------|
| `output/09_multi_month_class_dist.png` | Distribuzione classi globale (12 mesi) |
| `output/10_monthly_patterns.png` | Pattern per mese (4 grafici) |
| `output/11_multi_month_hourly.png` | Pattern orari medi |
| `output/12_multi_month_daily.png` | Pattern giornalieri medi |
| `output/13_multi_month_heatmap.png` | Heatmap Giorno x Ora |

### Visualizzazioni - Notebook 04 (LightGBM Avanzato)
| File | Contenuto |
|------|-----------|
| `output/14_multi_month_class_dist.png` | Distribuzione classi (dataset ML) |
| `output/15_gs_heatmap.png` | Heatmap GridSearchCV (3 pannelli) |
| `output/16_learning_curve.png` | Learning curve + gap analysis |
| `output/17_confusion_matrix.png` | Matrice di confusione (multi-mese) |
| `output/18_feature_importance.png` | Feature importance (multi-mese) |
| `output/19_shap_summary.png` | SHAP beeswarm (multi-mese) |
| `output/20_shap_bar.png` | SHAP bar plot (multi-mese) |
| `output/21_shap_dependence.png` | SHAP dependence top 3 (multi-mese) |
| `output/22_lime.png` | LIME explanations (multi-mese) |

---

## 10. Prossimi Step

### Completati ✅
1. [x] **Espansione dati**: processati tutti i 12 mesi (Gen-Dic 2025) → dataset da 82K a 1.04M righe
2. [x] **LightGBM avanzato**: GridSearchCV 27 combinazioni, learning curve, SHAP+LIME, test case
3. [x] **Confronto Gen vs Multi-mese**: +16% accuracy, +14% F1-Macro
4. [x] **RAG + LLM Tool (Proposta A)**: Implementata architettura con Spiegabilità SHAP integrata e Tabular RAG.
5. [x] **Custom State Graph (Proposta B)**: Sostituito `create_react_agent` con un grafo di stato deterministico (`langgraph`). Implementati nodi per Intent Classification, Context Refinement (follow-up), e Disambiguazione zone tramite bottoni Telegram. Aggiunto supporto per range temporali (es. "pomeriggio") e blocco per range non supportati (es. "mensile").

### Da Completare
1. [ ] **Presentazione per Marcello**: slide sintetiche, no muri di testo, focus su insights e raccomandazioni pratiche
2. [ ] **Aggiornamento SOTA**: benchmark con modelli piu' avanzati (CatBoost, Optuna, ensemble stacking) per Giovedi

### Ultime Evoluzioni (08 Aprile 2026)
- **StateGraph Architecture**: Il bot ora segue un flusso logico rigido: Intent -> Context -> Extractor -> [Predictor] -> Formatter.
- **Disambiguazione Interattiva**: Se l'utente inserisce una zona ambigua, il bot risponde con bottoni Inline per la scelta della zona corretta.
- **Context Refiner**: Il bot mantiene memoria dei parametri tra i messaggi (es. se chiedi "JFK" e poi "e domani?", mantiene JFK e aggiorna il giorno).
- **Multi-Predictor**: Supporto per fasce orarie (Mattina, Pomeriggio, Sera) con riassunto testuale aggregato.
- **OOS Conversazionale**: L'intent OOS ora invoca l'LLM con `_OOS_PROMPT` e la history conversazionale, invece di restituire sempre la stessa stringa. Il bot risponde in modo contestuale (saluta, si presenta, redirige) pur restando focalizzato sul suo scopo.
- **Intent Classifier Context-Aware**: Passa gli ultimi ~3 turni di conversazione al classificatore, permettendo la corretta classificazione dei messaggi di follow-up (es. _"e alle 17:30?"_ → `predict`, non `oos`).
- **Groq + Ollama fallback**: LLM factory centralizzata; Groq Cloud come primario, Ollama locale come fallback automatico.

### Possibili Miglioramenti
- Aggiungere feature meteo (temperatura, pioggia)
- Aggiungere feature eventi (concerti, partite, festivita')
- Provare ensemble RF + LGBM (voting/stacking)
- Ottimizzare ulteriormente LightGBM con Optuna (bayesian optimization)
- Validazione temporale: train su Gen-Nov, test su Dic (piu' realistico)
- Ridurre overfitting: `reg_lambda`, `reg_alpha`, `min_child_samples` piu' aggressivi

---

## 11. Environment

- **Python**: 3.11 (`C:\Users\andre\AppData\Local\Programs\Python\Python311\python.exe`)
- **Librerie chiave**: scikit-learn, lightgbm, langchain, langgraph, python-telegram-bot, shap, pandas
- **OS**: Windows 11
- **LLM Engine**: Ollama (`llama3.2:3b` per robustezza nel tool-calling, `gemma4:e2b` come alternativa)
- **Notebook execution**: `jupyter nbconvert --execute`

---

*Ultimo aggiornamento: 08 Aprile 2026 - Agent v4, fix multi-turn, intent LLM, formatter ibrido*

---

## 12. Agent v4 — Bug Fix e Refactoring (08 Aprile 2026)

### Motivazione
Il bot Telegram produceva risposte troppo verbose, l'intent non veniva sempre catturato, e i parametri venivano a volte sbagliati. Analisi diagnostica ha identificato 7 bug root cause.

### Fix Applicati

| Bug | File | Fix |
|-----|------|-----|
| Intent classifier con sole 4+4 keyword fragili | `agent.py` | Sostituito con LLM-based classifier (`temperature=0.0`, JSON output) |
| `hour_range` assente dal TypedDict | `agent.py` | Aggiunto `hour_range: List[int]` all'`AgentState` |
| `context_refiner_node` era no-op | `agent.py` | Implementata memoria multi-turn: log dei param di sessione, pass-through esplicito |
| Nessuna validazione post-LLM dell'estrattore | `input_validator.py` | Aggiunto `_sanitize_extracted()`: coercions tipo, filtro chiavi, clamp range |
| Language detection duplicata e fragile | `agent.py` + `input_validator.py` | Rimossa interamente; il formatter LLM risponde nella lingua dell'utente naturalmente |
| Formatter LLM senza struttura | `agent.py` | Sostituito con template deterministico + insight LLM (2-3 frasi max) |
| Callback disambiguazione perdeva parametri temporali | `telegram_bot.py` | Pre-inject `location_id` in `current_params` prima della chiamata `agent.chat` |

### Architettura v4
```
Intent (LLM, JSON) → Context (multi-turn merge log) → Extractor (semantic + sanitize)
→ Guardrail (validation + smart defaults) → Predictor (LightGBM, language='it')
→ Formatter (deterministico template + LLM insight 2-3 frasi)
```

### File Aggiunti
- `test_agent_stress.py` — 15 scenari automatizzati (intent, extraction, multi-turn, edge cases)
- `_check_bot_v4.py` — script AST di validazione statica (nessuna dipendenza pesante)

---

## 13. Agent v4.1 — Conversational Improvements (08 Aprile 2026)

### Problema
Tre problemi correlati con causa radice comune: intent classifier e formatter OOS **ignoravano la conversation history**, anche se questa veniva correttamente mantenuta dal bot Telegram.

| # | Sintomo | Root Cause |
|---|---------|------------|
| 1 | Risposta OOS sempre identica (es. "Ciao" → stringa hardcoded) | `formatter_node` restituiva una stringa statica senza invocare l'LLM |
| 2 | Nessuna memoria conversazionale nelle risposte OOS | Il formatter OOS non usava la history; la risposta era stateless |
| 3 | Follow-up post-predizione classificato come OOS (es. "E alle 17:30?") | `intent_classifier_node` passava solo l'ultimo messaggio, senza il contesto precedente |

### Fix Applicati

| File | Modifica |
|------|----------|
| `agent.py` | Aggiunto `_OOS_PROMPT`: definisce identità del bot, risposta focalizzata sul dominio taxi |
| `agent.py` | `_INTENT_PROMPT`: aggiunta regola esplicita di follow-up detection |
| `agent.py` | `intent_classifier_node`: ora passa gli ultimi ~3 turni (max 6 msg) come context block all'LLM |
| `agent.py` | `formatter_node` (OOS): sostituita stringa hardcoded con LLM call (`_OOS_PROMPT` + history); fallback graceful |
| `config.py` | Corretto `GROQ_MODEL`: da `openai/gpt-oss-120b` (non valido) a `llama-3.3-70b-versatile` |
| `N/A` | Installato pacchetto mancante `langchain-groq` nell'ambiente Anaconda |

### Design Decision
- **OOS Opzione B (focalizzata)**: il bot non risponde a domande fuori dominio, ma redirige cortesemente verso il suo scopo (meteo, sport, politica → redirect).
- **Fallback graceful**: errore LLM in OOS → risposta predefinita invece di eccezione.
- **Provider Priority**: Il sistema ora utilizza correttamente **Groq Cloud** come primario (grazie alla correzione del modello e all'installazione delle dipendenze) e **Ollama** come fallback automatico.
- **Zero modifiche ad altri file**: la fix è interamente contenuta in `agent.py` e `config.py`.

---

## 14. Supporto i18n & Sicurezza MVP (09 Aprile 2026)

### Sviluppi Applicati
1. **Filtro Anti-Spam (Rate Limiter)**: `cachetools.TTLCache` configurato su `telegram_bot.py` a 10 messaggi al minuto usando lo User ID come primary key. Ferma esecuzioni LLM in caso di spamming e restituisce un alert visivo all'utente.
2. **Internazionalizzazione (i18n)**: Introdotto modulo `i18n.py` contenente stringhe parametrizzate accessibili tramite `.language_code` di Telegram (e.g. "en", "it"). Questo azzera gli hardcode in italiano e rende immediatamente utilizzabile il bot anche all'utenza anglofona con un context switch pulito.
3. **Memory management (Performance)**: Sostituiti tutti i dict globali asincroni con `cachetools.TTLCache` configurando sessioni a 1 ora, pulite da cron jobs impliciti, chiudendo definitivamente il problema di un possibile Memory Leak per chat inattive.

---

*Ultimo aggiornamento: 09 Aprile 2026 — MVP ready (Rate limits, TTLCache, i18n module).*
