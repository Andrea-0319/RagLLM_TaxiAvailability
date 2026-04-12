import json

with open('04_lightgbm_advanced_training_executed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
errors = 0
for i, cell in enumerate(code_cells):
    outputs = cell.get('outputs', [])
    for out in outputs:
        if out.get('output_type') == 'error':
            ename = out.get('ename', '?')
            evalue = out.get('evalue', '?')
            print(f'ERRORE cella code #{i}: {ename}: {evalue}')
            errors += 1

if errors == 0:
    print('Nessun errore trovato in nessuna cella!')
print(f'Celle code: {len(code_cells)}')

# Extract key results
key_phrases = [
    'Miglior F1-Macro',
    'Migliori hyperparametri',
    'RISULTATI FINALI',
    'DIAGNOSI LEARNING CURVE',
    'Classification Report',
    'CONFRONTO:',
    'NOTEBOOK COMPLETATO',
    'Top 10 combinazioni',
    'Gap finale',
    'Learning curve calcolata',
    'Completato in'
]

for i, cell in enumerate(code_cells):
    outputs = cell.get('outputs', [])
    for out in outputs:
        if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
            text = ''.join(out.get('text', ''))
            for phrase in key_phrases:
                if phrase in text:
                    # Print the relevant lines
                    lines = text.strip().split('\n')
                    for line in lines:
                        if phrase in line or (phrase == 'RISULTATI FINALI' and any(k in line for k in ['Accuracy', 'F1-Macro', 'Tempo train'])):
                            print(f'\n[Cella #{i}] {line}')
                    break
