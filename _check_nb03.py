import json

with open('03_multi_month_processing_executed.ipynb', 'r', encoding='utf-8') as f:
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

# Also check for stdout output to see key stats
for i, cell in enumerate(code_cells):
    outputs = cell.get('outputs', [])
    for out in outputs:
        if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
            text = ''.join(out.get('text', ''))
            if 'PROCESSAMENTO MULTI-MESE COMPLETATO' in text or 'Dataset finale:' in text:
                print(f'\n--- Output cella #{i} (riepilogo finale) ---')
                print(text)
            if 'TOTALE corse grezze' in text or 'TOTALE righe aggregate' in text:
                print(f'\n--- Output cella #{i} (totali) ---')
                print(text)
