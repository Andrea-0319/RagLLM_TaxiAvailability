import json

with open('04_lightgbm_advanced_training_executed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

# Get full output from key cells
for i in [7, 8, 10, 11, 12, 13, 23, 24]:
    if i < len(code_cells):
        cell = code_cells[i]
        outputs = cell.get('outputs', [])
        for out in outputs:
            if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
                text = ''.join(out.get('text', ''))
                print(f"\n{'='*70}")
                print(f"=== CELLA CODE #{i} ===")
                print(f"{'='*70}")
                print(text)
