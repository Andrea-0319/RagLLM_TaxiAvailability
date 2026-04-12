import json, sys

for nb_file in ['03_multi_month_processing.ipynb', '04_lightgbm_advanced_training.ipynb']:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
    print(f"{nb_file}: {len(nb['cells'])} celle totali ({len(code_cells)} code, {len(md_cells)} markdown)")
    
    errors = 0
    for i, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        try:
            compile(source, f'{nb_file}:cell_{i}', 'exec')
        except SyntaxError as e:
            print(f"  ERRORE SINTASSI cella code #{i}: {e}")
            errors += 1
    
    if errors == 0:
        print("  Tutte le celle code sono sintatticamente valide")
    else:
        print(f"  {errors} celle con errori!")
