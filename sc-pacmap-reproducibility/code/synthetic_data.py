"""Import the shared synthetic-data definitions from their reviewable notebook.

Only explicitly tagged definition cells execute on import. Inspection cells,
benchmark fits, and file exports never run here. NumPy and pandas are the only
dependencies of the data generator; Jupyter is not required.
"""

import json
from pathlib import Path


def _load_definitions():
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'synthetic_data.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    namespace = {'__name__': 'synthetic_data_definitions'}
    definition_cells = [
        cell for cell in notebook['cells']
        if cell['cell_type'] == 'code'
        and 'synthetic-data-definitions' in cell.get('metadata', {}).get('tags', [])
    ]
    if not definition_cells:
        raise ValueError(f'No synthetic-data definition cells found in {notebook_path}')
    for index, cell in enumerate(definition_cells):
        source = ''.join(cell['source'])
        exec(compile(source, f'{notebook_path}:definition-{index}', 'exec'), namespace)
    required = {'make_synthetic_repeated_facies', 'FEATURE_COLUMNS', 'XY_COLUMNS', 'N_SAMPLES'}
    if not required <= namespace.keys():
        raise ValueError(f'Missing synthetic-data definitions: {required - namespace.keys()}')
    return {name: value for name, value in namespace.items() if not name.startswith('_')}


globals().update(_load_definitions())
