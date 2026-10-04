"""Portable, data-only exports for the three results notebooks."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    default=_json_default) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def export_section(root, dataset, section, *, metadata, tables=None,
                   arrays=None, stores=None):
    """Replace named section artifacts on rerun; never export figures or pickle.

    Array stores retain their method/seed/neighbourhood/K keys in JSON indexes.
    All sample-wise arrays use the row order saved in data_walkthrough/data.csv.
    """
    for name in (dataset, section):
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Dataset and section must be single folder names.')
    destination = Path(root) / dataset / section
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, table in (tables or {}).items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f'{name} must be a DataFrame')
        path = destination / f'{name}.csv'
        temporary = path.with_suffix('.csv.tmp')
        table.to_csv(temporary, index=False, encoding='utf-8', float_format='%.17g')
        os.replace(temporary, path)
        artifacts.append({'file': path.name, 'rows': len(table),
                          'columns': list(table.columns)})
    if arrays:
        path = destination / 'arrays.npz'
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(path.with_suffix('.npz.tmp'), path)
        artifacts.append({'file': path.name, 'arrays': list(arrays)})
    for name, (store, key_columns) in (stores or {}).items():
        packed, index = {}, []
        for number, (key, values) in enumerate(store.items()):
            key = key if isinstance(key, tuple) else (key,)
            if len(key) != len(key_columns):
                raise ValueError(f'{name}: key does not match {key_columns}')
            member = f'array_{number:06d}'
            values = np.asarray(values)
            if values.dtype.hasobject:
                raise TypeError(f'{name}: object arrays are not portable')
            packed[member] = values
            index.append({'array': member, **dict(zip(key_columns, key)),
                          'shape': list(values.shape), 'dtype': str(values.dtype)})
        path = destination / f'{name}.npz'
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **packed)
        os.replace(path.with_suffix('.npz.tmp'), path)
        _write_json(destination / f'{name}_index.json', index)
        artifacts.extend([{'file': path.name}, {'file': f'{name}_index.json'}])
    _write_json(destination / 'metadata.json', {
        'schema_version': 1, 'dataset_id': dataset, 'section': section,
        **metadata, 'artifacts': artifacts,
    })
    print(f'Saved {len(artifacts)} data artifacts: {destination}')
    return destination
