"""Check exported data and rebuild combined figures without fitting embeddings.

Run from the project root: python code/validate_result_exports.py
This command does not modify exports or notebook files.
"""

import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / 'results' / 'notebook_exports'
KEYS = ['seed', 'n_neighbors', 'method', 'K']


def require(condition, context):
    if not condition:
        raise ValueError(context)


def validate_dataset(dataset, samples, reference_k, max_k):
    folder = EXPORTS / dataset
    tables = {}
    for path in sorted(folder.glob('*/metadata.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        require(meta['sample_count'] == samples, f'{path}: sample count')
        require(meta['reference_seed'] == 25, f'{path}: example seed')
        require(meta['seeds'] == list(range(30)), f'{path}: algorithm seeds')
        require(meta['embedding_neighbors'] == [10, 15, 20], f'{path}: fitting neighbourhoods')
        require(meta['K_reference'] == reference_k, f'{path}: reference K')
        for artifact in meta['artifacts']:
            target = path.parent / artifact['file']
            require(target.is_file(), f'Missing {target}')
            if target.suffix == '.csv':
                table = pd.read_csv(target)
                require(len(table) == artifact['rows'], f'{target}: row count')
                require(table.columns.tolist() == artifact['columns'], f'{target}: schema')
                tables[(path.parent.name, target.name)] = table
            elif target.suffix == '.npz':
                index_path = target.with_name(target.stem + '_index.json')
                index = json.loads(index_path.read_text()) if index_path.exists() else None
                with np.load(target, allow_pickle=False) as archive:
                    expected = [item['array'] for item in index] if index is not None else artifact['arrays']
                    require(set(archive.files) == set(expected), f'{target}: array members')
                    for name in archive.files:
                        require(np.isfinite(archive[name]).all(), f'{target}:{name}: nonfinite values')
                    for item in index or []:
                        values = archive[item['array']]
                        require(list(values.shape) == item['shape'], f'{target}: shape')
                        require(str(values.dtype) == item['dtype'], f'{target}: dtype')
                        require(values.shape[0] == samples, f'{target}: sample order length')
                        if target.stem == 'clusters':
                            k = item.get('K', reference_k)
                            require(values.min() >= 0 and values.max() < k, f'{target}: cluster range')

    for section, methods in [('benchmark', 6), ('clustgeo', 3), ('evaluation_curves', 9)]:
        table = tables[(section, 'seed_metrics.csv')]
        require(len(table) == 30 * 3 * methods * (max_k - 1), f'{dataset}/{section}: completeness')
        require(not table.duplicated(KEYS).any(), f'{dataset}/{section}: duplicates')
        require(table.method.nunique() == methods, f'{dataset}/{section}: methods')
        require(set(table.K) == set(range(2, max_k + 1)), f'{dataset}/{section}: K sweep')
        for _, part in table.groupby(['method', 'n_neighbors', 'K']):
            require(sorted(part.seed.tolist()) == list(range(30)), f'{dataset}/{section}: missing seeds')
        require(np.isfinite(table[['H_per_sample', 'WCSS_TSS', 'ARI']]).all().all(),
                f'{dataset}/{section}: nonfinite metrics')
    clustgeo = tables[('clustgeo', 'seed_metrics.csv')]
    require(set(clustgeo.method) == {
        f'PaCMAP feature-only + ClustGeo (alpha={alpha:g})' for alpha in (0, .5, 1)
    }, f'{dataset}: ClustGeo alphas')
    expected = pd.concat([tables[('benchmark', 'seed_metrics.csv')], clustgeo], ignore_index=True)
    actual = tables[('evaluation_curves', 'seed_metrics.csv')]
    common = expected.columns.intersection(actual.columns).tolist()
    pd.testing.assert_frame_equal(
        expected[common].sort_values(KEYS).reset_index(drop=True),
        actual[common].sort_values(KEYS).reset_index(drop=True),
        check_dtype=False, atol=1e-10, rtol=1e-10)
    if dataset == 'synthetic':
        from synthetic_data import make_synthetic_repeated_facies
        actual = tables[('data_walkthrough', 'data.csv')].drop(columns='sample_index')
        expected, _ = make_synthetic_repeated_facies(42)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)
    print(f'Validated {dataset}: metadata, tables, arrays, seeds, and sweep completeness.', flush=True)


def main():
    manifest_path = EXPORTS / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for item in manifest['files']:
            path = EXPORTS / item['path']
            require(path.stat().st_size == item['bytes'], f'{path}: snapshot size changed')
            require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
                    f'{path}: snapshot checksum changed')
        print(f"Verified snapshot checksums for {len(manifest['files'])} files.", flush=True)
    for args in [('cmio', 1551, 7, 10), ('north_batten', 3222, 15, 15), ('synthetic', 2304, 4, 10)]:
        validate_dataset(*args)
    # Execute the complete combined notebook, including partition/score agreement
    # and paired-test consistency checks. Its default SAVE_OUTPUTS=False is kept.
    import IPython.display
    import matplotlib.pyplot as plt
    from unittest.mock import patch

    path = ROOT / 'notebooks' / 'Experiment_Benchmarks' / '04_combined_paper_figures.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with patch.object(IPython.display, 'display', lambda *args, **kwargs: None):
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), f'{path}:cell-{i}', 'exec'), namespace)
        print('Combined notebook rebuilt successfully from exported data.', flush=True)
    finally:
        os.chdir(previous)
        plt.close('all')


if __name__ == '__main__':
    main()
