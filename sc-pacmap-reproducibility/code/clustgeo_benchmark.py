"""Notebook bridge to original R ClustGeo.

Clustering itself is in clustgeo/run_clustgeo.R.
One-time R setup: python code/clustgeo_benchmark.py --setup
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R_CODE = ROOT / 'code' / 'clustgeo'
R_LIBRARY = Path(os.environ.get('CLUSTGEO_R_LIBRARY', ROOT / '.r-env' / 'clustgeo'))
# Use the installed study runtime on Windows, or Rscript on PATH elsewhere.
RSCRIPT = os.environ.get('CLUSTGEO_RSCRIPT') or shutil.which('Rscript')
if not RSCRIPT:
    RSCRIPT = 'C:/Program Files/R/R-4.3.1/bin/Rscript.exe'
ALPHAS = (0., .25, .50, .75)


def run_r(script, *args):
    """Run an R script using the isolated project library."""
    if not (Path(RSCRIPT).is_file() or shutil.which(RSCRIPT)):
        raise RuntimeError('Set CLUSTGEO_RSCRIPT to your Rscript executable.')
    env = os.environ.copy()
    env.update(R_LIBS_USER=str(R_LIBRARY), R_LIBS_SITE=str(R_LIBRARY),
               RENV_PATHS_ROOT=str(ROOT / '.r-env' / 'renv'),
               RENV_CONFIG_CACHE_ENABLED='FALSE', LANGUAGE='en', LC_ALL='C')
    command = [RSCRIPT, '--vanilla', str(R_CODE / script), str(R_LIBRARY)]
    command.extend(str(arg) for arg in args)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


def clustgeo_environment():
    """Check the R package and return its provenance."""
    output = run_r('run_clustgeo.R', '--check')
    metadata = {}
    for line in output.splitlines():
        key, value = line.split('=', 1)
        metadata[key] = value
    return metadata


def clustgeo_partitions(embedding, coordinates, k, alphas=ALPHAS):
    """Write inputs, call original R ClustGeo, then return labels per alpha."""
    alphas = tuple(alphas)
    if not isinstance(k, (int, np.integer)) or not 2 <= k <= len(embedding):
        raise ValueError('K must be an integer from 2 to the sample count.')
    if not alphas or not all(np.isfinite(a) and 0 <= a <= 1 for a in alphas):
        raise ValueError('Provide finite alpha values in [0, 1].')
    # R validates the matrices and constructs distances from these raw inputs.
    with tempfile.TemporaryDirectory(prefix='clustgeo_') as folder:
        work = Path(folder)
        np.savetxt(work / 'embedding.csv', embedding, delimiter=',', fmt='%.17g')
        np.savetxt(work / 'coordinates.csv', coordinates, delimiter=',', fmt='%.17g')
        run_r('run_clustgeo.R', work, k, *alphas)
        labels = np.loadtxt(work / 'partitions.csv', delimiter=',', dtype=int, ndmin=2)
    if labels.shape != (len(embedding), len(alphas)):
        raise RuntimeError('R returned an unexpected partition matrix.')
    for column, alpha in enumerate(alphas):
        partition = labels[:, column]
        if not np.array_equal(np.unique(partition), np.arange(k)):
            raise RuntimeError('R did not return exactly K zero-based clusters.')
        yield alpha, partition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', action='store_true')
    if parser.parse_args().setup:
        print(run_r('setup.R', R_CODE / 'renv.lock'))
    print(json.dumps(clustgeo_environment(), indent=2))
