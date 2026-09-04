from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def load_spasne(spasne_dir):
    module_path = spasne_dir / 'spasne.py'
    spec = importlib.util.spec_from_file_location('_geopacmap_spasne_wsl', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load SpaSNE wrapper from {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    if len(args) != 6:
        raise ValueError(
            'Expected SpaSNE directory, feature, coordinate, config, and output paths'
        )

    spasne_dir, feature_path, coord_path, config_path, output_path = map(Path, args[1:])
    features = np.load(feature_path, allow_pickle=False)
    coords = np.load(coord_path, allow_pickle=False)
    config = json.loads(config_path.read_text(encoding='utf-8'))

    spasne = load_spasne(spasne_dir)
    embedding = np.asarray(
        spasne.run_spasne(features, pixels=coords, **config),
        dtype=np.float64,
    )
    np.save(output_path, embedding, allow_pickle=False)


if __name__ == '__main__':
    main(sys.argv)
