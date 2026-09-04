from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def _resolve_spasne_dir(spasne_dir=None):
    if spasne_dir is None:
        spasne_dir = os.environ.get("SPASNE_DIR")
    if spasne_dir is None:
        spasne_dir = Path(__file__).resolve().parent / "spasne_original"
    return Path(spasne_dir).expanduser().resolve()


def _load_spasne_module(spasne_dir=None):
    spasne_dir = _resolve_spasne_dir(spasne_dir)
    module_path = spasne_dir / "spasne.py"
    executable = spasne_dir / "spasne"
    if not module_path.is_file() or not executable.is_file():
        raise RuntimeError(
            f"SpaSNE is not built at {spasne_dir}. Run 'make' in that directory "
            "on Linux or WSL, or set SPASNE_DIR."
        )

    specification = importlib.util.spec_from_file_location("_sc_pacmap_spasne", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load SpaSNE wrapper from {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _windows_to_wsl_path(path):
    result = subprocess.run(
        ["wsl.exe", "--exec", "wslpath", "-a", str(Path(path).resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_spasne_wsl(features, coordinates, config, spasne_dir):
    spasne_dir = _resolve_spasne_dir(spasne_dir)
    bridge_path = Path(__file__).resolve().parent / "spasne_wsl_bridge.py"
    if not bridge_path.is_file() or not (spasne_dir / "spasne").is_file():
        raise RuntimeError(
            f"SpaSNE is not built at {spasne_dir}. Run 'make' there from WSL."
        )

    try:
        with tempfile.TemporaryDirectory(prefix="sc_pacmap_spasne_") as temporary_directory:
            temporary_path = Path(temporary_directory)
            feature_path = temporary_path / "features.npy"
            coordinate_path = temporary_path / "coordinates.npy"
            config_path = temporary_path / "config.json"
            output_path = temporary_path / "embedding.npy"
            np.save(feature_path, features, allow_pickle=False)
            np.save(coordinate_path, coordinates, allow_pickle=False)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            command = [
                "wsl.exe",
                "python3",
                _windows_to_wsl_path(bridge_path),
                _windows_to_wsl_path(spasne_dir),
                _windows_to_wsl_path(feature_path),
                _windows_to_wsl_path(coordinate_path),
                _windows_to_wsl_path(config_path),
                _windows_to_wsl_path(output_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"SpaSNE failed in WSL: {detail}")
            if not output_path.is_file():
                raise RuntimeError("SpaSNE WSL bridge completed without an output file")
            return np.load(output_path, allow_pickle=False)
    except FileNotFoundError as error:
        raise RuntimeError("WSL is required to run SpaSNE from a Windows kernel") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Unable to translate a SpaSNE path for WSL: {error.stderr}") from error


def python_spasne(
    X_feat,
    coordinates,
    random_state=42,
    alpha=10.0,
    beta=5.0,
    exaggeration=4,
    perplexity=50,
    theta=0.5,
    max_iter=1000,
    use_pca=True,
    spasne_dir=None,
):
    """Run the pinned upstream SpaSNE implementation, using WSL on Windows."""
    features = np.asarray(X_feat, dtype=np.float64)
    spatial_coordinates = np.asarray(coordinates, dtype=np.float64)
    if features.ndim != 2 or spatial_coordinates.ndim != 2 or len(features) != len(spatial_coordinates):
        raise ValueError("X_feat and coordinates must be 2-D arrays with equal row counts")
    if len(features) < 2 or features.shape[1] < 1 or spatial_coordinates.shape[1] < 2:
        raise ValueError("SpaSNE requires at least two samples, one feature, and two coordinates")
    if not np.isfinite(features).all() or not np.isfinite(spatial_coordinates).all():
        raise ValueError("SpaSNE inputs must contain only finite values")
    if perplexity <= 0 or perplexity >= len(features):
        raise ValueError("perplexity must be positive and smaller than the sample count")

    config = {
        "alpha": alpha,
        "beta": beta,
        "exaggeration": exaggeration,
        "no_dims": 2,
        "perplexity": perplexity,
        "theta": theta,
        "randseed": int(random_state),
        "verbose": False,
        "initial_dims": min(features.shape[1], 200),
        "use_pca": use_pca,
        "max_iter": max_iter,
    }
    if os.name == "nt":
        embedding = _run_spasne_wsl(features, spatial_coordinates, config, spasne_dir)
    else:
        spasne = _load_spasne_module(spasne_dir)
        embedding = spasne.run_spasne(features, pixels=spatial_coordinates, **config)

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.shape != (len(features), 2) or not np.isfinite(embedding).all():
        raise RuntimeError(
            f"SpaSNE returned invalid embedding shape {embedding.shape}; "
            f"expected ({len(features)}, 2) with finite values"
        )
    return embedding
