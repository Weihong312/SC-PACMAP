# SC-PaCMAP reproducibility

Minimal code, data preparation, and notebooks for three SC-PaCMAP benchmark experiments:

1. Critical Minerals in Ores (CMiO)
2. North Batten drill-core geochemistry
3. Synthetic spatial-conflict data

## Repository layout

- `code/sc_pacmap/sc_pacmap.py`: neighbour-replacement-graph and full-graph SC-PaCMAP implementations.
- `code/sc_pacmap/pacmap_original/`: focused PaCMAP 0.9.1 reference source, license, and provenance.
- `code/spasne/spasne_original/`: unmodified upstream SpaSNE source, license, and provenance.
- `code/spasne/spasne_adapter.py` and `spasne_wsl_bridge.py`: project-authored SpaSNE integration.
- `data_preparation/prepare_datasets.ipynb`: creates the two real-data analysis files.
- `data/processed/`: prepared real datasets used directly by the benchmark notebooks.
- `notebooks/`: the three benchmark experiments.

## Setup

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

SpaSNE must be compiled once on Linux or WSL:

```bash
cd code/spasne/spasne_original
make
```

On Windows, run that command inside WSL. The notebooks automatically use the project adapter and WSL bridge. Set `NUMBA_NUM_THREADS=1` before importing Numba or PaCMAP; the notebooks already do this.

## Run the experiments

Open JupyterLab or VS Code at the repository root and run these notebooks from top to bottom:

- `notebooks/01_cmio_sc_pacmap_benchmark.ipynb`
- `notebooks/02_north_batten_sc_pacmap_benchmark.ipynb`
- `notebooks/03_synthetic_sc_pacmap_benchmark.ipynb`

The additional notebook groups are:

- `notebooks/New_Notebooks/`: results, neighbourhood robustness, and cross-group checks.
- `notebooks/sensitivity test/`: overlap factor and ramp duration, each using 30 paired seeds.
- `notebooks/Geary Test/`: PCA-Geary permutation diagnostics using positive continuity.
- `notebooks/lambda test/lambda_sweep_study.ipynb`: ten fixed lambda values from 0 to 1 across 30 paired algorithm seeds (0-29) per dataset (900 fits total), with fixed data and K-means seed 42. Curves show seed means with pointwise 95% Student-t confidence bands; dashed lines mark mean inferred automatic lambda. A notebook-local override supplies each fixed lambda to the normal target-share calculation and is restored after each fit; no additional cap is applied.

SC-PaCMAP uses positive continuity `s_q = max(0, 1 - C_q)` and
`lambda = sum(pi_q * s_q)` in both graph variants. The Phase-3 target share is
the analytical ratio `lambda * A_S / (lambda * A_S + (1 - lambda) * A_F)`,
with the existing degenerate-attraction guards. These quantities are naturally
bounded; use `SCPaCMAPConfig()` without `max_target_share`. The linear ramp
ends explicitly at one. Restart the notebook kernel after updating the module.
Saved notebook outputs were cleared because they described the previous method.

Run the focused regression checks from this repository root with:

```bash
python -m unittest discover -s tests -v
```

The real-data notebooks use the committed files in `data/processed`. To regenerate them, place the source files described in `data/raw/README.md`, then run `data_preparation/prepare_datasets.ipynb`.

## Source boundaries

SC-PaCMAP project code is kept separate from copied reference implementations. Each third-party source directory contains its upstream license and a `SOURCE.txt` describing the exact release or commit and any file-level provenance. Runtime PaCMAP and LocalMAP are supplied by the pinned `pacmap==0.9.1` package.
