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
- Robustness diagnostics reuse the principal k=10 reference-K partitions saved
  by the benchmark loop. Cross-deposit (CMiO) and cross-hole (North Batten)
  diagnostics use all 30 seeds. All three notebooks add a supplementary spatial
  contiguity curve at evaluation neighbourhoods 5, 10, 20, 40, keeping the
  embedding and partition fixed and excluding the focal sample. These cells
  do not refit embeddings. Missing historical partitions cannot be reconstructed
  from score tables. Outputs go to `results/robustness/`.
- The three results notebooks include the R13 **feature-only PaCMAP → ClustGeo**
  partition comparator at embedding k=10, 15, 20, seeds 0–29, and fixed alpha values
  0, 0.25, 0.50, 0.75. Reference K is 4 (Synthetic), 7 (CMiO), or 15 (North Batten).
  This gives 360 partitions per dataset (3 neighbourhoods x 30 seeds x 4 alphas).
  All alpha values at every k enter the main tables, paired tests, and trade-off plot;
  none is selected using labels. `code/clustgeo_benchmark.py` is an I/O bridge to
  the **original CRAN ClustGeo 2.1 R package**, not a Python reimplementation.
  `code/clustgeo/run_clustgeo.R` calls `ClustGeo::hclustgeo` and `stats::cutree`,
  with separate maximum-distance scaling of embedding and raw spatial distances.
  See [Chavent et al. (2018)](https://doi.org/10.1007/s00180-018-0791-1).
  The existing benchmark loop saves its k=10, 15, 20 feature-only embeddings under
  `results/embedding_cache/<dataset>/`. The ClustGeo cell only loads them and
  checks that features and coordinate row order match. It never fits PaCMAP.
  Run the benchmark cell first if embeddings are missing, and rerun it whenever
  embedding settings change. Once the existing evaluation helpers are loaded, rerun only the R13 cell.
  Existing score CSVs cannot reconstruct embeddings. The R13 cell saves per-seed
  scores and partitions in `results/clustgeo/`; run it after the benchmark to add
  the comparator to the main outputs. Spatial evaluation uses 10 non-self neighbours
  and `log(min(K, 10))` for every method. Saved main metric displays were cleared
  because they predate this comparator and explicit self-exclusion at tied coordinates.
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

For R14, the 30 seeds quantify optimizer/restart variability on fixed datasets,
not sampling or population variability. Cross-group neighbour exclusion and
evaluation-scale curves are full-data diagnostics, not held-out validation.
Spatially blocked resampling of the complete preprocessing/fitting pipeline is
required for population-level inference and is reserved for future work; this
study does not claim to have performed it. See Roberts et al. (2017),
*Cross-validation strategies for data with temporal, spatial, hierarchical, or
phylogenetic structure*, Ecography, 40, 913-929,
[doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).

The `*_available_pacmap_diagnostics.csv` files contain the completed 30-seed
diagnostics for feature-only PaCMAP only, using the embeddings currently saved
on disk. They are partial results, not a completed comparison of all methods.

## ClustGeo R environment

For code review, start with `code/clustgeo/run_clustgeo.R`, which calls the
original package. One Python helper, `code/clustgeo_benchmark.py`, handles the
R call. The notebook benchmark loop saves embeddings for its ClustGeo cell. `setup.R` and `renv.lock` restore
the isolated R environment.

Install R, then run from the repository root using the notebook's Python environment:

```bash
python code/clustgeo_benchmark.py --setup
python code/clustgeo_benchmark.py
```

Setup installs/restores the original ClustGeo package and its dependencies into
`.r-env/clustgeo/`, separate from personal R libraries. The checked-in
`code/clustgeo/renv.lock` pins package versions; renv state also stays under
`.r-env/`. Setup needs network access; benchmark runs do not. No RStudio or
Python `rpy2` installation is needed. On this Windows machine the bridge uses
native R; it does not need WSL. On Linux/macOS it uses `Rscript` on PATH.
The recorded and tested R runtime is **4.3.1**; use that version to reproduce
this environment. Restoring compiled dependencies from source on another
platform may require that platform's R build tools and spatial libraries.

The bridge uses `Rscript` on PATH, with R 4.3.1 under `Program Files/R` as the Windows default.
Set `CLUSTGEO_RSCRIPT` to choose a specific executable, and optionally
`CLUSTGEO_R_LIBRARY` to choose another isolated library (then run setup).
It fails explicitly if R/ClustGeo is missing; there is no Python fallback.
The R13 cells preserve PaCMAP caches and record R version, ClustGeo version,
and backend provenance alongside the per-seed outputs. Restart an existing
notebook kernel after replacing the earlier Python implementation.
