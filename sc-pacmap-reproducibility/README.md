# SC-PaCMAP reproducibility

Code, prepared data, and notebooks for reviewing and reproducing SC-PaCMAP
experiments on CMiO mineral geochemistry, North Batten drill-core geochemistry,
and synthetic repeated-facies data.

## Start here

1. Review [the method implementation](code/sc_pacmap/sc_pacmap.py).
2. Read the three [experiment notebooks](notebooks/Experiment_Benchmarks/) for preprocessing, embeddings, clustering, metrics, and statistical comparisons.
3. Inspect [the synthetic-data notebook](notebooks/synthetic_data.ipynb) independently of the experiments.
4. Read [the results guide](results/README.md) before using saved exports. The local export folders are not yet committed as a final results snapshot.

## Repository layout

| Location | Purpose |
| --- | --- |
| `code/sc_pacmap/sc_pacmap.py` | Neighbour-replacement and full-graph SC-PaCMAP |
| `code/sc_pacmap/pacmap_original/` | PaCMAP reference source, license, and provenance |
| `code/spasne/` | Original SpaSNE plus the project adapter and WSL bridge |
| `code/clustgeo_benchmark.py`, `code/clustgeo/` | Python/R bridge to original ClustGeo and pinned R environment |
| `code/synthetic_data.py` | Imports shared synthetic-data definitions from the notebook |
| `code/notebook_result_export.py` | Exports section tables, arrays, indexes, and metadata |
| `data_preparation/prepare_datasets.ipynb` | Regenerates the prepared real datasets |
| `data/processed/` | Committed real datasets |
| `notebooks/synthetic_data.ipynb` | Synthetic generator, explanation, and inspection |
| `notebooks/Experiment_Benchmarks/` | Three main experiments and combined paper figures |
| `results/` | Numerical outputs; see the results guide for their status |

## Environment setup

The project root below is the directory containing this README, `requirements.txt`,
`code/`, and `notebooks/`. From the enclosing repository, first run
`cd sc-pacmap-reproducibility`.

Use Python 3.11 and the pinned dependencies:

```bash
python -m venv .venv
```

Activate in Windows PowerShell with `.venv/Scripts/Activate.ps1`, or on Linux/macOS
with `source .venv/bin/activate`, then install:

```bash
python -m pip install -r requirements.txt
```

The notebooks configure Numba before numerical imports. Restart an existing kernel
after changing the environment or implementation.

### SpaSNE

Compile once on Linux or inside WSL:

```bash
cd code/spasne/spasne_original
make
```

Return to the project root afterwards. Windows notebooks use the project adapter
and WSL bridge. The standalone synthetic-data notebook needs only NumPy, pandas,
and a notebook environment; it does not need SpaSNE or R.

### ClustGeo

The comparator calls the original **CRAN ClustGeo 2.1 R package**. The recorded
R runtime is **4.3.1**. Install R, then run from the project root using the notebook's
Python environment:

```bash
python code/clustgeo_benchmark.py --setup
python code/clustgeo_benchmark.py
```

Setup restores `code/clustgeo/renv.lock` into `.r-env/clustgeo/` and needs network
access. Benchmark calls do not. Windows uses R 4.3.1 under `Program Files/R` by
default; other platforms use `Rscript` on PATH. Set `CLUSTGEO_RSCRIPT` to choose
an executable and `CLUSTGEO_R_LIBRARY` to choose another isolated library.
Missing R or ClustGeo causes an explicit error, with no Python fallback.
Review `code/clustgeo/run_clustgeo.R` for the original package call.

## Run the main experiments

Start JupyterLab or VS Code at the project root. Use a fresh kernel for each
dataset notebook and run its cells from top to bottom.

| Notebook | Reference cluster count | Cluster-count sweep |
| --- | --- | --- |
| [01 CMiO](notebooks/Experiment_Benchmarks/01_cmio_sc_pacmap_results.ipynb) | 7 | 2-10 |
| [02 North Batten](notebooks/Experiment_Benchmarks/02_north_batten_sc_pacmap_results.ipynb) | 15 | 2-15 |
| [03 Synthetic](notebooks/Experiment_Benchmarks/03_synthetic_sc_pacmap_results.ipynb) | 4 | 2-10 |

Example embedding panels use algorithm seed **25**. The main comparison uses
**30 algorithm seeds (0-29)** and embedding neighbourhoods **10, 15, and 20**.
The replacement variant is disabled by default with `INCLUDE_REPLACEMENT = False`.
Labels enter evaluation and reference cluster-count selection, not embedding
features or K-means fitting.

ClustGeo clusters each saved feature-only PaCMAP embedding with fixed alpha values
**0, 0.5, and 1**, across the cluster-count sweep. All three alpha values enter
the comparison; labels do not select an alpha. Run the embedding benchmark first.
Changing embedding settings requires regenerating embeddings; score CSVs cannot
reconstruct them.

Spatial evaluation uses 10 non-self neighbours and `log(min(K, 10))` normalization.
Multivariate delineation is `100 * (1 - WCSS/TSS)`. SC-PaCMAP uses positive Geary
continuity `s_q = max(0, 1 - C_q)`, `lambda = sum(pi_q * s_q)`, and the analytical
Phase-3 target share `lambda * A_S / (lambda * A_S + (1 - lambda) * A_F)`, with
degenerate-attraction guards. Both variants use `SCPaCMAPConfig()` without a
configurable target-share cap.

Each result section exports unrounded tables and relevant arrays beneath
`results/notebook_exports/<dataset>/<section>/`. Figures remain inline. Rerunning
an export replaces its named files; retain a separate copy for a frozen snapshot.

After all three datasets have been exported, run
[04 combined paper figures](notebooks/Experiment_Benchmarks/04_combined_paper_figures.ipynb).
It reads exports to build the performance plane, cross-group continuity plots,
evaluation-scale diagnostics, and principal-neighbourhood benchmark table.
It does not refit embeddings or cluster assignments. Evaluation-scale diagnostics
recompute neighbours using saved coordinates and partitions, so they need arrays
as well as CSVs. `SAVE_OUTPUTS = False` displays outputs inline; set it to `True`
to export figures and tables to `results/combined_paper_figures/`.

## Data preparation

Real-data experiments read committed files in `data/processed/`. To regenerate
them, obtain sources described in [data/raw/README.md](data/raw/README.md) and run
[prepare_datasets.ipynb](data_preparation/prepare_datasets.ipynb).

The [synthetic-data notebook](notebooks/synthetic_data.ipynb) defines one fixed
seed-42 population: **2,304 samples**, a jittered **48 x 48** grid, **16 patches**,
**four facies**, and **50 features**. Feature blocks contain 10 noisy facies
signals, 20 nonlinear confounders, and 20 independent noise variables.

All five synthetic experiment workflows import this generator through
`code/synthetic_data.py`. The loader executes only notebook cells tagged
`synthetic-data-definitions`, excluding inspection cells. No prior notebook
execution or saved synthetic CSV is required. Preprocessing stays in each
experiment notebook.

## Additional experiments

| Notebook group | Purpose |
| --- | --- |
| [XYZ_scaling_test](notebooks/XYZ_scaling_test/README.md) | Raw versus standardized coordinates for UMAP, PaCMAP, and LocalMAP; seed 42 by default; Synthetic uses XY |
| [sensitivity test](notebooks/sensitivity%20test/) | Overlap-factor and ramp-duration sweeps with 30 paired seeds |
| [Combined sensitivity figure](notebooks/sensitivity%20test/04_combined_cmio_north_batten_sensitivity.ipynb) | CMiO and North Batten figure/table; 540 distinct fits, reusing shared baselines |
| [Geary Test](notebooks/Geary%20Test/) | PCA-Geary permutation diagnostics |
| [lambda test](notebooks/lambda%20test/lambda_sweep_study.ipynb) | Ten fixed lambda values over 30 paired seeds per dataset; 900 fits total |
| [Run_Time_test](notebooks/Run_Time_test/01_one_run_runtime.ipynb) | One-run runtime experiment |

The main real-data notebooks also include cross-deposit/cross-hole continuity
checks and evaluation-scale diagnostics using fixed embeddings and partitions.
The lambda experiment reports pointwise 95% Student-t confidence intervals over
algorithm seeds and restores its temporary lambda override after each fit.

## Interpretation and source provenance

The 30 seeds describe optimizer/restart variability on fixed datasets. They do
not measure population-sampling uncertainty. Cross-group exclusions and
evaluation-scale curves are full-data diagnostics, not held-out validation.
The synthetic experiment is a designed mechanism stress test.

Project code is separate from copied reference implementations. Each third-party
source directory includes its upstream license and `SOURCE.txt` provenance.
Runtime PaCMAP and LocalMAP come from pinned `pacmap==0.9.1`.
