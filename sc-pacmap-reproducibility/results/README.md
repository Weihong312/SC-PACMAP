# Saved results

The experiment notebooks export numerical data so researchers can inspect scores
and regenerate paper figures without repeating all embedding fits. The export
folders below are currently local generated files, not a committed final results
snapshot. The two tracked `*_algorithm_stability.csv` files are legacy outputs,
not a substitute for the section exports.

## Output folders

| Folder | Contents | Producer |
| --- | --- | --- |
| `notebook_exports/<dataset>/<section>/` | CSV tables, compressed NumPy arrays, JSON indexes and metadata | Three `Experiment_Benchmarks/*_results.ipynb` notebooks |
| `xyz_scaling_test/<dataset>/` | Coordinate-scaling comparisons, paired differences, arrays and settings | Three `XYZ_scaling_test` notebooks |
| `embedding_cache/` | Intermediate feature-only PaCMAP embeddings for ClustGeo | Main benchmark loops; ignored by Git |
| `combined_paper_figures/` | Optional figures and tables | Combined notebook with `SAVE_OUTPUTS = True` |

Dataset IDs are `cmio`, `north_batten`, and `synthetic`. Rerunning a section
replaces its named artifacts. Export metadata records settings and package
versions; it does not currently record a source Git commit or file checksums.

## Files needed for review

Keep CSV tables together with their `metadata.json` files. NumPy stores need
their matching `*_index.json` files to identify method, seed, neighbourhood,
and cluster-count keys. Load `.npz` files with `allow_pickle=False`.
Sample-wise arrays follow the row order in `data_walkthrough/data.csv`.

The combined paper notebook reads tables for the performance plane, evaluation
curves, and cross-group comparisons. Its evaluation-scale diagnostics also need
`data_walkthrough/arrays.npz`, `benchmark/clusters.npz`, `clustgeo/clusters.npz`,
and the cluster-store indexes. A CSV-only snapshot cannot reproduce every
combined figure.

## Results versioning

A validated, frozen copy of the numerical exports belongs with a reproducibility
release. Keep tables, metadata, and arrays needed to recreate its figures together.
Intermediate caches and repeatedly generated figure copies can be regenerated
and should stay outside that snapshot.

The current local exports total approximately **78.6 MiB** across **143 files**:
77.2 MiB of main notebook exports and 1.4 MiB of coordinate-scaling exports.
Before committing these as paper results:

1. Regenerate the example-panel sections using the current seed **25**. On
   inspection, `north_batten/one_seed/metadata.json` records seed **0** and
   `synthetic/one_seed/metadata.json` records seed **10**. CMiO records seed 25.
   These values differ from the current notebook configuration; do not relabel
   old arrays as new runs.
2. Check settings across sections. Current ClustGeo alpha values are **0, 0.5,
   and 1**; the main comparison uses seeds **0–29** and embedding neighbourhoods
   **10, 15, and 20**. Historical `reference_seed` fields in other sections can
   also retain older example-panel settings.
3. Rebuild combined figures and tables from the snapshot and verify agreement
   with the intended paper results.
4. Record the generating source commit and file checksums in a snapshot manifest.
   Retain package versions, data settings, and array indexes.

The current export folders have been left unchanged and untracked while these
differences are unresolved. No full experiment rerun was performed as part of
the repository reorganization or documentation update.
