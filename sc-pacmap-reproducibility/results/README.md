# Saved results

`notebook_exports/` contains a validated numerical snapshot for CMiO, North
Batten, and Synthetic: **122 export files**, approximately **77.2 MiB**, plus
`manifest.json`. It includes all three main benchmarks, ClustGeo comparisons,
example-panel data, statistical comparisons, and available cross-group diagnostics.
The snapshot passed the validation command below and the complete combined paper
notebook rebuilt successfully from these files without embedding fits.

## Verify and reproduce figures

From the project root, with the Python dependencies installed:

```bash
python code/validate_result_exports.py
```

This verifies the manifest checksums, metadata, CSV schemas and counts, array
indexes/shapes/dtypes, finite values, complete seed and cluster-count sweeps,
and agreement between source scores and evaluation tables. It checks the
synthetic population against the shared seed-42 generator, then executes the
combined notebook, including saved-partition versus reported-score checks and
paired continuity checks. It does not modify exports or notebook files.

For visible figures and tables, run
[04_combined_paper_figures.ipynb](../notebooks/Experiment_Benchmarks/04_combined_paper_figures.ipynb).
It reads this snapshot; it does not refit embeddings or partitions.
`SAVE_OUTPUTS = False` displays figures and tables inline. Setting it to `True`
writes outputs to `combined_paper_figures/`.

## Snapshot settings and provenance

- Example-panel algorithm seed: **25** for all three datasets.
- Main comparison: **30 algorithm seeds (0–29)**, embedding neighbourhoods **10, 15, 20**.
- ClustGeo alpha values: **0, 0.5, 1**.
- Reference cluster counts: CMiO **7**, North Batten **15**, Synthetic **4**.
- Synthetic population: **2,304 samples**, **50 features**, data seed **42**.

Each section's `metadata.json` records settings, package versions, and artifact
schemas. `manifest.json` records file sizes and SHA-256 checksums, the source
commit present at validation, and source hashes. Notebook source hashes exclude
saved outputs and execution counts. The generating commit was not independently
recorded at export time; the validation commit identifies the reviewed source
state rather than proving run-time provenance.

The tracked North Batten notebook retains a historical `KeyError` output.
Its saved displays were not used to validate the snapshot. The combined notebook
was executed afresh from exported data and passed. Refresh the North Batten
notebook's displayed outputs separately before presenting it as a clean run.

## File layout

| Folder | Contents |
| --- | --- |
| `notebook_exports/<dataset>/<section>/` | Committed CSV tables, compressed NumPy arrays, JSON indexes and metadata |
| `xyz_scaling_test/<dataset>/` | Separate local coordinate-scaling exports; outside this validated snapshot |
| `embedding_cache/` | Intermediate PaCMAP embeddings used by ClustGeo; ignored by Git |
| `combined_paper_figures/` | Optional generated figures and tables |

Dataset IDs are `cmio`, `north_batten`, and `synthetic`. The two tracked
`*_algorithm_stability.csv` files are legacy outputs, separate from this snapshot.

Keep CSV tables with their metadata. NumPy stores need their matching
`*_index.json` files to identify method, seed, neighbourhood, and cluster-count
keys. Load `.npz` files with `allow_pickle=False`. Sample-wise arrays follow
`data_walkthrough/data.csv` row order. Evaluation-scale figures require the saved
coordinates and partitions, so a CSV-only copy cannot reproduce every figure.

Rerunning source notebooks replaces section exports. Preserve this snapshot
before experimenting; after intentional changes, validate the new files and
regenerate the manifest before committing a replacement. Caches and repeatedly
generated figure copies are not part of the snapshot.
