# Coordinate scaling comparison

Based on the three results notebooks in `../Experiment_Benchmarks/`: CMiO, North Batten, and Synthetic. Each compares raw versus per-axis standardized coordinates appended to identical standardized features for UMAP, PaCMAP, and LocalMAP. Synthetic uses XY; the real datasets use XYZ.

Defaults: full populations, seed 42, embedding k=10, reference K=7/15/4, fixed K-means seed and 20 initializations. Tables show spatial contiguity, multivariate delineation, ARI, and paired standardized-minus-raw changes. Positive changes favour standardization. One seed is descriptive only.

Use Run All. To expand later, change `SEEDS = (42,)` to `SEEDS = tuple(range(30))` in each setup cell, then Run All. Undefined single-seed SD is intentional. Exports go to `results/xyz_scaling_test/<dataset>/`, separate from benchmark exports. Reruns replace these exports. Figures remain in the notebooks.
