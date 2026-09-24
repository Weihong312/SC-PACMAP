"""Three PaCMAP modes using an untouched PaCMAP 0.9.1 core.

Inputs
------
features_standardized : n x p standardized feature matrix
coordinates_m        : n x 2 or n x 3 coordinates in metres

Modes
-----
1. ``pacmap_features``
   Original PaCMAP. Input = features only.

2. ``pacmap_features_xyz``
   Original PaCMAP again. Input = features plus separately standardized XYZ.
   The PaCMAP code is identical to mode 1; only the input matrix changes.

3. ``sc_pacmap``
    Keep the complete original feature-neighbour graph ``G_F`` and construct a
    separate PCA-Geary spatial graph ``G_S`` from geographic candidates. Add one
    weighted SC attraction only in Phase 3. Its target attraction share is
    calculated once from Geary lambda and the unresolved NN/SC attraction gap,
    then ramped in while the same share is transferred out of NN attraction.

The official copied functions are in ``pacmap_original/pacmap_core_reference.py``.
Their signatures and function bodies are not edited. Search this wrapper for
``SC INJECTION`` to see every added algorithm block.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
from typing import Any

import numba
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pacmap_original import pacmap_core_reference as original


MODE_PACMAP_FEATURES = "pacmap_features"
MODE_PACMAP_FEATURES_XYZ = "pacmap_features_xyz"
MODE_SC_PACMAP = "sc_pacmap"


@dataclass(frozen=True)
class PaCMAPSettings:
    """Settings shared by copied PaCMAP, library PaCMAP and SC-PaCMAP."""

    n_components: int = 2
    n_neighbors: int | None = 15
    mn_ratio: float = 0.5
    fp_ratio: float = 2.0
    learning_rate: float = 1.0
    num_iters: tuple[int, int, int] = (100, 100, 250)
    random_state: int = 42
    knn_backend: str = "faiss"
    verbose: bool = False


@dataclass(frozen=True)
class SCPaCMAPConfig:
    """Settings for the separate Phase-3 spatial-continuity injection."""

    c_sc: float = 10.0
    phase3_ramp_iterations: int = 25
    overlap_sc_factor: float = 0.5


@dataclass
class PaCMAPResult:
    """Embedding and pair arrays returned by one run."""

    embedding: np.ndarray
    pair_neighbors: np.ndarray
    pair_mn: np.ndarray
    pair_fp: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class SCDiagnostics:
    """Graph, weighting, and Phase-3 diagnostics for the SC injection."""

    geary_c: np.ndarray
    continuity_weight: np.ndarray
    spatial_strength_lambda: float
    k_spatial: int
    k_attribute: int
    ordinary_neighbors: np.ndarray
    selected_spatial_neighbors: np.ndarray
    final_neighbors: np.ndarray
    pair_sc: np.ndarray
    sc_distance: np.ndarray
    sc_edge_weight: np.ndarray
    overlap_mask: np.ndarray
    automatic_target_share: float
    overlap_fraction: float
    sc_distance_scale: float
    mean_sc_weight_before_overlap: float
    mean_sc_weight_after_overlap: float
    mean_realized_sc_share_active: float
    mean_w_sc_active: float
    max_w_sc_active: float
    c_sc: float
    phase3_ramp_iterations: int


@dataclass
class LegacySCDiagnostics:
    """Diagnostics for the superseded neighbour-replacement benchmark only."""

    geary_c: np.ndarray
    continuity_weight: np.ndarray
    spatial_strength_lambda: float
    k_spatial: int
    k_attribute: int
    ordinary_neighbors: np.ndarray
    selected_spatial_neighbors: np.ndarray
    final_neighbors: np.ndarray


@dataclass
class _SCGraph:
    """Internal data for the separate spatial-continuity graph."""

    ordinary_neighbors: np.ndarray
    selected_spatial_neighbors: np.ndarray
    pair_sc: np.ndarray
    sc_distance: np.ndarray
    overlap_mask: np.ndarray
    geary_c: np.ndarray
    continuity_weight: np.ndarray
    spatial_strength_lambda: float


def _as_features(x: np.ndarray) -> np.ndarray:
    """Validate a feature matrix without silently filling missing values."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("features must be a two-dimensional non-empty matrix")
    if not np.isfinite(x).all():
        raise ValueError("features contain missing or non-finite values")
    return np.ascontiguousarray(x)


def _as_coordinates(xyz: np.ndarray, n_rows: int) -> np.ndarray:
    """Validate X/Y or X/Y/Z coordinates."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[0] != n_rows:
        raise ValueError("coordinates must have one row per feature row")
    if xyz.shape[1] not in (2, 3):
        raise ValueError("coordinates must contain X/Y or X/Y/Z")
    if not np.isfinite(xyz).all():
        raise ValueError("coordinates contain missing or non-finite values")
    return np.ascontiguousarray(xyz)


def _decide_pair_counts(
    n_samples: int,
    n_neighbors: int | None,
    mn_ratio: float,
    fp_ratio: float,
) -> tuple[int, int, int]:
    """Match PaCMAP 0.9.1 ``PaCMAP.decide_num_pairs``."""
    if n_neighbors is None:
        if n_samples <= 10000:
            n_neighbors = 10
        else:
            n_neighbors = int(
                round(10 + 15 * (np.log10(n_samples) - 4))
            )

    n_mn = int(round(n_neighbors * mn_ratio))
    n_fp = int(round(n_neighbors * fp_ratio))
    n_neighbors = min(n_neighbors, n_samples - 1)
    n_fp = min(n_fp, n_samples - 1 - n_neighbors)
    n_mn = min(n_mn, n_samples - 1)

    if n_neighbors + n_mn + n_fp >= n_samples:
        n_neighbors = int(n_samples / (1 + mn_ratio + fp_ratio))
        n_mn = int(
            n_samples / (1 + mn_ratio + fp_ratio) * mn_ratio
        )
        n_fp = int(
            n_samples / (1 + mn_ratio + fp_ratio) * fp_ratio
        )

    if n_neighbors < 1:
        raise ValueError("The number of nearest neighbors cannot be below 1")
    if n_fp < 1:
        raise ValueError("The number of further points cannot be below 1")
    return n_neighbors, n_mn, n_fp


def _preprocess_with_original(
    x: np.ndarray,
    settings: PaCMAPSettings,
) -> tuple[np.ndarray, bool, Any]:
    """Call the copied original ``preprocess_X`` function unchanged."""
    x = np.copy(x).astype(np.float32)
    x_pre, pca_solution, transformer, _, _, _ = original.preprocess_X(
        x,
        "euclidean",
        True,
        settings.verbose,
        settings.random_state,
        x.shape[1],
        settings.n_components,
    )
    return np.ascontiguousarray(x_pre, dtype=np.float32), pca_solution, transformer


def _optimize_with_original(
    x_preprocessed: np.ndarray,
    pca_solution: bool,
    transformer: Any,
    pair_neighbors: np.ndarray,
    pair_mn: np.ndarray,
    pair_fp: np.ndarray,
    settings: PaCMAPSettings,
) -> np.ndarray:
    """Call the copied original ``pacmap`` optimizer unchanged."""
    y, _, _, _, _ = original.pacmap(
        x_preprocessed,
        settings.n_components,
        np.ascontiguousarray(pair_neighbors, dtype=np.int32),
        np.ascontiguousarray(pair_mn, dtype=np.int32),
        np.ascontiguousarray(pair_fp, dtype=np.int32),
        settings.learning_rate,
        settings.num_iters,
        "pca",
        settings.verbose,
        False,
        [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450],
        pca_solution,
        transformer,
    )
    return np.ascontiguousarray(y, dtype=np.float32)


def run_copied_original_pacmap(
    x_input: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
) -> PaCMAPResult:
    """Run the copied PaCMAP 0.9.1 path with no SC modification."""
    x_input = _as_features(x_input)
    n_neighbors, n_mn, n_fp = _decide_pair_counts(
        len(x_input),
        settings.n_neighbors,
        settings.mn_ratio,
        settings.fp_ratio,
    )

    # The official class writes the seed to this module-level variable.
    original._RANDOM_STATE = settings.random_state

    x_pre, pca_solution, transformer = _preprocess_with_original(
        x_input, settings
    )
    pair_neighbors, pair_mn, pair_fp, _ = original.generate_pair(
        x_pre,
        n_neighbors,
        n_mn,
        n_fp,
        "euclidean",
        settings.verbose,
        settings.knn_backend,
    )
    embedding = _optimize_with_original(
        x_pre,
        pca_solution,
        transformer,
        pair_neighbors,
        pair_mn,
        pair_fp,
        settings,
    )
    return PaCMAPResult(
        embedding=embedding,
        pair_neighbors=pair_neighbors,
        pair_mn=pair_mn,
        pair_fp=pair_fp,
        diagnostics={
            "mode": "copied_original_pacmap",
            "n_neighbors": n_neighbors,
            "n_mn": n_mn,
            "n_fp": n_fp,
        },
    )


def _geary_c_by_component(
    standardized_scores: np.ndarray,
    geographic_neighbors: np.ndarray,
) -> np.ndarray:
    """Directed-kNN Geary's C for each standardized PCA component."""
    n = standardized_scores.shape[0]
    w0 = float(n * geographic_neighbors.shape[1])
    centered = standardized_scores - standardized_scores.mean(
        axis=0, keepdims=True
    )
    denominator = np.sum(centered * centered, axis=0)
    differences = (
        standardized_scores[:, None, :]
        - standardized_scores[geographic_neighbors, :]
    )
    numerator = np.sum(differences * differences, axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        geary_c = (n - 1.0) * numerator / (2.0 * w0 * denominator)
    return np.where(np.isfinite(geary_c), geary_c, 1.0)


@numba.njit(cache=True, fastmath=True)
def _continuity_distance(
    standardized_scores: np.ndarray,
    geographic_neighbors: np.ndarray,
    component_weights: np.ndarray,
) -> np.ndarray:
    """Weighted PCA distance for geographic candidate neighbours."""
    n, k = geographic_neighbors.shape
    q = standardized_scores.shape[1]
    output = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for position in range(k):
            j = geographic_neighbors[i, position]
            value = 0.0
            for component in range(q):
                difference = (
                    standardized_scores[i, component]
                    - standardized_scores[j, component]
                )
                value += (
                    component_weights[component]
                    * difference
                    * difference
                )
            output[i, position] = value
    return output


def _neighbors_to_pairs(neighbors: np.ndarray) -> np.ndarray:
    """Convert an n-by-k neighbour matrix to PaCMAP's pair format."""
    n, k = neighbors.shape
    pairs = np.empty((n * k, 2), dtype=np.int32)
    pairs[:, 0] = np.repeat(np.arange(n, dtype=np.int32), k)
    pairs[:, 1] = neighbors.reshape(-1)
    return np.ascontiguousarray(pairs)


# ---------------------------------------------------------------------------
# NEIGHBOUR-REPLACEMENT-GRAPH BENCHMARK ONLY
# ---------------------------------------------------------------------------


def _attribute_neighbor_ranking(
    x_preprocessed: np.ndarray,
    n_neighbors: int,
    settings: PaCMAPSettings,
) -> np.ndarray:
    """Reproduce the ordinary locally scaled PaCMAP neighbour ranking."""
    n_search = min(n_neighbors + 50, len(x_preprocessed) - 1)
    _, neighbors, distances = original.compute_nearest_neighbors(
        x_preprocessed,
        n_search,
        "euclidean",
        settings.knn_backend,
        settings.random_state,
    )
    sigma = np.maximum(np.mean(distances[:, 3:6], axis=1), 1e-10)
    scaled_distance = original.scale_dist(distances, sigma, neighbors)
    ranked_pairs = original.sample_neighbors_pair(
        x_preprocessed,
        scaled_distance,
        neighbors,
        n_search,
    )
    return np.ascontiguousarray(
        ranked_pairs[:, 1].reshape(len(x_preprocessed), n_search),
        dtype=np.int32,
    )


@numba.njit(cache=True)
def _combine_legacy_neighbors(
    spatial_ranking: np.ndarray,
    k_spatial: int,
    attribute_ranking: np.ndarray,
    k_total: int,
) -> np.ndarray:
    """Fill a fixed-size graph with SC neighbours first, then feature NN."""
    n = attribute_ranking.shape[0]
    output = np.empty((n, k_total), dtype=np.int32)
    for i in range(n):
        count = 0
        for position in range(k_spatial):
            candidate = spatial_ranking[i, position]
            duplicate = False
            for previous in range(count):
                if output[i, previous] == candidate:
                    duplicate = True
                    break
            if not duplicate:
                output[i, count] = candidate
                count += 1

        for position in range(attribute_ranking.shape[1]):
            if count == k_total:
                break
            candidate = attribute_ranking[i, position]
            duplicate = False
            for previous in range(count):
                if output[i, previous] == candidate:
                    duplicate = True
                    break
            if not duplicate:
                output[i, count] = candidate
                count += 1
    return output


def _construct_legacy_sc_pairs(
    x_preprocessed: np.ndarray,
    coordinates_m: np.ndarray,
    n_neighbors: int,
    settings: PaCMAPSettings,
) -> tuple[np.ndarray, LegacySCDiagnostics]:
    """Construct the superseded mixed NN graph used as the red baseline."""
    attribute_ranking = _attribute_neighbor_ranking(
        x_preprocessed, n_neighbors, settings
    )
    ordinary_neighbors = attribute_ranking[:, :n_neighbors].copy()
    _, geographic_neighbors, _ = original.compute_nearest_neighbors(
        np.ascontiguousarray(coordinates_m, dtype=np.float32),
        n_neighbors,
        "euclidean",
        settings.knn_backend,
        settings.random_state,
    )

    n_components = min(
        100, x_preprocessed.shape[1], x_preprocessed.shape[0] - 1
    )
    pca = PCA(n_components=n_components, random_state=settings.random_state)
    scores = pca.fit_transform(x_preprocessed).astype(np.float64)
    score_std = np.std(scores, axis=0, ddof=0)
    score_std = np.where(score_std > 0.0, score_std, 1.0)
    standardized_scores = (
        scores - np.mean(scores, axis=0, keepdims=True)
    ) / score_std
    geary_c = _geary_c_by_component(
        standardized_scores, geographic_neighbors
    )
    # Geary C is non-negative, so positive continuity is naturally in [0, 1].
    continuity_weight = np.maximum(0.0, 1.0 - geary_c)
    spatial_strength = float(
        np.sum(
            pca.explained_variance_ratio_.astype(np.float64)
            * continuity_weight
        )
    )
    k_spatial = int(np.round(n_neighbors * spatial_strength))
    k_attribute = n_neighbors - k_spatial
    component_weights = (
        pca.explained_variance_.astype(np.float64) * continuity_weight
    )
    continuity_distance = _continuity_distance(
        np.ascontiguousarray(standardized_scores),
        np.ascontiguousarray(geographic_neighbors, dtype=np.int32),
        np.ascontiguousarray(component_weights),
    )
    spatial_order = np.argsort(
        continuity_distance, axis=1, kind="stable"
    )
    spatial_ranking = np.take_along_axis(
        geographic_neighbors, spatial_order, axis=1
    ).astype(np.int32)
    selected_spatial = spatial_ranking[:, :k_spatial].copy()
    final_neighbors = _combine_legacy_neighbors(
        np.ascontiguousarray(spatial_ranking),
        k_spatial,
        np.ascontiguousarray(attribute_ranking),
        n_neighbors,
    )
    diagnostics = LegacySCDiagnostics(
        geary_c=geary_c,
        continuity_weight=continuity_weight,
        spatial_strength_lambda=spatial_strength,
        k_spatial=k_spatial,
        k_attribute=k_attribute,
        ordinary_neighbors=ordinary_neighbors,
        selected_spatial_neighbors=selected_spatial,
        final_neighbors=final_neighbors,
    )
    return _neighbors_to_pairs(final_neighbors), diagnostics


# ---------------------------------------------------------------------------
# SC INJECTION 1: construct a separate PCA-Geary graph G_S
# ---------------------------------------------------------------------------


def _construct_separate_sc_graph(
    x_preprocessed: np.ndarray,
    coordinates_m: np.ndarray,
    pair_feature: np.ndarray,
    n_neighbors: int,
    settings: PaCMAPSettings,
) -> _SCGraph:
    """Keep original G_F complete and build an additional SC graph."""
    ordinary_neighbors = np.ascontiguousarray(
        pair_feature[:, 1].reshape(len(x_preprocessed), n_neighbors),
        dtype=np.int32,
    )

    _, geographic_neighbors, _ = original.compute_nearest_neighbors(
        np.ascontiguousarray(coordinates_m, dtype=np.float32),
        n_neighbors,
        "euclidean",
        settings.knn_backend,
        settings.random_state,
    )

    n_components = min(
        100,
        x_preprocessed.shape[1],
        x_preprocessed.shape[0] - 1,
    )
    pca = PCA(n_components=n_components, random_state=settings.random_state)
    scores = pca.fit_transform(x_preprocessed).astype(np.float64)
    score_std = np.std(scores, axis=0, ddof=0)
    score_std = np.where(score_std > 0.0, score_std, 1.0)
    standardized_scores = (
        scores - np.mean(scores, axis=0, keepdims=True)
    ) / score_std

    geary_c = _geary_c_by_component(
        standardized_scores, geographic_neighbors
    )

    # Geary C is non-negative, so positive continuity is naturally in [0, 1].
    continuity_weight = np.maximum(0.0, 1.0 - geary_c)
    spatial_strength = float(
        np.sum(
            pca.explained_variance_ratio_.astype(np.float64)
            * continuity_weight
        )
    )
    k_spatial = int(np.round(n_neighbors * spatial_strength))

    if k_spatial == 0:
        selected_spatial = np.empty(
            (len(x_preprocessed), 0), dtype=np.int32
        )
        pair_sc = np.empty((0, 2), dtype=np.int32)
        selected_distance = np.empty(0, dtype=np.float32)
    else:
        component_weights = (
            pca.explained_variance_.astype(np.float64) * continuity_weight
        )
        continuity_distance = _continuity_distance(
            np.ascontiguousarray(standardized_scores),
            np.ascontiguousarray(geographic_neighbors, dtype=np.int32),
            np.ascontiguousarray(component_weights),
        )
        spatial_order = np.argsort(
            continuity_distance, axis=1, kind="stable"
        )[:, :k_spatial]
        selected_spatial = np.take_along_axis(
            geographic_neighbors, spatial_order, axis=1
        ).astype(np.int32)
        selected_distance = np.take_along_axis(
            continuity_distance, spatial_order, axis=1
        ).reshape(-1).astype(np.float32)
        pair_sc = _neighbors_to_pairs(selected_spatial)

    feature_pair_set = {
        (int(i), int(j)) for i, j in np.asarray(pair_feature, dtype=np.int32)
    }
    overlap_mask = np.fromiter(
        ((int(i), int(j)) in feature_pair_set for i, j in pair_sc),
        dtype=np.bool_,
        count=len(pair_sc),
    )

    return _SCGraph(
        ordinary_neighbors=ordinary_neighbors,
        selected_spatial_neighbors=np.ascontiguousarray(selected_spatial),
        pair_sc=np.ascontiguousarray(pair_sc, dtype=np.int32),
        sc_distance=np.ascontiguousarray(selected_distance, dtype=np.float32),
        overlap_mask=np.ascontiguousarray(overlap_mask),
        geary_c=np.asarray(geary_c, dtype=np.float64),
        continuity_weight=np.asarray(continuity_weight, dtype=np.float64),
        spatial_strength_lambda=spatial_strength,
    )


# ---------------------------------------------------------------------------
# SC INJECTION 2: convert direct D_SC to edge confidence
# ---------------------------------------------------------------------------


def _make_sc_edge_weights(
    sc_distance: np.ndarray,
    overlap_mask: np.ndarray,
    overlap_factor: float,
) -> tuple[np.ndarray, float, float, float]:
    """Give smaller D_SC stronger attraction, then downweight overlap."""
    if len(sc_distance) == 0:
        return np.empty(0, dtype=np.float32), 1.0, 0.0, 0.0

    positive = sc_distance[sc_distance > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    scale = max(scale, 1e-12)
    raw_weight = 1.0 / (1.0 + sc_distance.astype(np.float64) / scale)
    raw_weight /= max(float(np.mean(raw_weight)), 1e-12)
    mean_before_overlap = float(np.mean(raw_weight))

    final_weight = raw_weight.astype(np.float32)
    final_weight[overlap_mask] *= np.float32(overlap_factor)
    return (
        np.ascontiguousarray(final_weight),
        scale,
        mean_before_overlap,
        float(np.mean(final_weight)),
    )


# ---------------------------------------------------------------------------
# SC INJECTION 3: calculate the Phase-3 attraction-gap share once
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _unit_attraction_sums(
    y: np.ndarray,
    pair_feature: np.ndarray,
    pair_sc: np.ndarray,
    sc_edge_weight: np.ndarray,
    c_sc: float,
) -> tuple[float, float]:
    """Return NN and SC force totals when both graph weights equal one."""
    nn_sum = 0.0
    for edge in range(pair_feature.shape[0]):
        i = pair_feature[edge, 0]
        j = pair_feature[edge, 1]
        squared_distance = 0.0
        for dimension in range(y.shape[1]):
            difference = y[i, dimension] - y[j, dimension]
            squared_distance += difference * difference
        d_tilde = 1.0 + squared_distance
        nn_sum += (
            20.0
            * math.sqrt(squared_distance)
            / ((10.0 + d_tilde) * (10.0 + d_tilde))
        )

    sc_sum = 0.0
    for edge in range(pair_sc.shape[0]):
        i = pair_sc[edge, 0]
        j = pair_sc[edge, 1]
        squared_distance = 0.0
        for dimension in range(y.shape[1]):
            difference = y[i, dimension] - y[j, dimension]
            squared_distance += difference * difference
        d_tilde = 1.0 + squared_distance
        sc_sum += (
            sc_edge_weight[edge]
            * (2.0 * c_sc)
            * math.sqrt(squared_distance)
            / ((c_sc + d_tilde) * (c_sc + d_tilde))
        )
    return nn_sum, sc_sum


def _automatic_target_share(
    spatial_strength_lambda: float,
    unit_nn_total: float,
    unit_sc_total: float,
    n_nn_edges: int,
    effective_sc_edge_count: float,
) -> float:
    """Convert lambda and the unresolved attraction gap to an SC share."""
    lam = float(spatial_strength_lambda)
    if unit_sc_total <= 1e-20 or effective_sc_edge_count <= 1e-20:
        return 0.0
    if unit_nn_total <= 1e-20:
        return lam

    mean_nn = unit_nn_total / max(float(n_nn_edges), 1.0)
    mean_sc = unit_sc_total / max(float(effective_sc_edge_count), 1e-12)
    numerator = lam * mean_sc
    denominator = numerator + (1.0 - lam) * mean_nn
    if denominator <= 1e-20:
        return lam
    return float(numerator / denominator)


def _phase3_ramp(
    iteration: int,
    num_iters: tuple[int, int, int],
    ramp_iterations: int,
) -> float:
    """Return zero before Phase 3, then linearly ramp to one."""
    phase3_start = num_iters[0] + num_iters[1]
    if iteration < phase3_start:
        return 0.0
    if ramp_iterations <= 0:
        return 1.0
    elapsed = iteration - phase3_start
    if elapsed >= ramp_iterations:
        return 1.0
    return float(elapsed / ramp_iterations)


# ---------------------------------------------------------------------------
# SC INJECTION 4: append one SC loss gradient to original.pacmap_grad
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _add_sc_gradient(
    gradient: np.ndarray,
    y: np.ndarray,
    pair_sc: np.ndarray,
    sc_edge_weight: np.ndarray,
    w_sc: float,
    c_sc: float,
) -> None:
    """Add L_SC while leaving original NN/MN/FP loss code untouched."""
    sc_loss = 0.0
    for edge in range(pair_sc.shape[0]):
        i = pair_sc[edge, 0]
        j = pair_sc[edge, 1]
        d_tilde = 1.0
        for dimension in range(y.shape[1]):
            difference = y[i, dimension] - y[j, dimension]
            d_tilde += difference * difference

        edge_coefficient = w_sc * sc_edge_weight[edge]
        sc_loss += edge_coefficient * d_tilde / (c_sc + d_tilde)
        gradient_coefficient = (
            edge_coefficient * (2.0 * c_sc) / ((c_sc + d_tilde) ** 2)
        )
        for dimension in range(y.shape[1]):
            difference = y[i, dimension] - y[j, dimension]
            gradient[i, dimension] += gradient_coefficient * difference
            gradient[j, dimension] -= gradient_coefficient * difference
    gradient[-1, 0] += sc_loss


def _initialize_embedding(
    x_preprocessed: np.ndarray,
    pca_solution: bool,
    transformer: Any,
    n_components: int,
) -> np.ndarray:
    """Repeat only the copied optimizer's PCA initialization expression."""
    if pca_solution:
        embedding = 0.01 * x_preprocessed[:, :n_components]
    else:
        embedding = 0.01 * transformer.transform(
            x_preprocessed
        ).astype(np.float32)
    return np.ascontiguousarray(embedding, dtype=np.float32)


def _optimize_with_sc_injection(
    x_preprocessed: np.ndarray,
    pca_solution: bool,
    transformer: Any,
    pair_feature: np.ndarray,
    pair_mn: np.ndarray,
    pair_fp: np.ndarray,
    sc_graph: _SCGraph,
    sc_edge_weight: np.ndarray,
    settings: PaCMAPSettings,
    config: SCPaCMAPConfig,
) -> tuple[np.ndarray, float, float, float, float]:
    """Run original PaCMAP with a separate SC attraction only in Phase 3."""
    embedding = _initialize_embedding(
        x_preprocessed,
        pca_solution,
        transformer,
        settings.n_components,
    )
    first_moment = np.zeros_like(embedding, dtype=np.float32)
    second_moment = np.zeros_like(embedding, dtype=np.float32)
    phase3_start = settings.num_iters[0] + settings.num_iters[1]
    target_share: float | None = None
    effective_sc_edge_count = float(np.sum(sc_edge_weight))
    active_realized_shares: list[float] = []
    active_w_sc: list[float] = []

    for iteration in range(sum(settings.num_iters)):
        w_mn, w_nn_original, w_fp = original.find_weight(
            1000.0,
            iteration,
            num_iters=settings.num_iters,
        )

        if iteration < phase3_start:
            w_feature = float(w_nn_original)
            w_sc = 0.0
            realized_share = 0.0
        else:
            unit_nn, unit_sc = _unit_attraction_sums(
                embedding,
                pair_feature,
                sc_graph.pair_sc,
                sc_edge_weight,
                float(config.c_sc),
            )
            if target_share is None:
                target_share = _automatic_target_share(
                    sc_graph.spatial_strength_lambda,
                    unit_nn,
                    unit_sc,
                    len(pair_feature),
                    effective_sc_edge_count,
                )

            ramp = _phase3_ramp(
                iteration,
                settings.num_iters,
                config.phase3_ramp_iterations,
            )
            scheduled_share = float(target_share * ramp)

            # Transfer the scheduled share out of NN attraction into G_S.
            w_feature = float(w_nn_original * (1.0 - scheduled_share))
            if scheduled_share > 0.0 and unit_sc > 1e-20:
                w_sc = float(
                    w_nn_original
                    * scheduled_share
                    * unit_nn
                    / unit_sc
                )
                realized_share = float(
                    (w_sc * unit_sc)
                    / max(w_feature * unit_nn + w_sc * unit_sc, 1e-20)
                )
            else:
                w_sc = 0.0
                realized_share = 0.0

        # ORIGINAL NN/MN/FP loss and gradient are called, not reimplemented.
        gradient = original.pacmap_grad(
            embedding,
            pair_feature,
            pair_mn,
            pair_fp,
            float(w_feature),
            float(w_mn),
            float(w_fp),
        )
        if w_sc > 0.0:
            _add_sc_gradient(
                gradient,
                embedding,
                sc_graph.pair_sc,
                sc_edge_weight,
                float(w_sc),
                float(config.c_sc),
            )
        original.update_embedding_adam(
            embedding,
            gradient,
            first_moment,
            second_moment,
            0.9,
            0.999,
            settings.learning_rate,
            iteration,
        )

        if iteration >= phase3_start and w_sc > 0.0:
            active_realized_shares.append(realized_share)
            active_w_sc.append(w_sc)

        if settings.verbose and (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1:4d} | "
                f"loss={gradient[-1, 0]:.6f} | "
                f"target={0.0 if target_share is None else target_share:.3f} | "
                f"w_sc={w_sc:.3f}"
            )

    return (
        np.ascontiguousarray(embedding, dtype=np.float32),
        0.0 if target_share is None else float(target_share),
        float(np.mean(active_realized_shares))
        if active_realized_shares
        else 0.0,
        float(np.mean(active_w_sc)) if active_w_sc else 0.0,
        float(np.max(active_w_sc)) if active_w_sc else 0.0,
    )


def run_legacy_sc_pacmap(
    features_standardized: np.ndarray,
    coordinates_m: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
) -> tuple[PaCMAPResult, LegacySCDiagnostics]:
    """Run SC-PaCMAP (neighbour replacement graph) as a benchmark baseline."""
    features = _as_features(features_standardized)
    coordinates = _as_coordinates(coordinates_m, len(features))
    n_neighbors, n_mn, n_fp = _decide_pair_counts(
        len(features),
        settings.n_neighbors,
        settings.mn_ratio,
        settings.fp_ratio,
    )
    original._RANDOM_STATE = settings.random_state
    x_pre, pca_solution, transformer = _preprocess_with_original(
        features, settings
    )
    pair_neighbors, diagnostics = _construct_legacy_sc_pairs(
        x_pre,
        coordinates,
        n_neighbors,
        settings,
    )
    pair_neighbors, pair_mn, pair_fp = original.generate_pair_no_neighbors(
        x_pre,
        n_neighbors,
        n_mn,
        n_fp,
        pair_neighbors,
        "euclidean",
        settings.verbose,
    )
    embedding = _optimize_with_original(
        x_pre,
        pca_solution,
        transformer,
        pair_neighbors,
        pair_mn,
        pair_fp,
        settings,
    )
    return PaCMAPResult(
        embedding=embedding,
        pair_neighbors=pair_neighbors,
        pair_mn=pair_mn,
        pair_fp=pair_fp,
        diagnostics={
            "mode": "legacy_sc_pacmap_benchmark",
            "n_neighbors": n_neighbors,
            "n_mn": n_mn,
            "n_fp": n_fp,
        },
    ), diagnostics


def run_sc_pacmap(
    features_standardized: np.ndarray,
    coordinates_m: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
    *,
    sc_enabled: bool = True,
    config: SCPaCMAPConfig = SCPaCMAPConfig(),
) -> tuple[PaCMAPResult, SCDiagnostics | None]:
    """Run SC-PaCMAP (full graph) with a separate spatial loss."""
    features = _as_features(features_standardized)
    coordinates = _as_coordinates(coordinates_m, len(features))

    if not sc_enabled:
        return run_copied_original_pacmap(features, settings), None

    n_neighbors, n_mn, n_fp = _decide_pair_counts(
        len(features),
        settings.n_neighbors,
        settings.mn_ratio,
        settings.fp_ratio,
    )
    original._RANDOM_STATE = settings.random_state
    x_pre, pca_solution, transformer = _preprocess_with_original(
        features, settings
    )

    # ORIGINAL: construct and retain the complete feature NN/MN/FP graphs.
    pair_neighbors, pair_mn, pair_fp, _ = original.generate_pair(
        x_pre,
        n_neighbors,
        n_mn,
        n_fp,
        "euclidean",
        settings.verbose,
        settings.knn_backend,
    )

    sc_graph = _construct_separate_sc_graph(
        x_pre,
        coordinates,
        pair_neighbors,
        n_neighbors,
        settings,
    )
    sc_edge_weight, distance_scale, mean_before, mean_after = (
        _make_sc_edge_weights(
            sc_graph.sc_distance,
            sc_graph.overlap_mask,
            config.overlap_sc_factor,
        )
    )

    if len(sc_graph.pair_sc) == 0:
        embedding = _optimize_with_original(
            x_pre,
            pca_solution,
            transformer,
            pair_neighbors,
            pair_mn,
            pair_fp,
            settings,
        )
        target_share = 0.0
        mean_realized_share = 0.0
        mean_w_sc = 0.0
        max_w_sc = 0.0
    else:
        (
            embedding,
            target_share,
            mean_realized_share,
            mean_w_sc,
            max_w_sc,
        ) = _optimize_with_sc_injection(
            x_pre,
            pca_solution,
            transformer,
            pair_neighbors,
            pair_mn,
            pair_fp,
            sc_graph,
            sc_edge_weight,
            settings,
            config,
        )

    sc_diagnostics = SCDiagnostics(
        geary_c=sc_graph.geary_c,
        continuity_weight=sc_graph.continuity_weight,
        spatial_strength_lambda=sc_graph.spatial_strength_lambda,
        k_spatial=sc_graph.selected_spatial_neighbors.shape[1],
        k_attribute=n_neighbors,
        ordinary_neighbors=sc_graph.ordinary_neighbors,
        selected_spatial_neighbors=sc_graph.selected_spatial_neighbors,
        # Kept for compatibility. G_F is no longer replaced or merged.
        final_neighbors=sc_graph.ordinary_neighbors.copy(),
        pair_sc=sc_graph.pair_sc,
        sc_distance=sc_graph.sc_distance,
        sc_edge_weight=sc_edge_weight,
        overlap_mask=sc_graph.overlap_mask,
        automatic_target_share=target_share,
        overlap_fraction=(
            float(np.mean(sc_graph.overlap_mask))
            if len(sc_graph.overlap_mask)
            else 0.0
        ),
        sc_distance_scale=distance_scale,
        mean_sc_weight_before_overlap=mean_before,
        mean_sc_weight_after_overlap=mean_after,
        mean_realized_sc_share_active=mean_realized_share,
        mean_w_sc_active=mean_w_sc,
        max_w_sc_active=max_w_sc,
        c_sc=config.c_sc,
        phase3_ramp_iterations=config.phase3_ramp_iterations,
    )
    return PaCMAPResult(
        embedding=embedding,
        pair_neighbors=pair_neighbors,
        pair_mn=pair_mn,
        pair_fp=pair_fp,
        diagnostics={
            "mode": "sc_pacmap",
            "n_neighbors": n_neighbors,
            "n_mn": n_mn,
            "n_fp": n_fp,
            "n_sc": sc_graph.selected_spatial_neighbors.shape[1],
            "automatic_target_share": target_share,
        },
    ), sc_diagnostics


def fit_transform(
    features_standardized: np.ndarray,
    coordinates_m: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
    config: SCPaCMAPConfig = SCPaCMAPConfig(),
) -> tuple[np.ndarray, SCDiagnostics]:
    """Notebook-friendly enhanced runner returning embedding and diagnostics."""
    result, diagnostics = run_sc_pacmap(
        features_standardized,
        coordinates_m,
        settings,
        config=config,
    )
    if diagnostics is None:
        raise RuntimeError("SC diagnostics were not produced")
    return result.embedding, diagnostics


def run_mode(
    mode: str,
    features_standardized: np.ndarray,
    coordinates_m: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
) -> tuple[PaCMAPResult, SCDiagnostics | None]:
    """The requested three-mode if/elif selector."""
    features = _as_features(features_standardized)
    coordinates = _as_coordinates(coordinates_m, len(features))

    if mode == MODE_PACMAP_FEATURES:
        # Mode 1: original PaCMAP, features only.
        return run_copied_original_pacmap(features, settings), None

    elif mode == MODE_PACMAP_FEATURES_XYZ:
        # Mode 2: exactly the same original PaCMAP code. Only input changes.
        xyz_standardized = StandardScaler().fit_transform(
            coordinates
        ).astype(np.float32)
        combined_input = np.column_stack(
            [features, xyz_standardized]
        ).astype(np.float32)
        return run_copied_original_pacmap(combined_input, settings), None

    elif mode == MODE_SC_PACMAP:
        # Mode 3: retain G_F and add a separate Phase-3 G_S attraction.
        return run_sc_pacmap(features, coordinates, settings, sc_enabled=True)

    raise ValueError(
        "mode must be 'pacmap_features', 'pacmap_features_xyz' or 'sc_pacmap'"
    )


def verify_sc_disabled_matches_original(
    features_standardized: np.ndarray,
    coordinates_m: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
) -> dict[str, bool | float]:
    """Prove the wrapper is exact when the SC injections are disabled."""
    ordinary = run_copied_original_pacmap(features_standardized, settings)
    disabled, _ = run_sc_pacmap(
        features_standardized,
        coordinates_m,
        settings,
        sc_enabled=False,
    )
    return {
        "neighbor_pairs_equal": np.array_equal(
            ordinary.pair_neighbors, disabled.pair_neighbors
        ),
        "mid_near_pairs_equal": np.array_equal(
            ordinary.pair_mn, disabled.pair_mn
        ),
        "further_pairs_equal": np.array_equal(
            ordinary.pair_fp, disabled.pair_fp
        ),
        "embedding_equal": np.array_equal(
            ordinary.embedding, disabled.embedding
        ),
        "maximum_embedding_error": float(
            np.max(np.abs(ordinary.embedding - disabled.embedding))
        ),
    }


def verify_library_matches_copied_code(
    x_input: np.ndarray,
    settings: PaCMAPSettings = PaCMAPSettings(),
) -> dict[str, Any]:
    """Compare installed ``pacmap==0.9.1`` with the copied code mode."""
    try:
        installed_version = version("pacmap")
    except PackageNotFoundError:
        return {
            "status": "skipped",
            "reason": "the external pacmap package is not installed",
        }

    if installed_version != "0.9.1":
        return {
            "status": "skipped",
            "reason": f"pacmap {installed_version} is installed, not 0.9.1",
        }

    import pacmap as pacmap_library

    x_input = _as_features(x_input)
    copied = run_copied_original_pacmap(x_input, settings)
    library_model = pacmap_library.PaCMAP(
        n_components=settings.n_components,
        n_neighbors=settings.n_neighbors,
        MN_ratio=settings.mn_ratio,
        FP_ratio=settings.fp_ratio,
        distance="euclidean",
        lr=settings.learning_rate,
        num_iters=settings.num_iters,
        verbose=settings.verbose,
        apply_pca=True,
        random_state=settings.random_state,
        save_tree=False,
        knn_backend=settings.knn_backend,
    )
    library_embedding = library_model.fit_transform(
        x_input, init="pca", save_pairs=True
    )

    result = {
        "status": "pass",
        "library_version": installed_version,
        "neighbor_pairs_equal": np.array_equal(
            copied.pair_neighbors, library_model.pair_neighbors
        ),
        "mid_near_pairs_equal": np.array_equal(
            copied.pair_mn, library_model.pair_MN
        ),
        "further_pairs_equal": np.array_equal(
            copied.pair_fp, library_model.pair_FP
        ),
        "embedding_equal": np.array_equal(
            copied.embedding, library_embedding
        ),
        "maximum_embedding_error": float(
            np.max(np.abs(copied.embedding - library_embedding))
        ),
    }
    if not all(
        result[key]
        for key in (
            "neighbor_pairs_equal",
            "mid_near_pairs_equal",
            "further_pairs_equal",
            "embedding_equal",
        )
    ):
        result["status"] = "fail"
    return result


def make_synthetic_data(
    n_samples: int = 240,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Small in-memory spatial dataset for a quick three-mode check."""
    rng = np.random.default_rng(random_state)
    coordinates = rng.uniform(0.0, 1000.0, size=(n_samples, 3))
    x, y, z = coordinates.T
    latent = np.column_stack(
        [
            np.sin(x / 150.0) + np.cos(y / 180.0),
            (x - y) / 700.0 + np.sin(z / 120.0),
            np.cos((x + y) / 260.0) + z / 900.0,
        ]
    )
    loading = rng.normal(size=(3, 12))
    features = latent @ loading + rng.normal(
        0.0, 0.35, size=(n_samples, 12)
    )
    features = StandardScaler().fit_transform(features).astype(np.float32)
    return features, coordinates.astype(np.float32)


def main() -> None:
    """Run the small in-memory demonstration; no result files are written."""
    features, coordinates = make_synthetic_data()
    settings = PaCMAPSettings()

    for mode in (
        MODE_PACMAP_FEATURES,
        MODE_PACMAP_FEATURES_XYZ,
        MODE_SC_PACMAP,
    ):
        result, sc = run_mode(mode, features, coordinates, settings)
        print(mode, result.embedding.shape)
        if sc is not None:
            print(
                "  lambda=",
                round(sc.spatial_strength_lambda, 4),
                "k_spatial=",
                sc.k_spatial,
                "k_attribute=",
                sc.k_attribute,
            )

    print(
        "SC disabled check:",
        verify_sc_disabled_matches_original(
            features, coordinates, settings
        ),
    )
    print(
        "Library versus copied code:",
        verify_library_matches_copied_code(features, settings),
    )


if __name__ == "__main__":
    main()
