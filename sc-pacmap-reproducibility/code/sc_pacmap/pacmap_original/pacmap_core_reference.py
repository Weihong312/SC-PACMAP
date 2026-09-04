"""Verbatim function extracts from the official PaCMAP 0.9.1 source.

The functions below are copied without changing their signatures or bodies.
Only this file-level provenance note was added. See SOURCE.txt for the exact
upstream URL and function map.
"""

import logging
import math
import time

import numba
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import preprocessing

# Mandatory FAISS import (default backend)
import faiss
# Optional imports for annoy and voyager
try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None

try:
    import voyager
except ImportError:
    voyager = None

global _RANDOM_STATE
_RANDOM_STATE = None

logger = logging.getLogger(__name__)


@numba.njit("f4(f4[:])", cache=True)
def l2_norm(x):
    """
    L2 norm of a vector.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def manhattan_dist(x1, x2):
    """
    Manhattan distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i] - x2[i])
    return result


@numba.njit("f4(f4[:],f4[:])", cache=True)
def angular_dist(x1, x2):
    """
    Angular (i.e. cosine) distance between two vectors.
    """
    x1_norm = np.maximum(l2_norm(x1), 1e-20)
    x2_norm = np.maximum(l2_norm(x2), 1e-20)
    result = 0.0
    for i in range(x1.shape[0]):
        result += x1[i] * x2[i]
    return np.sqrt(2.0 - 2.0 * result / x1_norm / x2_norm)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def hamming_dist(x1, x2):
    """
    Hamming distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@numba.njit(cache=True)
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:  # euclidean
        return euclid_dist(x1, x2)
    elif distance_index == 1:  # manhattan
        return manhattan_dist(x1, x2)
    elif distance_index == 2:  # angular
        return angular_dist(x1, x2)
    elif distance_index == 3:  # hamming
        return hamming_dist(x1, x2)


@numba.njit("i4[:](i4,i4,i4[:],i4)", nogil=True, cache=True)
def sample_FP(n_samples, maximum, reject_ind, self_ind):
    """Sample `n_samples` samples from `maximum` points, excluding the `reject_ind` points."""
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(maximum)
            if j == self_ind:
                continue
            for k in range(i):
                if j == result[k]:
                    break
            else:
                for k in range(reject_ind.shape[0]):
                    if j == reject_ind[k]:
                        break
                else:
                    reject_sample = False
        result[i] = j
    return result


@numba.njit("i4[:,:](f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True, cache=True)
def sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors):
    n = X.shape[0]
    pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)
    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](f4[:,:],i4,i4)", nogil=True, cache=True)
def sample_MN_pair(X, n_MN, option=0):
    '''Sample Mid Near pairs.'''
    n = X.shape[0]
    pair_MN = np.empty((n * n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for j in range(n_MN):
            sampled = sample_FP(
                n_samples=6,
                maximum=n,
                reject_ind=pair_MN[i * n_MN:i * n_MN + j, 1],
                self_ind=i
            )
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_index=option)
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i * n_MN + j][0] = i
            pair_MN[i * n_MN + j][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4,i4,i4)", nogil=True, cache=True)
def sample_MN_pair_deterministic(X, n_MN, random_state, option=0):
    '''Sample Mid Near pairs using the given random state.'''
    n = X.shape[0]
    pair_MN = np.empty((n * n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for j in range(n_MN):
            # Shifting the seed to prevent sampling the same pairs
            np.random.seed(random_state + i * n_MN + j)
            sampled = sample_FP(
                n_samples=6,
                maximum=n,
                reject_ind=pair_MN[i * n_MN:i * n_MN + j, 1],
                self_ind=i
            )
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_index=option)
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i * n_MN + j][0] = i
            pair_MN[i * n_MN + j][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP):
    '''Sample Further pairs.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        FP_index = sample_FP(
            n_samples=n_FP,
            maximum=n,
            reject_ind=pair_neighbors[i * n_neighbors:(i + 1) * n_neighbors, 1],
            self_ind=i
        )
        for k in numba.prange(n_FP):
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair_deterministic(X, pair_neighbors, n_neighbors, n_FP, random_state):
    '''Sample Further pairs using the given random state.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        np.random.seed(random_state + i)
        FP_index = sample_FP(
            n_samples=n_FP,
            maximum=n,
            reject_ind=pair_neighbors[i * n_neighbors:(i + 1) * n_neighbors, 1],
            self_ind=i
        )
        for k in numba.prange(n_FP):
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True, cache=True)
def scale_dist(knn_distance, sig, nbrs):
    '''Scale the distance'''
    n, num_neighbors = knn_distance.shape
    scaled_dist = np.zeros((n, num_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(num_neighbors):
            scaled_dist[i, j] = knn_distance[i, j] ** 2 / \
                sig[i] / sig[nbrs[i, j]]
    return scaled_dist


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4,f4,i4)", parallel=True, nogil=True, cache=True)
def update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr):
    '''Update the embedding with the gradient'''
    n, dim = Y.shape
    lr_t = lr * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
    for i in numba.prange(n):
        for d in numba.prange(dim):
            m[i][d] += (1 - beta1) * (grad[i][d] - m[i][d])
            v[i][d] += (1 - beta2) * (grad[i][d]**2 - v[i][d])
            Y[i][d] -= lr_t * m[i][d]/(math.sqrt(v[i][d]) + 1e-7)


@numba.njit("f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4[:,:],f4,f4,f4)", parallel=True, nogil=True, cache=True)
def pacmap_grad(Y, pair_neighbors, pair_MN, pair_FP, w_neighbors, w_MN, w_FP):
    '''Calculate the gradient for pacmap embedding given the particular set of weights.'''
    n, dim = Y.shape
    grad = np.zeros((n + 1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(4, dtype=np.float32)
    # NN
    for t in range(pair_neighbors.shape[0]):
        i = pair_neighbors[t, 0]
        j = pair_neighbors[t, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[0] += w_neighbors * (d_ij / (10. + d_ij))
        w1 = w_neighbors * (20. / (10. + d_ij) ** 2)
        for d in range(dim):
            grad[i, d] += w1 * y_ij[d]
            grad[j, d] -= w1 * y_ij[d]
    # MN
    for tt in range(pair_MN.shape[0]):
        i = pair_MN[tt, 0]
        j = pair_MN[tt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i][d] - Y[j][d]
            d_ij += y_ij[d] ** 2
        loss[1] += w_MN * d_ij / (10000. + d_ij)
        w = w_MN * 20000. / (10000. + d_ij) ** 2
        for d in range(dim):
            grad[i, d] += w * y_ij[d]
            grad[j, d] -= w * y_ij[d]
    # FP
    for ttt in range(pair_FP.shape[0]):
        i = pair_FP[ttt, 0]
        j = pair_FP[ttt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[2] += w_FP * 1. / (1. + d_ij)
        w1 = w_FP * 2. / (1. + d_ij) ** 2
        for d in range(dim):
            grad[i, d] -= w1 * y_ij[d]
            grad[j, d] += w1 * y_ij[d]
    grad[-1, 0] = loss.sum()
    return grad


def find_weight(w_MN_init, itr, *, num_iters):
    '''Find the corresponding weight given the index of an iteration'''
    (phase_1_iters, phase_2_iters, _) = num_iters
    if itr < phase_1_iters:
        w_MN = (1 - itr/phase_1_iters) * w_MN_init + itr/phase_1_iters * 3.0
        w_neighbors = 2.0
        w_FP = 1.0
    elif itr < phase_1_iters + phase_2_iters:
        w_MN = 3.0
        w_neighbors = 3
        w_FP = 1
    else:
        w_MN = 0.0
        w_neighbors = 1.
        w_FP = 1.
    return w_MN, w_neighbors, w_FP


def preprocess_X(X, distance, apply_pca, verbose, seed, high_dim, low_dim):
    '''Preprocess a dataset.
    '''
    tsvd = None
    pca_solution = False
    if distance != "hamming" and high_dim > 100 and apply_pca:
        xmin = 0  # placeholder
        xmax = 0  # placeholder
        xmean = np.mean(X, axis=0)
        X -= xmean
        tsvd = TruncatedSVD(n_components=100, random_state=seed)
        X = tsvd.fit_transform(X)
        pca_solution = True
        print_verbose("Applied PCA, the dimensionality becomes 100", verbose)
    else:
        xmin = np.min(X)
        X -= xmin
        xmax = np.max(X)
        X /= xmax
        xmean = np.mean(X, axis=0)
        X -= xmean
        tsvd = PCA(n_components=low_dim, random_state=seed)  # for init only
        tsvd.fit(X)
        print_verbose("X is normalized", verbose)
    return X, pca_solution, tsvd, xmin, xmax, xmean


def distance_to_option(distance='euclidean'):
    '''A helper function that translates distance metric to int options.
    Such a translation is useful for numba acceleration.
    '''
    if distance == 'euclidean':
        option = 0
    elif distance == 'manhattan':
        option = 1
    elif distance == 'angular':
        option = 2
    elif distance == 'hamming':
        option = 3
    else:
        raise NotImplementedError('Distance other than euclidean, manhattan,' +
                                  'angular or hamming is not supported')
    return option


def print_verbose(msg, verbose, **kwargs):
    if verbose:
        print(msg, **kwargs)


def compute_nearest_neighbors(X, n_neighbors, distance, knn_backend, random_state):
    '''Helper function to compute nearest neighbors.'''
    n, dim = X.shape
    # sample more neighbors than needed because the first one is the point itself
    k = n_neighbors + 1

    tree = None
    nbrs = None
    knn_distances = None
    
    if knn_backend == 'annoy':
        if AnnoyIndex is None:
            raise ImportError("knn_backend='annoy' requested, but annoy is not installed.")
        tree = AnnoyIndex(dim, metric=distance)
        if random_state is not None:
            tree.set_seed(random_state)
        for i in range(n):
            tree.add_item(i, X[i, :])
        tree.build(20)

        nbrs = np.zeros((n, n_neighbors), dtype=np.int32)
        knn_distances = np.empty((n, n_neighbors), dtype=np.float32)
        for i in range(n):
            nbrs_ = tree.get_nns_by_item(i, k)
            nbrs[i, :] = nbrs_[1:]
            for j in range(n_neighbors):
                knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])

    elif knn_backend == 'faiss':
        if faiss is None:
            raise ImportError("faiss is not installed.")

        if random_state is not None:
            # logger.warning("Using Faiss with random state: HNSW construction is non-deterministic.")
            faiss.omp_set_num_threads(1)
        # Determine metric
        if distance == 'euclidean':
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
            index.add(X)
            tree = index
            D, I = index.search(X, k)
            nbrs = I[:, 1:].astype(np.int32)
            knn_distances = np.sqrt(D[:, 1:]).astype(np.float32)

        elif distance == 'angular':
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            X_norm = X.copy()
            faiss.normalize_L2(X_norm)
            index.add(X_norm)
            tree = index

            D, I = index.search(X_norm, k)
            nbrs = I[:, 1:].astype(np.int32)
            D = np.minimum(D[:, 1:], 1.0)
            knn_distances = np.sqrt(2 * (1 - D)).astype(np.float32)

        elif distance == 'manhattan':
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L1)
            index.add(X)
            tree = index

            D, I = index.search(X, k)
            nbrs = I[:, 1:].astype(np.int32)
            knn_distances = D[:, 1:].astype(np.float32)
        else:
            raise NotImplementedError(f"Faiss backend does not support {distance} metric in this implementation.")

    elif knn_backend == 'voyager':
        if voyager is None:
            raise ImportError("voyager is not installed.")

        if random_state is not None:
            logger.warning("Using Voyager with random state: Determinism depends on the underlying Voyager implementation.")
        if distance == 'euclidean':
            metric = voyager.Space.Euclidean
        elif distance == 'angular':
            metric = voyager.Space.Cosine
        else:
            raise NotImplementedError(f"Voyager backend does not support {distance} metric in this implementation.")

        # Voyager accepts random_seed for deterministic behavior
        index = voyager.Index(metric, num_dimensions=dim, random_seed=random_state if random_state is not None else 1)
        index.add_items(X, num_threads=1)
        tree = index

        # Query
        I, D = index.query(X, k=k)

        # Exclude self
        nbrs = I[:, 1:].astype(np.int32)
        knn_distances = np.sqrt(D[:, 1:]).astype(np.float32)
    else:
        raise ValueError(f"Unknown knn_backend: {knn_backend}")

    return tree, nbrs, knn_distances


def generate_pair(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        distance='euclidean',
        verbose=True,
        knn_backend='faiss'
):
    '''Generate pairs for the dataset.
    '''
    n, dim = X.shape
    # sample more neighbors than needed
    if n - 1 < n_neighbors:
        logger.warning("Sample size is smaller than number of neighbor pairs"
                       " requested. n_neighbors will be reduced.")
    n_neighbors_extra = min(n_neighbors + 50, n - 1)
    n_neighbors = min(n_neighbors, n - 1)
    if n - 1 < n_FP:
        logger.warning("Sample size is smaller than number of further pairs"
                       " requested. n_FP will be reduced.")
    n_FP = min(n_FP, n - 1)
    if n - 1 < n_MN:
        logger.warning("Sample size is smaller than number of mid-near pairs"
                       " requested. n_MN will be reduced.")
    n_MN = min(n_MN, n - 1)
    if n_neighbors + n_MN + n_FP >= n:
        logger.warning("Sample size is smaller than the total number of assigned points, n_neighbors, n_MN, n_FP would be reorganized")

        # Infer ratios from the inputs provided
        if n_neighbors > 0:
            mn_r = n_MN / n_neighbors
            fp_r = n_FP / n_neighbors

            n_neighbors = int(n / (1 + mn_r + fp_r))
            n_MN = int(n_neighbors * mn_r)
            n_FP = int(n_neighbors * fp_r)
        else:
            # Edge case handling
            n_neighbors = 0
            n_MN = 0
            n_FP = 0

    tree, nbrs, knn_distances = compute_nearest_neighbors(
        X, n_neighbors_extra, distance, knn_backend, _RANDOM_STATE
    )
    print_verbose("Found nearest neighbor", verbose)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    print_verbose("Calculated sigma", verbose)
    scaled_dist = scale_dist(knn_distances, sig, nbrs)
    print_verbose("Found scaled dist", verbose)
    pair_neighbors = sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors)

    option = distance_to_option(distance=distance)

    if _RANDOM_STATE is None:
        pair_MN = sample_MN_pair(X, n_MN, option)
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_MN = sample_MN_pair_deterministic(X, n_MN, _RANDOM_STATE, option)
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    return pair_neighbors, pair_MN, pair_FP, tree


def generate_pair_no_neighbors(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        pair_neighbors,
        distance='euclidean',
        verbose=True
):
    '''Generate mid-near pairs and further pairs for a given dataset.
    This function is useful when the nearest neighbors comes from a given set.
    '''
    option = distance_to_option(distance=distance)
    
    if _RANDOM_STATE is None:
        pair_MN = sample_MN_pair(X, n_MN, option)
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_MN = sample_MN_pair_deterministic(X, n_MN, _RANDOM_STATE, option)
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    return pair_neighbors, pair_MN, pair_FP


def pacmap(
        X,
        n_dims,
        pair_neighbors,
        pair_MN,
        pair_FP,
        lr,
        num_iters,
        Yinit,
        verbose,
        intermediate,
        inter_snapshots,
        pca_solution,
        tsvd=None
):
    start_time = time.time()
    n, _ = X.shape

    if intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32)
    else:
        intermediate_states = None
    # Initialize the embedding
    if isinstance(Yinit, np.ndarray):
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = scaler.transform(Yinit) * 0.0001
    elif Yinit is None or (isinstance(Yinit, str) and Yinit == "pca"):
        if pca_solution:
            Y = 0.01 * X[:, :n_dims]
        else:
            Y = 0.01 * tsvd.transform(X).astype(np.float32)
    elif (isinstance(Yinit, str) and Yinit == "random"):  # random or hamming distance
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else:
        raise ValueError((f"The argument init is of the type {type(Yinit)}. "
                          "Currently, PaCMAP only supports user supplied "
                          "numpy.ndarray object as input, or one of "
                          "['pca', 'random']."))
    # Initialize parameters for optimizer
    w_MN_init = 1000.
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(
        (pair_neighbors.shape, pair_MN.shape, pair_FP.shape), verbose)

    num_iters_total = sum(num_iters)
    for itr in range(num_iters_total):
        w_MN, w_neighbors, w_FP = find_weight(w_MN_init, itr, num_iters=num_iters)

        grad = pacmap_grad(Y, pair_neighbors, pair_MN,
                           pair_FP, w_neighbors, w_MN, w_FP)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)
        if intermediate:
            if (itr + 1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = time.time() - start_time
    print_verbose(f"Elapsed time: {elapsed:.2f}s", verbose)

    return Y, intermediate_states, pair_neighbors, pair_MN, pair_FP
