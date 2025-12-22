import math
import warnings

import numpy as np
from numba import njit
from scipy import sparse as sc_sparse
from typing import Literal
from illico.utils.groups import GroupContainer


@njit(fastmath=True, nogil=True, cache=False)
def _add_at_scalar(a: np.ndarray, b: np.ndarray, c: float | int) -> None:
    """Equivalent on np.add.at with a scalar value to accumulate.

    Args:
        a (np.ndarray): Placeholder result array
        b (np.ndarray): Indices
        c (float | int): Scalar value to accumulate in `a` at `indices`.

    Author: Rémy Dubois
    """
    for i in range(len(b)):
        a[b[i]] += c


@njit(fastmath=True, nogil=True, cache=False)
def _add_at_vec(a: np.ndarray, b: np.ndarray, c: float | int) -> None:
    """Equivalent of np.add.at with a vector holding values to accumulate.

    Args:
        a (np.ndarray): Placeholder result array
        b (np.ndarray): Indices
        c (float | int): Vector holding value to accumulate in `a` at `indices`.

    Author: Rémy Dubois
    """
    for i in range(len(b)):
        a[b[i]] += c[i]


# TODO: overload np.diff properly
@njit(fastmath=True, nogil=True, cache=False)
def diff(x: np.ndarray) -> np.ndarray:
    """Equivalent of np.diff.

    For some reasons, np.ndiff failed to compile properly.

    Args:
        x (np.ndarray): Input 1-d array

    Returns:
        np.ndarray: Results diff array, of size x.size - 1.

    Author: Rémy Dubois
    """
    assert x.ndim == 1
    result = np.empty(x.size - 1, dtype=x.dtype)
    for i in range(x.size - 1):
        result[i] = x[i + 1] - x[i]
    return result


@njit(nogil=True, fastmath=False, cache=False)
def compute_pval(
    n_ref: int,
    n_tgt: int,
    n: int,
    tie_sum: float,
    U: float,
    mu: float,
    contin_corr: float = 0.0,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> float:
    """Compute p-value.

    This small piece of code was isolated here because it was duplicated in the
    all six routines.

    Args:
        n_ref (int): Number of reference (control) values (cells)
        n_tgt (int): Number of perturbed (targeted) values (cells)
        n (int): Total number of values (cells)
        tie_sum (float): Tie sum
        U (float): U-statistic
        mu (float): Mean
        contin_corr (float, optional): Continuity correction factor. Defaults to 0.0.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis.

    Returns:
        float: P-value

    Author: Rémy Dubois
    """
    tie_corr = 1.0 - tie_sum / (n * (n - 1) * (n + 1))
    if tie_corr > 1.0e-9:  # TODO: do that properly
        sigma = np.sqrt(n_ref * n_tgt * (n_ref + n_tgt + 1) / 12.0 * tie_corr)
        
        if alternative == "two-sided":  # two-sided
            # Compute both-sided statistic
            U = min(U, n_ref * n_tgt - U)
            delta = U - mu
            z = (np.abs(delta) + np.sign(delta) * contin_corr) / sigma
            return math.erfc(z / math.sqrt(2.0))
        elif alternative == "greater":  # greater (right-tailed)
            delta = U - mu
            z = (delta - contin_corr) / sigma
            # P(Z > z) = 0.5 * erfc(z / sqrt(2))
            return 0.5 * math.erfc(z / math.sqrt(2.0))
        elif alternative == "less" :  # less (left-tailed)
            delta = U - mu
            z = (delta + contin_corr) / sigma
            # P(Z < z) = 0.5 * erfc(-z / sqrt(2))
            return 0.5 * math.erfc(-z / math.sqrt(2.0))
        else:
            raise ValueError(f"Unsupported alternative hypothesis: {alternative}")
    else:
        return 1.0


@njit(nogil=True, cache=False)
def sampled_max(data: np.ndarray, sample_size: int = 200_000) -> float:
    max_val = -np.inf
    n = data.size
    step = max(1, n // sample_size)
    for i in range(0, n, step):
        if data[i] > max_val:
            max_val = data[i]
    return max_val


def _warn_log1p(X: np.ndarray | sc_sparse.spmatrix, is_log1p: bool, sample_size: int = 2e5):
    """Warns if user's indication and data values don't match.

    Args:
        X (np.ndarray|sc_sparse.spmatrix): Values
        is_log1p (bool): User-indicated flag
        sample_size (int, optional): Number of values to sample to estimate max. Defaults to 2e5.

    Raises:
        ValueError: If data is neither sparse nor dense.

    Author: Rémy Dubois
    """
    if isinstance(X, sc_sparse.spmatrix):
        data = X.data
    elif isinstance(X, np.ndarray):
        data = X.ravel()
    else:
        raise ValueError(f"Unsupported data type: {type(X)}")
    max_val = sampled_max(data, sample_size=sample_size)
    if is_log1p:
        if max_val > 15:
            warnings.warn(
                f"User indicated is_log1p=True, but estimated data max value is {max_val:.2f}, "
                "Which seems inconsistent. Make sure data is indeed log1p transformed.",
                UserWarning,
            )
    else:
        if max_val < 15:
            warnings.warn(
                f"User indicated is_log1p=False, but estimated data max value is {max_val:.2f}, "
                "Which seems inconsistent. Make sure data is indeed raw counts.",
                UserWarning,
            )


@njit(nogil=True, fastmath=True, cache=False)
def fold_change_from_summed_expr(group_agg_counts: np.ndarray, grpc: GroupContainer) -> np.ndarray:
    """Compute fold change from summed expression values, per group.

    Args:
        group_agg_counts (np.ndarray): Sum of expression values of shape (n_groups, n_genes)
        grpc (GroupContainer): GroupContainer holding group information

    Returns:
        np.ndarray: Fold change values of shape (n_groups, n_genes)

    Author: Rémy Dubois
    """
    assert group_agg_counts.shape[0] == grpc.counts.size
    assert group_agg_counts.ndim == 2
    mu_tgt = group_agg_counts / np.expand_dims(grpc.counts, -1)
    if grpc.encoded_ref_group == -1:
        # If one-versus-rest, the reference is all but the group
        ref_agg_counts = group_agg_counts.sum(axis=0)[np.newaxis, :] - group_agg_counts  # (n_groups, n_genes)
        ref_counts = np.expand_dims(grpc.counts.sum() - grpc.counts, -1)  # (n_groups, 1)
        mu_ref = ref_agg_counts / ref_counts
    else:
        # Else, the reference is the reference group
        mu_ref = np.expand_dims(mu_tgt[grpc.encoded_ref_group], 0)  # (1, n_genes)
    fold_change = np.where(mu_ref == 0, np.inf, mu_tgt / mu_ref)
    return fold_change


@njit(nogil=True, fastmath=True, cache=False)
def dense_fold_change(X: np.ndarray, grpc: GroupContainer, is_log1p: bool) -> np.ndarray:
    """Compute fold change from a dense array of expression counts.

    Args:
        X (np.ndarray): Expression counts
        grpc (GroupContainer): GroupContainer holding group information
        is_log1p (bool): User-indicated flag if data is log1p or not.

    Returns:
        np.ndarray: Fold change values of shape (n_groups, n_genes)

    Author: Rémy Dubois
    """
    group_agg_counts = np.zeros(shape=(grpc.counts.size, X.shape[1]), dtype=np.float64)
    # Sum expressions per group
    _add_at_vec(group_agg_counts, grpc.encoded_groups, np.expm1(X) if is_log1p else X)
    # for group_id in range(grpc.counts.size):
    #     idx_start = grpc.indptr[group_id]
    #     idx_end = grpc.indptr[group_id + 1]
    #     if is_log1p:
    #         group_agg_counts[group_id, :] = np.expm1(X[idx_start:idx_end]).sum(axis=0)
    #     else:
    #         group_agg_counts[group_id, :] = X[idx_start:idx_end].sum(axis=0)
    fold_change = fold_change_from_summed_expr(group_agg_counts, grpc)
    return fold_change


def compute_sparsity(X: np.ndarray | sc_sparse.spmatrix) -> float:
    """Compute sparsity of the data matrix.

    Args:
        X (np.ndarray | sc_sparse.spmatrix): Data matrix

    Returns:
        float: Sparsity (fraction of zero elements)

    Author: Rémy Dubois
    """
    if isinstance(X, sc_sparse.spmatrix):
        n_elements = X.shape[0] * X.shape[1]
        n_nonzero = X.nnz
    elif isinstance(X, np.ndarray):
        n_elements = X.size
        n_nonzero = np.count_nonzero(X)
    else:
        raise ValueError(f"Unsupported data type: {type(X)}")
    sparsity = 1.0 - (n_nonzero / n_elements)
    return sparsity


@njit(nogil=True, fastmath=True, cache=True, boundscheck=False)
def chunk_and_fortranize(X: np.ndarray, chunk_lb: int, chunk_ub: int, indices: np.ndarray | None = None) -> np.ndarray:
    """Vertically chunk the input array and converts it to Fortran-contiguous.

    The reason to be of the conversion is that later operations access the columns of this array so F order is advantageous. Also,
    this function performs one memory allocation instead of 2, which happens if calling np.asfortranarray on top of fancy-indexing.

    NB: If indices is None, then all rows are taken as is.

    Args:
        X (np.ndarray): Input dense array
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        indices (np.ndarray): Indices to reorder rows. There can be less indices than rows in X.

    Returns:
        np.ndarray: Chunked Fortran-contiguous array with reordered rows.

    Author: Rémy Dubois
    """
    # Now just fill it by making groups contiguous vertically, this will speed up sorting later on.
    if indices is not None:
        chunk = np.empty((chunk_ub - chunk_lb, indices.size), dtype=X.dtype).T  # transpose it to get Fortran order
        for i in range(indices.size):
            for j in range(0, chunk_ub - chunk_lb):
                chunk[i, j] = X[indices[i], chunk_lb + j]
    else:
        chunk = np.empty((chunk_ub - chunk_lb, X.shape[0]), dtype=X.dtype).T  # transpose it to get Fortran order
        for i in range(X.shape[0]):
            for j in range(0, chunk_ub - chunk_lb):
                chunk[i, j] = X[i, chunk_lb + j]
    return chunk
