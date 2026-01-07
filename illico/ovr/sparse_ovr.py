from typing import Literal

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import _add_at_scalar, compute_pval, diff
from illico.utils.ranking import _accumulate_group_ranksums_from_argsort
from illico.utils.registry import KernelDataFormat, Test, dispatcher_registry
from illico.utils.sparse.csc import (
    CSCMatrix,
    _assert_is_csc,
    csc_fold_change,
    csc_get_cols,
)
from illico.utils.sparse.csr import (
    CSRMatrix,
    _assert_is_csr,
    csr_get_contig_cols_into_csc,
)


@njit(fastmath=True, nogil=True, cache=False)
def sparse_ovr_mwu_kernel(
    X: CSCMatrix,
    groups: np.ndarray,
    group_counts: np.ndarray,
    use_continuity: bool = True,
    tie_correct: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Perform OVR ranksum test group wise and column wise on a CSC matrix.

    Args:
        X (CSCMatrix): CSC matrix holding expression counts.
        groups (np.ndarray): np.ndarray of shape (n_cells, ) holding encoded group labels.
        group_counts (np.ndarray): Count of cells per group.
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis. Defaults to "two-sided".

    Returns:
        tuple[np.ndarray]: Two-sided p-values and U-statistics, per group and per gene (column).

    Author: Rémy Dubois
    """
    _, n_cols = X.shape
    # Convert n_zeros to float64 as they will be used for tie sum later
    n_zeros = (X.shape[0] - diff(X.indptr)).astype(np.float64)
    # Allocate placeholders for U stats and pvals
    U = np.empty((group_counts.size, n_cols), dtype=np.float64)
    pvals = np.empty((group_counts.size, n_cols), dtype=np.float64)

    # Note that because this function does not involve inner parallelism, this could be allocated per-col, but I find it cleaner this way
    nnz_per_group = np.zeros((group_counts.size, n_cols), dtype=np.float64)
    R1_nz = np.zeros((group_counts.size, n_cols), dtype=np.float64)
    # Note that if we run this function over chunks of columns, this work is repeated in each func but this is cheap
    n = group_counts.sum()
    n_ref = n - group_counts
    n_tgt = group_counts
    mu = n_ref * n_tgt / 2.0
    for j in range(n_cols):
        start, end = X.indptr[j], X.indptr[j + 1]
        nz_idx = X.indices[start:end]
        """Step 1: compute ranksum of non-zero elements, per group"""
        _idxs = np.argsort(X.data[start:end])
        tie_sum = _accumulate_group_ranksums_from_argsort(X.data[start:end], _idxs, groups[nz_idx], R1_nz[:, j])
        n0 = n_zeros[j]

        """Step 2: offset non-zero elements ranks by the number of zeros that precedes them"""
        if nz_idx.size:
            _add_at_scalar(nnz_per_group[:, j], groups[nz_idx], 1.0)
        # Deduce number of zeros per group
        nz_per_group = group_counts - nnz_per_group[:, j]
        # Offset the non-zero ranks by the amount of 0 that precedes them
        # All ranks must be shifted, so the sum is shifted by that many elements.
        R1_nz[:, j] += n0 * nnz_per_group[:, j]

        """ Step 3: Add ranksums of zero elements, per group"""
        # add zero contribution: number of zeros * avg rank
        R1 = R1_nz[:, j] + nz_per_group * (n0 + 1) / 2.0
        U[:, j] = n_ref * n_tgt + n_tgt * (n_tgt + 1) / 2 - R1
        tie_sum += n0**3 - n0

        for k in range(group_counts.size):
            pvals[k, j] = compute_pval(
                n_ref=n_ref[k],
                n_tgt=n_tgt[k],
                n=n,
                tie_sum=tie_sum if tie_correct else 0.0,
                U=U[k, j],
                mu=mu[k],
                contin_corr=0.5 if use_continuity else 0.0,
                alternative=alternative,
            )

    return pvals, U


@dispatcher_registry.register(Test.OVR, KernelDataFormat.CSC)
@njit(nogil=True, fastmath=True, cache=False)
def csc_ovr_mwu_kernel_over_contiguous_col_chunk(
    X: CSCMatrix,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    tie_correct: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
):
    """Perform OVR ranksum test over the contiguous column chunk defined by the bounds.

    This function only applies to data stored in a CSC matrix.

    Args:
        X (CSCMatrix): Input CSC matrix holding expression counts
        chunk_lb (int): Lower bound of the vertial slice
        chunk_ub (int): Upper bound of the vertical slice
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data was log1p transformed or not.
        use_continuity (bool): Whether to use continuity correction when computing p-values. Defaults to True.
        tie_correct (bool): Whether to apply tie correction when computing p-values. Defaults to True.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis.

    Raises:
        ValueError: If bounds are not intelligible

    Returns:
        tuple[np.ndarray]: two-sided p-values, u-statistics and fold changes,
        each of shape (n_groups, chunk_lb - chunk_ub).

    Author: Rémy Dubois
    """
    _assert_is_csc(X)

    csc_chunk = csc_get_cols(csc_matrix=X, indices=np.arange(chunk_lb, chunk_ub))

    # TODO: un-jitting this function comes at close to no cost, and allows to do argsorting out of the njit function
    # on linux machines, it is 3 to 4 times faster than numba.np.argsort and sorting seems to be half the compute time of the whole function
    # idxs = np.empty_like(csc_chunk.indices)
    # for j in range(csc_chunk.shape[1]):
    #     start, end = csc_chunk.indptr[j], csc_chunk.indptr[j + 1]
    #     idxs[start:end] = np.argsort(csc_chunk.data[start:end])
    pvalues, statistics = sparse_ovr_mwu_kernel(
        X=csc_chunk,
        groups=grpc.encoded_groups,
        group_counts=grpc.counts,
        use_continuity=use_continuity,
        tie_correct=tie_correct,
        alternative=alternative,
    )

    fold_change = csc_fold_change(X=csc_chunk, grpc=grpc, is_log1p=is_log1p)
    return pvalues, statistics, fold_change


@dispatcher_registry.register(Test.OVR, KernelDataFormat.CSR)
@njit(nogil=True, fastmath=True, cache=False)  # This requires too many caching
def csr_ovr_mwu_kernel_over_contiguous_col_chunk(
    X: CSRMatrix,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    tie_correct: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Perform OVR ranksum test over the contiguous column chunk defined by the bounds.

    This function only applies to data stored in a CSR matrix.

    Args:
        X (CSRMatrix): Input CSR matrix holding expression counts
        chunk_lb (int): Lower bound of the vertial slice
        chunk_ub (int): Upper bound of the vertical slice
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data was log1p transformed or not.
        use_continuity (bool): Whether to use continuity correction when computing p-values.
        tie_correct (bool): Whether to apply tie correction when computing p-values.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Raises:
        ValueError: If bounds are not intelligible

    Returns:
        tuple[np.ndarray]: two-sided p-values, u-statistics and fold changes,
        each of shape (n_groups, chunk_lb - chunk_ub).

    Author: Rémy Dubois
    """
    _assert_is_csr(X)

    csc_chunk = csr_get_contig_cols_into_csc(csr_matrix=X, chunk_lb=chunk_lb, chunk_ub=chunk_ub)

    # TODO: same remark as csc regarding sorting
    pvalues, statistics = sparse_ovr_mwu_kernel(
        X=csc_chunk,
        groups=grpc.encoded_groups,
        group_counts=grpc.counts,
        use_continuity=use_continuity,
        tie_correct=tie_correct,
        alternative=alternative,
    )
    fold_change = csc_fold_change(X=csc_chunk, grpc=grpc, is_log1p=is_log1p)

    return pvalues, statistics, fold_change
