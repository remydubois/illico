from typing import Literal

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import chunk_and_fortranize, compute_pval, dense_fold_change
from illico.utils.ranking import (
    _sort_along_axis_inplace,
    rank_sum_and_ties_from_binsearch,
    unique_from_sorted,
)
from illico.utils.registry import KernelDataFormat, Test, nb_dispatcher_registry


@njit(nogil=True, fastmath=True, cache=False)
def _single_group_dense_ovo_mwu_kernel(
    sorted_ref_data: np.ndarray,
    sorted_tgt_data: np.ndarray,
    n_uniques_ref: np.ndarray,
    ref_block_offsets: np.ndarray,
    tie_sums_ref: np.ndarray,
    use_continuity: bool,
    tie_correct: bool,
    alternative: Literal["two-sided", "less", "greater"],
    _pvalues: np.ndarray,
    _zscores: np.ndarray,
    _statistics: np.ndarray,
) -> tuple[np.ndarray]:
    """Compute p-values, z-scores and statistics for a single group."""
    n_tgt, n_cols = sorted_tgt_data.shape
    n_ref = sorted_ref_data.shape[0]
    n = n_ref + n_tgt
    for j in range(n_cols):
        # Compute rank sum and tie sum using binary search
        rs, ts, _ = rank_sum_and_ties_from_binsearch(
            sorted_ref_data[:, j][: n_uniques_ref[j]],
            n_uniques_ref[j],
            ref_block_offsets[j, : n_uniques_ref[j] + 1],
            sorted_tgt_data[:, j],
            0,
        )
        ts += tie_sums_ref[j]  # Add the tie sum from the reference chunk

        tie_sum = ts
        mu = n_ref * n_tgt / 2.0
        U1 = rs - n_tgt * (n_tgt + 1) / 2.0

        _pvalues[j], _zscores[j] = compute_pval(
            n_ref=n_ref,
            n_tgt=n_tgt,
            n=n,
            tie_sum=tie_sum if tie_correct else 0.0,
            U=U1,
            mu=mu,
            contin_corr=0.5 if use_continuity else 0.0,
            alternative=alternative,
        )
        _statistics[j] = U1


@njit(nogil=True, fastmath=True, cache=False)
def compute_unique_values_and_offsets(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unique values, offsets, and tie sums for sorted data.

    Args:
        x (np.ndarray): Sorted input matrix of shape (n_rows, n_cols)

    Returns:
        np.ndarray: Tie block offsets per column
        np.ndarray: Number of unique values (blocks) per column
        np.ndarray: Tie sums for each column

    """
    nrows, ncols = x.shape
    offsets = np.zeros((ncols, nrows + 1), dtype=np.int64)  # row-major to slice eff
    tie_sums = np.zeros(ncols, dtype=np.int64)
    n_uniques = np.zeros(ncols, dtype=np.uint32)
    for j in range(ncols):
        # Store unique values in x itself
        _, _, n_uniques[j] = unique_from_sorted(x[:, j], uniques=x[:, j], counts=offsets[j, 1:])
        # Update the tie sums before we compute the prefix sum
        counts = offsets[j, 1 : n_uniques[j] + 1]
        tie_sums[j] = (counts**3 - counts).sum()
        # Now prefix sum
        offsets[j, 1 : n_uniques[j] + 1] = np.cumsum(offsets[j, 1 : n_uniques[j] + 1])

    return offsets, n_uniques, tie_sums


@nb_dispatcher_registry.register(Test.OVO, KernelDataFormat.DENSE)
@njit(nogil=True, fastmath=True, cache=False, boundscheck=False)
def dense_ovo_mwu_kernel_over_contiguous_col_chunk(
    X: np.ndarray,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    tie_correct: bool = True,
    exp_post_agg: bool = False,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Perform OVO tests group-wise and gene(col)-wise.

    Update: There is no need to fortranize the whole chunk at once, it can be done group by group within the loop.

    Memory footprint investigations:
    1. Because fortranizing has to happen per group


    Args:
        X (np.ndarray): Input dense expression matrix of shape (n_cells, n_genes)
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        grpc (GroupContainer): GroupContainer, contains information about which group each row belongs to.
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        exp_post_agg (bool, optional): Whether to exponentiate the fold change after aggregation. This is relevant if the input data is log1p. See documentation for details. Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and exponentiates post aggregation by default. Defaults to False.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis
        is_log1p (bool, optional): User-indicated flag telling if data underwent log1p transform or not. Defaults to False.

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, z-scores, fold change. Each
        of shape (n_groups, chunk_ub - chunk_lb).

    Author: Rémy Dubois

    """
    chunk = X[:, chunk_lb:chunk_ub]
    n_groups = grpc.counts.size

    ref_indices = grpc.indices[grpc.indptr[grpc.encoded_ref_group] : grpc.indptr[grpc.encoded_ref_group + 1]]
    # TODO: still have to benchmark speedup of F order
    ref_chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, ref_indices)
    _sort_along_axis_inplace(ref_chunk, axis=0)

    # Pre allocate output
    pvalues = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)
    zscores = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)
    statistics = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)

    # Compute unique values, counts and offsets for the reference chunk
    ctrl_offsets, ctrl_n_uniques, ctrl_tiesums = compute_unique_values_and_offsets(ref_chunk)

    # Go through all groups and compute output
    for group_id in range(grpc.n_selected_groups):
        if group_id == grpc.encoded_ref_group:
            pvalues[group_id, :] = 1.0
            zscores[group_id, :] = 0.0
            statistics[group_id, :] = -1.0
            continue
        tgt_indices = grpc.indices[grpc.indptr[group_id] : grpc.indptr[group_id + 1]]
        tgt_chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, tgt_indices)
        _sort_along_axis_inplace(tgt_chunk, axis=0)

        # Compute p-values, z-scores and statistics for this group
        _single_group_dense_ovo_mwu_kernel(
            sorted_ref_data=ref_chunk,
            sorted_tgt_data=tgt_chunk,
            n_uniques_ref=ctrl_n_uniques,
            ref_block_offsets=ctrl_offsets,
            tie_sums_ref=ctrl_tiesums,
            use_continuity=use_continuity,
            tie_correct=tie_correct,
            alternative=alternative,
            _pvalues=pvalues[group_id, :],
            _zscores=zscores[group_id, :],
            _statistics=statistics[group_id, :],
        )

    # Compute fold change
    fc = dense_fold_change(chunk, grpc, is_log1p=is_log1p, exp_post_agg=exp_post_agg)

    return pvalues, statistics, zscores, fc
