from typing import Literal

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import compute_pval, diff, fold_change_from_summed_expr
from illico.utils.ranking import (
    _sort_csc_columns_inplace,
    rank_sum_and_ties_from_binsearch,
    rank_sum_and_ties_from_sorted,
    unique_from_sorted,
)
from illico.utils.registry import KernelDataFormat, Test, nb_dispatcher_registry
from illico.utils.sparse.csc import (
    CSCMatrix,
    csc_get_contig_cols_into_csr,
    csc_sum_axis0,
)
from illico.utils.sparse.csr import (
    CSRMatrix,
    csr_get_rows_contig_cols_into_csc,
    csr_get_rows_into_csc,
)


@njit(nogil=True, fastmath=True, cache=False)
def single_group_sparse_ovo_mwu_kernel(
    sorted_ref_data: CSCMatrix,
    sorted_tgt_data: CSCMatrix,
    use_continuity: bool = True,
    tie_correct: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Perform OVO tests gene wise using the two **sorted** CSC matrix given as input.

    The test performed is the equivalent of:
    `scipy.stats.mannwhitneyu(sorted_ref_data.toarray(), sorted_tgt_data.toarray(), use_continuity=True)`

    Args:
        sorted_ref_data (CSCMatrix): Reference data stored in CSC, sorted column-wise
        sorted_tgt_data (CSCMatrix): Perturbed data stored in CSC, sorted column-wise
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Raises:
        ValueError: If shape mismatche

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, zscores. Each of shape (n_genes,).

    Author: Rémy Dubois

    """

    n_ref, n_cols_ref = sorted_ref_data.shape
    n_tgt, n_cols_tgt = sorted_tgt_data.shape
    # TODO: make this check everywhere or nowhere
    if n_cols_ref != n_cols_tgt:
        raise ValueError("Uneven number of columns between ref and perturbed.")

    # Allocate placeholders
    n_zeros_tgt = (n_tgt - diff(sorted_tgt_data.indptr)).astype(np.int64)
    n_zeros_ref = (n_ref - diff(sorted_ref_data.indptr)).astype(np.int64)
    U_statistics = np.empty(n_cols_ref, dtype=np.float64)
    pvals = np.empty(n_cols_ref, dtype=np.float64)
    zscores = np.empty(n_cols_ref, dtype=np.float64)
    n = n_ref + n_tgt
    mu = n_ref * n_tgt / 2.0
    for j in range(n_cols_ref):
        n_zeros_combined = n_zeros_ref[j] + n_zeros_tgt[j]
        # Get the bounds
        lbt, ubt = sorted_tgt_data.indptr[j], sorted_tgt_data.indptr[j + 1]
        lbr, ubr = sorted_ref_data.indptr[j], sorted_ref_data.indptr[j + 1]

        # Compute ranksum and tie sum for non zero values
        nz_ranksum, tie_sum, zpos = rank_sum_and_ties_from_sorted(
            sorted_ref_data.data[lbr:ubr], sorted_tgt_data.data[lbt:ubt], zero_values_offset=n_zeros_combined
        )

        # Compute ranksum
        n0 = n_zeros_tgt[j]
        z_ranksum = (zpos + (n_zeros_ref[j] + n0 + 1) / 2.0) * n0  # Add sumranks of zeros
        R1 = nz_ranksum + z_ranksum  # Add sumranks of zeros

        # Compute U-stat
        U1 = R1 - n_tgt * (n_tgt + 1) / 2

        # Compute sigma
        tie_sum += n_zeros_combined**3 - n_zeros_combined
        pvals[j], zscores[j] = compute_pval(
            n_ref=n_ref,
            n_tgt=n_tgt,
            n=n,
            tie_sum=tie_sum if tie_correct else 0.0,
            U=U1,
            mu=mu,
            contin_corr=0.5 if use_continuity else 0.0,
            alternative=alternative,
        )

        # Regardless of the alternative, always return U1 like scipy
        U_statistics[j] = U1

    return pvals, U_statistics, zscores


@njit(nogil=True, fastmath=True, cache=False)
def compute_sparse_unique_values_and_offsets(csc_mat: CSCMatrix):
    """Compute unique values, their counts, and cumulative offsets for each column in a CSC matrix.

    Args:
        csc_mat: Sparse matrix in CSC format.

    Returns:
        offsets: Cumulative count offsets for unique values per column (int64).
        n_uniques: Number of unique values per column (uint32).
        tie_sums: Sum of (count^3 - count) for ties per column (float64).

    """
    ncols = csc_mat.shape[1]
    # ctrl_offsets have to be 8 bytes otherwise **3 will cause overflow
    offsets = np.zeros(csc_mat.data.size + ncols, dtype=np.int64)
    tie_sums = np.empty(ncols, dtype=np.int64)
    n_uniques = np.empty(ncols, dtype=np.uint32)
    for j in range(ncols):
        # This slices into the original CSC
        org_lb, org_ub = csc_mat.indptr[j], csc_mat.indptr[j + 1]
        # This slices into the CSC with nrows+1 rows.
        # +1 because first position is always 0 (exclusive prefix sum)
        offsets_lb, offsets_ub = org_lb + j + 1, org_ub + j + 1
        _, _, n_uniques[j] = unique_from_sorted(
            x=csc_mat.data[org_lb:org_ub],
            uniques=csc_mat.data[org_lb:org_ub],  # placeholder for output
            counts=offsets[offsets_lb:offsets_ub],  # placeholder for output
        )
        # The tail of the offsets is just garbage zeros,
        # not filled because not enough unique values
        nz_offsets_ub = offsets_lb + n_uniques[j]
        this_col_counts = offsets[offsets_lb:nz_offsets_ub]
        # Increment the tie sum before we cumsum in-place to save memory
        tie_sums[j] = np.sum(this_col_counts**3 - this_col_counts)
        offsets[offsets_lb:nz_offsets_ub] = np.cumsum(offsets[offsets_lb:nz_offsets_ub])

    return offsets, n_uniques, tie_sums


@njit(nogil=True, fastmath=True, cache=False)
def _single_group_sparse_ovo_mwu_kernel(
    sorted_ref_data: CSCMatrix,
    sorted_tgt_data: CSCMatrix,
    n_zeros_ref: np.ndarray,
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
    """Perform Mann-Whitney U test between reference and target groups for sparse data.

    Computes p-values, z-scores, and U-statistics for each column using a binary search approach with proper handling of
    tied values and zero counts. Results stored in-place.

    """

    n_ref, n_cols = sorted_ref_data.shape
    n_tgt = sorted_tgt_data.shape[0]
    # Compute number of zeros per column in the target group
    n_zeros_tgt = (n_tgt - diff(sorted_tgt_data.indptr)).astype(np.int64)
    n = n_ref + n_tgt
    mu = n_ref * n_tgt / 2.0
    for j in range(n_cols):
        n_zeros_combined = n_zeros_ref[j] + n_zeros_tgt[j]
        # Compute indptr bounds
        lbr = sorted_ref_data.indptr[j]
        ubr = lbr + n_uniques_ref[j]
        lbt, ubt = sorted_tgt_data.indptr[j], sorted_tgt_data.indptr[j + 1]
        lboffsets = lbr + j
        uboffsets = lbr + j + 1 + n_uniques_ref[j]
        rs, ts, zpos = rank_sum_and_ties_from_binsearch(
            sorted_ref_data.data[lbr:ubr],
            n_uniques_ref[j],
            ref_block_offsets[lboffsets:uboffsets],
            sorted_tgt_data.data[lbt:ubt],
            zero_values_offset=n_zeros_combined,
        )
        ts += tie_sums_ref[j]  # Add tie sum from controls
        ts += float(n_zeros_combined) ** 3 - float(n_zeros_combined)  # Add tie sum from zeros

        # Compute ranksum
        n0 = n_zeros_tgt[j]
        z_ranksum = (zpos + (n_zeros_ref[j] + n0 + 1) / 2.0) * n0  # Add sumranks of zeros
        R1 = rs + z_ranksum  # Add sumranks of zeros

        # Compute U-stat
        U1 = R1 - n_tgt * (n_tgt + 1) / 2

        _pvalues[j], _zscores[j] = compute_pval(
            n_ref=n_ref,
            n_tgt=n_tgt,
            n=n,
            tie_sum=ts if tie_correct else 0.0,
            U=U1,
            mu=mu,
            contin_corr=0.5 if use_continuity else 0.0,
            alternative=alternative,
        )
        _statistics[j] = U1


@nb_dispatcher_registry.register(Test.OVO, KernelDataFormat.CSR)
@njit(nogil=True, fastmath=True, cache=False)
def csr_ovo_mwu_kernel_over_contiguous_col_chunk(
    X: CSRMatrix,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    tie_correct: bool = True,
    exp_post_agg: bool = False,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
):
    """Sequentially perform group-wise OVO tests along the columns on a CSR matrix.

    Also computes fold change based on the aggregated sums for each group, eventually exponentiating.
    The reason why this function is user facing is because I want to register it in the dispatcher
    registry.

    NB: this function is very similar to the one for CSC, but those two can't be unified as the slicing
    and chunking operations do no create the same intermediate variables, and Numba fails to compile.

    Compared to the CSC version, this is faster and ligher on RAM because the CSR format allows to easily
    gather scattered rows (each group is made of scattered cells) while the CSC format requires a CSR conversion
    first, in order to later gather each group's cells (rows).

    Args:
        X (CSRMatrix): Input CSR matrix of shape (n_cells, n_genes)
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data underwent log1p transform.
        use_continuity (bool, optional): Whether to use continuity correction when computing p-values. Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        exp_post_agg (bool, optional): Whether to exponentiate the fold change after aggregation.
            This is relevant if the input data is log1p. See documentation for details.
            Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and exponentiates post aggregation by
            default. Defaults to False.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, zscores, fold-change. Each of shape (n_groups, n_genes).

    Author: Rémy Dubois

    """
    group_indices = grpc.indices
    group_indptr = grpc.indptr
    ref_group_id = grpc.encoded_ref_group
    # Now, get the ref group, and convert it back to CSC because the test requires CSC data
    n_groups = group_indptr.size - 1
    ref_indices = group_indices[group_indptr[ref_group_id] : group_indptr[ref_group_id + 1]]
    # Slice
    csc_X_ref = csr_get_rows_contig_cols_into_csc(X, chunk_lb, chunk_ub, ref_indices)
    n_ref = csc_X_ref.shape[0]
    n_zeros_ref = (n_ref - diff(csc_X_ref.indptr)).astype(np.int64)
    # Sort
    _sort_csc_columns_inplace(csc_matrix=csc_X_ref)

    # Initalize aggregated matrix to compute fold change later on
    agg_counts = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)

    # Need to compute fold change right away as we modify the ref chunk inplace thereafter
    agg_counts[ref_group_id, :] = csc_sum_axis0(csc_X_ref, expm1=is_log1p & (not exp_post_agg))

    # Precompute per-column unique values, counts, offsets, and ctrl self-tie sums (reused across groups)
    # After this call, csc_X_ref.data no longer contains the sorted, but the sorted unique values only
    ctrl_offsets, ctrl_n_uniques, ctrl_tie_sums = compute_sparse_unique_values_and_offsets(csc_X_ref)

    # Now go through all the groups one by one
    pvalues = np.empty((n_groups, csc_X_ref.shape[1]), dtype=np.float64)
    zscores = np.empty((n_groups, csc_X_ref.shape[1]), dtype=np.float64)
    statistics = np.empty((n_groups, csc_X_ref.shape[1]), dtype=np.float64)
    for group_id in range(grpc.n_selected_groups):
        if group_id == ref_group_id:
            pvalues[group_id, :] = 1.0
            zscores[group_id, :] = 0.0
            statistics[group_id, :] = -1.0
            continue

        # Chunk
        tgt_idxs = group_indices[group_indptr[group_id] : group_indptr[group_id + 1]]
        csc_X_tgt = csr_get_rows_contig_cols_into_csc(X, chunk_lb, chunk_ub, tgt_idxs)
        # Sort
        _sort_csc_columns_inplace(csc_X_tgt)
        # Aggregate
        agg_counts[group_id, :] = csc_sum_axis0(csc_X_tgt, expm1=is_log1p & (not exp_post_agg))
        # Run mwu via binary search — O(n_tgt * log(n_uniques)) per col vs O(n_ref + n_tgt) merge
        _single_group_sparse_ovo_mwu_kernel(
            sorted_ref_data=csc_X_ref,
            sorted_tgt_data=csc_X_tgt,
            n_zeros_ref=n_zeros_ref,
            n_uniques_ref=ctrl_n_uniques,
            ref_block_offsets=ctrl_offsets,
            tie_sums_ref=ctrl_tie_sums,
            use_continuity=use_continuity,
            tie_correct=tie_correct,
            alternative=alternative,
            _pvalues=pvalues[group_id],  # Pass the row to be modified in-place
            _zscores=zscores[group_id],  # same
            _statistics=statistics[group_id],  # same
        )

    fold_change = fold_change_from_summed_expr(agg_counts, grpc, exp_post_agg=exp_post_agg & is_log1p)

    return pvalues, statistics, zscores, fold_change


@nb_dispatcher_registry.register(Test.OVO, KernelDataFormat.CSC)
@njit(nogil=True, fastmath=True, cache=False)  # This requires too many caching, too dangerous
def csc_ovo_mwu_kernel_over_contiguous_col_chunk(
    X: CSCMatrix,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    tie_correct: bool = True,
    exp_post_agg: bool = False,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
):
    """Sequentially perform group-wise OVO tests along the columns on a CSC matrix.

    Also computes fold change based on the aggregated sums for each group, eventually exponentiating.
    The reason why this function is user facing is because I want to register it in the dispatcher
    registry.

    NB: this function is very similar to the one for CSC, but those two can't be unified as the slicing
    and chunking operations do no create the same intermediate variables, and Numba fails to compile.

    Compared to the CSC version, this is faster and ligher on RAM because the CSC format allows to easily
    gather scattered rows (each group is made of scattered cells) while the CSC format requires a CSC conversion
    first, in order to later gather each group's cells (rows).

    Args:
        X (CSCMatrix): Input CSC matrix of shape (n_cells, n_genes)
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data underwent log1p transform.
        use_continuity (bool, optional): Whether to use continuity correction when computing p-values. Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        exp_post_agg (bool, optional): Whether to exponentiate the fold change after aggregation.
            This is relevant if the input data is log1p. See documentation for details.
            Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and exponentiates post aggregation by
            default. Defaults to False.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, zscores, fold-change. Each of shape (n_groups, n_genes).

    Author: Rémy Dubois

    """  # This is memory intensive, but there is no other choice for CSC-stored data,
    # as gathering scattered rows from CSC is very costly.
    csr_chunk = csc_get_contig_cols_into_csr(X, chunk_lb, chunk_ub)

    group_indices = grpc.indices
    group_indptr = grpc.indptr
    ref_group_id = grpc.encoded_ref_group
    # Now, get the ref group, and convert it back to CSC because the test requires CSC data
    n_groups = group_indptr.size - 1
    ref_indices = group_indices[group_indptr[ref_group_id] : group_indptr[ref_group_id + 1]]
    # Slice
    csc_X_ref = csr_get_rows_into_csc(csr_chunk, ref_indices)
    n_ref = csc_X_ref.shape[0]
    n_zeros_ref = (n_ref - diff(csc_X_ref.indptr)).astype(np.int64)
    # Sort
    _sort_csc_columns_inplace(csc_matrix=csc_X_ref)

    # Initalize aggregated matrix to compute fold change later on
    agg_counts = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)

    # Need to compute fold change right away as we modify the ref chunk inplace thereafter
    agg_counts[ref_group_id, :] = csc_sum_axis0(csc_X_ref, expm1=is_log1p & (not exp_post_agg))

    # Precompute per-column unique values, counts, offsets, and ctrl self-tie sums (reused across groups)
    # After this call, csc_X_ref.data no longer contains the sorted data, but the sorted unique values only
    ctrl_offsets, ctrl_n_uniques, ctrl_tie_sums = compute_sparse_unique_values_and_offsets(csc_X_ref)

    # Now go through all the groups one by one
    pvalues = np.empty((n_groups, csc_X_ref.shape[1]), dtype=np.float64)
    zscores = np.empty((n_groups, csc_X_ref.shape[1]), dtype=np.float64)
    statistics = np.empty((n_groups, csc_X_ref.shape[1]), dtype=np.float64)
    for group_id in range(grpc.n_selected_groups):
        if group_id == ref_group_id:
            pvalues[group_id, :] = 1.0
            zscores[group_id, :] = 0.0
            statistics[group_id, :] = -1.0
            continue

        # Chunk
        tgt_idxs = group_indices[group_indptr[group_id] : group_indptr[group_id + 1]]
        csc_X_tgt = csr_get_rows_into_csc(csr_chunk, tgt_idxs)
        # Sort
        _sort_csc_columns_inplace(csc_X_tgt)
        # Aggregate
        agg_counts[group_id, :] = csc_sum_axis0(csc_X_tgt, expm1=is_log1p & (not exp_post_agg))
        # Run mwu via binary search — O(n_tgt * log(n_uniques)) per col vs O(n_ref + n_tgt) merge
        _single_group_sparse_ovo_mwu_kernel(
            sorted_ref_data=csc_X_ref,
            sorted_tgt_data=csc_X_tgt,
            n_zeros_ref=n_zeros_ref,
            n_uniques_ref=ctrl_n_uniques,
            ref_block_offsets=ctrl_offsets,
            tie_sums_ref=ctrl_tie_sums,
            use_continuity=use_continuity,
            tie_correct=tie_correct,
            alternative=alternative,
            _pvalues=pvalues[group_id],  # Pass the row to be modified in-place
            _zscores=zscores[group_id],  # same
            _statistics=statistics[group_id],  # same
        )

    fold_change = fold_change_from_summed_expr(agg_counts, grpc, exp_post_agg=exp_post_agg & is_log1p)

    return pvalues, statistics, zscores, fold_change
