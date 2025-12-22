import numpy as np
from numba import njit
from typing import Literal

from illico.utils.groups import GroupContainer
from illico.utils.math import compute_pval, diff
from illico.utils.ranking import (
    _sort_csc_columns_inplace,
    rank_sum_and_ties_from_sorted,
)
from illico.utils.sparse.csc import csc_get_contig_cols_into_csr
from illico.utils.sparse.csr import (
    csr_fold_change,
    csr_get_contig_cols_into_csr,
    csr_get_rows_into_csc,
)
from illico.utils.type import CSCMatrix, CSRMatrix


@njit(nogil=True, fastmath=True, cache=False)
def single_group_sparse_ovo_mwu_kernel(
    sorted_ref_data: CSCMatrix,
    sorted_tgt_data: CSCMatrix,
    use_continuity: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Perform OVO tests gene wise using the two **sorted** CSC matrix given as input.

    The test performed is the equivalent of:
    `scipy.stats.mannwhitneyu(sorted_ref_data.toarray(), sorted_tgt_data.toarray(), use_continuity=True)`

    Args:
        sorted_ref_data (CSCMatrix): Reference data stored in CSC, sorted column-wise
        sorted_tgt_data (CSCMatrix): Perturbed data stored in CSC, sorted column-wise
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Raises:
        ValueError: If shape mismatche

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics. Each of shape (n_genes,).

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
    n = n_ref + n_tgt
    mu = n_ref * n_tgt / 2.0
    for j in range(n_cols_ref):
        n_zeros_combined = n_zeros_ref[j] + n_zeros_tgt[j]
        # Get the bounds
        lbt, ubt = sorted_tgt_data.indptr[j], sorted_tgt_data.indptr[j + 1]
        lbr, ubr = sorted_ref_data.indptr[j], sorted_ref_data.indptr[j + 1]

        # Compute ranksum and tie sum for non zero values
        ranksum, tie_sum = rank_sum_and_ties_from_sorted(sorted_ref_data.data[lbr:ubr], sorted_tgt_data.data[lbt:ubt])

        # Offset the ranks of the number of zeros in ref and perturbed
        ranksum += n_zeros_combined * (ubt - lbt)

        # Compute ranksum
        n0 = n_zeros_tgt[j]
        R1_nz = ranksum  # Sum ranks
        R1 = R1_nz + n0 * (n_zeros_ref[j] + n0 + 1) / 2.0  # Add sumranks of zeros

        # Compute U-stat
        U1 = n_ref * n_tgt + n_tgt * (n_tgt + 1) / 2 - R1

        # Compute sigma
        tie_sum += n_zeros_combined**3 - n_zeros_combined
        pvals[j] = compute_pval(
            n_ref=n_ref,
            n_tgt=n_tgt,
            n=n,
            tie_sum=tie_sum,
            U=U1,
            mu=mu,
            contin_corr=0.5 if use_continuity else 0.0,
            alternative=alternative,
        )

        # Regardless of the alternative, always return U1 like scipy
        U_statistics[j] = U1

    return pvals, U_statistics


@njit(nogil=True, fastmath=True, cache=False)
def multi_group_sparse_ovo_mwu_kernel(
    X: CSRMatrix,
    grpc: GroupContainer,
    ref_group_id: int,
    use_continuity: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Sequentially perform group-wise OVO tests along the columns on a CSR matrix.

    Args:
        X (CSRMatrix): Input CSR matrix of shape (n_cells, n_genes)
        grpc (GroupContainer): GroupContainer
        ref_group_id (int): Encoded reference group ID.
        use_continuity (bool, optional): Whether to use continuity correction when computing p-values. Defaults to True.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics. Each of shape (n_groups, n_genes).

    Author: Rémy Dubois
    """
    # Avoid accessing attributes in the for loop
    group_indices = grpc.indices
    group_indptr = grpc.indptr
    # Now, get the ref group, and convert it back to CSC because the test requires CSC data
    n_groups = group_indptr.size - 1
    ref_indices = group_indices[group_indptr[ref_group_id] : group_indptr[ref_group_id + 1]]
    csc_X_ref = csr_get_rows_into_csc(X, ref_indices)
    _sort_csc_columns_inplace(csc_matrix=csc_X_ref)

    # Now go through all the groups one by one
    pvalues = np.empty((n_groups, X.shape[1]), dtype=np.float64)
    statistics = np.empty((n_groups, X.shape[1]), dtype=np.float64)
    for group_id in range(group_indptr.size - 1):
        if group_id == ref_group_id:
            pvalues[group_id, :] = 1.0
            statistics[group_id, :] = -1.0
            continue

        tgt_idxs = group_indices[group_indptr[group_id] : group_indptr[group_id + 1]]
        X_tgt = csr_get_rows_into_csc(X, tgt_idxs)
        _sort_csc_columns_inplace(X_tgt)
        pvalue, statistic = single_group_sparse_ovo_mwu_kernel(
            sorted_ref_data=csc_X_ref, sorted_tgt_data=X_tgt, use_continuity=use_continuity, alternative=alternative
        )
        pvalues[group_id, :] = pvalue
        statistics[group_id, :] = statistic

    return pvalues, statistics


# Not jitting this and sorting all the cells at once is 1.5x slower. Ideally, it would be faster to sort only groups one by one but
# doubt this would be enough faster (think of mergesort) => It is twice faster, so i dont think it will bridge the gap
@njit(nogil=True, fastmath=True, cache=False)  # This requires too many caching, too dangerous
def csc_ovo_mwu_kernel_over_contiguous_col_chunk(
    X: CSCMatrix,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
):
    """Perform OVO test over contiguous chunk of column of a CSC matrix.

    Args:
        X (CSCMatrix): Input matrix of shape (n_cells, n_genes)
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data underwent log1p transform.

    Raises:
        ValueError: If chunk bounds are inintelligible.

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, fold change. Each
        of shape (n_groups, chunk_ub - chunk_lb).

    Author: Rémy Dubois
    """
    if chunk_lb < 0 or chunk_ub > X.shape[1] or chunk_lb > chunk_ub:
        raise ValueError((chunk_lb, chunk_ub))

    # This copies the data, so all that follow can happen in-place
    csr_chunk = csc_get_contig_cols_into_csr(X, chunk_lb, chunk_ub)
    pvalues, statistics = multi_group_sparse_ovo_mwu_kernel(
        X=csr_chunk,
        grpc=grpc,
        ref_group_id=grpc.encoded_ref_group,
        use_continuity=use_continuity,
        alternative=alternative,
    )

    # Compute fold change
    fold_change = csr_fold_change(csr_chunk, grpc, is_log1p=is_log1p)

    return pvalues, statistics, fold_change


# Real scale tests on whole H1 showed 24secs on 8 threads and 2min45s on 1, so a speedup of 165 / 24 = 6.875x
@njit(nogil=True, fastmath=True, cache=False)
def csr_ovo_mwu_kernel_over_contiguous_col_chunk(
    X: CSRMatrix,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
):
    """Perform OVO test over contiguous chunk of column of a CSR matrix.

    Args:
        X (CSRMatrix): Input matrix of shape (n_cells, n_genes)
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data underwent log1p transform.
        use_continuity (bool): Whether to use continuity correction when computing p-values.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Raises:
        ValueError: If chunk bounds are inintelligible.

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, fold change. Each
        of shape (n_groups, chunk_ub - chunk_lb).

    Author: Rémy Dubois
    """
    if chunk_lb < 0 or chunk_ub > X.shape[1] or chunk_lb > chunk_ub:
        raise ValueError((chunk_lb, chunk_ub))

    # This copies the data, so all that follow can happen in-place
    csr_chunk = csr_get_contig_cols_into_csr(X, chunk_lb, chunk_ub)
    pvalues, statistics = multi_group_sparse_ovo_mwu_kernel(
        X=csr_chunk,
        grpc=grpc,
        ref_group_id=grpc.encoded_ref_group,
        use_continuity=use_continuity,
        alternative=alternative,
    )
    # Compute fold change
    fold_change = csr_fold_change(csr_chunk, grpc, is_log1p=is_log1p)

    return pvalues, statistics, fold_change
