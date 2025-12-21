import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import chunk_and_fortranize, compute_pval, dense_fold_change
from illico.utils.ranking import (
    _sort_along_axis_inplace,
    rank_sum_and_ties_from_sorted,
)


@njit(nogil=True, fastmath=True, parallel=False, cache=False)
def dense_ovo_mwu_kernel(
    sorted_ref_data: np.ndarray, sorted_tgt_data: np.ndarray, use_continuity: bool = True
) -> tuple[np.ndarray]:
    """Sequentially perform OVO tests on columns between sorted ref and sorted perturbed data.

    Args:
        sorted_ref_data (np.ndarray): Vertically sorted reference data.
        sorted_tgt_data (_type_): Vertically sorted perturbed data.
        use_continuity (bool, optional): Apply continuity factor or not . Defaults to True.

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics. Each of shape (n_genes,).

    Author: Rémy Dubois
    """
    contin_corr = 0.5 if use_continuity else 0.0
    n_ref, ncols = sorted_ref_data.shape
    n_tgt, _ = sorted_tgt_data.shape
    
    U_statistics = np.empty(ncols, dtype=np.float64)
    pvals = np.empty(ncols, dtype=np.float64)
    n = n_ref + n_tgt
    mu = n_ref * n_tgt / 2.0
    for j in range(ncols):
        R1, tie_sum = rank_sum_and_ties_from_sorted(sorted_ref_data[:, j], sorted_tgt_data[:, j])

        # Compute U-stat
        U1 = n_ref * n_tgt + n_tgt * (n_tgt + 1) / 2 - R1

        pvals[j] = compute_pval(
            n_ref=n_ref,
            n_tgt=n_tgt,
            n=n,
            tie_sum=tie_sum,
            U=U1,
            mu=mu,
            contin_corr=contin_corr,
        )
        U_statistics[j] = U1

    return pvals, U_statistics


@njit(nogil=True, fastmath=True, cache=False, boundscheck=False)
def dense_ovo_mwu_kernel_over_contiguous_col_chunk(
    X: np.ndarray,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
) -> tuple[np.ndarray]:
    """Perform OVO tests group-wise and gene(col)-wise.

    Update: There is no need to fortranize the whole chunk at once, it can be done group by group within the loop. The memo

    Memory footprint investigations:
    1. ad.read_h5ad allocates memory in such a weird way that zeros are not properly assigned. Most likely this differs on unix systems. All investigations done on my MBP were so weird because of that.
    Tests were so weird because during the wilcoxon test, values were accessed and re-ordered and all those things. As a result, some memory allocation was happening within my functions although
    I was not allocating anything.
    2. np.asfortranarray(X[:, chunk_lb:chunk_ub][grpc.indices]) seems to allocate 2x the memory needed for the chunk. So if chunk is 2GB, this line allocates 4GB temporarily.
    3.


    Args:
        X (np.ndarray): Input dense expression matrix of shape (n_cells, n_genes)
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing
        grpc (GroupContainer): GroupContainer, contains information about which group each row belongs to.
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        is_log1p (bool, optional): User-indicated flag telling if data underwent log1p transform or not. Defaults to False.

    Raises:
        ValueError: If bounds are not intelligible.

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, fold change. Each
        of shape (n_groups, chunk_ub - chunk_lb).

    Author: Rémy Dubois
    """
    chunk = X[:, chunk_lb:chunk_ub]
    n_groups = grpc.counts.size

    ref_indices = grpc.indices[grpc.indptr[grpc.encoded_ref_group] : grpc.indptr[grpc.encoded_ref_group + 1]]
    # TODO: still have to benchmark speedup of F order
    ref_chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, ref_indices)
    _sort_along_axis_inplace(ref_chunk, axis=0)

    pvalues = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)
    statistics = np.empty((n_groups, chunk_ub - chunk_lb), dtype=np.float64)
    for group_id in range(n_groups):
        if group_id == grpc.encoded_ref_group:
            continue
        tgt_indices = grpc.indices[grpc.indptr[group_id] : grpc.indptr[group_id + 1]]
        # tgt_chunk = np.asfortranarray(chunk[tgt_indices, :])
        tgt_chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, tgt_indices)
        _sort_along_axis_inplace(tgt_chunk, axis=0)

        pvalues[group_id], statistics[group_id] = dense_ovo_mwu_kernel(
            sorted_ref_data=ref_chunk,
            sorted_tgt_data=tgt_chunk,
            use_continuity=use_continuity,
        )
    
    # Compute fold change
    fc = dense_fold_change(chunk, grpc, is_log1p=is_log1p)

    return pvalues, statistics, fc
