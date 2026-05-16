from typing import Literal

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import (
    chunk_and_fortranize,
    compute_pval,
    dense_fold_change,
    fancy_indexing_axis0,
)
from illico.utils.ranking import (
    _sort_along_axis_inplace,
    rank_sum_and_ties_from_sorted,
)
from illico.utils.registry import KernelDataFormat, Test, nb_dispatcher_registry


@njit(nogil=True, fastmath=True, parallel=False, cache=False)
def dense_ovo_mwu_kernel(
    sorted_ref_data: np.ndarray,
    sorted_tgt_data: np.ndarray,
    use_continuity: bool = True,
    tie_correct: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Sequentially perform OVO tests on columns between sorted ref and sorted perturbed data.

    Args:
        sorted_ref_data (np.ndarray): Vertically sorted reference data.
        sorted_tgt_data (_type_): Vertically sorted perturbed data.
        use_continuity (bool, optional): Apply continuity factor or not . Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis

    Returns:
        tuple[np.ndarray]: two-sided p-values, U-statistics, z-scores. Each of shape (n_genes,).

    Author: Rémy Dubois

    """
    n_ref, ncols = sorted_ref_data.shape
    n_tgt, _ = sorted_tgt_data.shape

    U_statistics = np.empty(ncols, dtype=np.float64)
    pvals = np.empty(ncols, dtype=np.float64)
    zscores = np.empty(ncols, dtype=np.float64)
    n = n_ref + n_tgt
    mu = n_ref * n_tgt / 2.0
    for j in range(ncols):
        ranksum, tie_sum, _ = rank_sum_and_ties_from_sorted(sorted_ref_data[:, j], sorted_tgt_data[:, j])

        # Compute U-stat
        U1 = ranksum - n_tgt * (n_tgt + 1) / 2.0

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
        U_statistics[j] = U1

    return pvals, U_statistics, zscores


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
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        exp_post_agg (bool, optional): Whether to exponentiate the fold change after aggregation. This is relevant if the input data is log1p. See documentation for details. Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and exponentiates post aggregation by default. Defaults to False.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis
        is_log1p (bool, optional): User-indicated flag telling if data underwent log1p transform or not. Defaults to False.

    Raises:
        ValueError: If bounds are not intelligible.

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

    n_selected_groups = grpc.selected_group_ids.size
    pvalues = np.empty((n_selected_groups, chunk_ub - chunk_lb), dtype=np.float64)
    zscores = np.empty((n_selected_groups, chunk_ub - chunk_lb), dtype=np.float64)
    statistics = np.empty((n_selected_groups, chunk_ub - chunk_lb), dtype=np.float64)
    for k, group_id in enumerate(grpc.selected_group_ids):
        if group_id == grpc.encoded_ref_group:
            pvalues[k, :] = 1.0
            zscores[k, :] = 0.0
            statistics[k, :] = -1.0
            continue
        tgt_indices = grpc.indices[grpc.indptr[group_id] : grpc.indptr[group_id + 1]]
        # tgt_chunk = np.asfortranarray(chunk[tgt_indices, :])
        tgt_chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, tgt_indices)
        _sort_along_axis_inplace(tgt_chunk, axis=0)

        pvalues[k], statistics[k], zscores[k] = dense_ovo_mwu_kernel(
            sorted_ref_data=ref_chunk,
            sorted_tgt_data=tgt_chunk,
            use_continuity=use_continuity,
            tie_correct=tie_correct,
            alternative=alternative,
        )

    # Compute fold change on all groups, but return it only for the selected groups
    fc = dense_fold_change(chunk, grpc, is_log1p=is_log1p, exp_post_agg=exp_post_agg)
    if n_selected_groups < n_groups:
        fc = fancy_indexing_axis0(fc, grpc.selected_group_ids)

    return pvalues, statistics, zscores, fc
