"""Runs in 3.4 seconds for 100 genes, so roughly a 3.4 x 180 = 9 minutes for the whole H1."""

from typing import Literal

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import (
    _add_at_vec,
    chunk_and_fortranize,
    compute_pval,
    fancy_indexing_axis0,
    fold_change_from_summed_expr,
)
from illico.utils.ranking import _accumulate_group_ranksums_from_argsort
from illico.utils.registry import KernelDataFormat, Test, nb_dispatcher_registry


# TODO: check if njit this or not: on my mbp, it is 2 faster when not jitted
@nb_dispatcher_registry.register(Test.OVR, KernelDataFormat.DENSE)
@njit(nogil=True, fastmath=True, cache=False)
def dense_ovr_mwu_kernel_over_contiguous_col_chunk(
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
    """Compute OVR ranksum test on a dense matrix of expression counts.

    Args:
        X (np.ndarray): Input dense raw counts matrix
        grpc (GroupContainer): GroupContainer
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        is_log1p (bool, optional): User-indicated flag telling if data underwent log1p
        transformation or not. Defaults to False.
        use_continuity (bool, optional): Whether to use continuity correction when computing p-values. Defaults to True.
        tie_correct (bool, optional): Whether to apply tie correction when computing p-values. Defaults to True.
        exp_post_agg (bool, optional): Whether to exponentiate the fold change after aggregation. This is relevant if the input data is log1p. See documentation for details. Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and exponentiates post aggregation by default. Defaults to False.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis. Defaults to "two-sided".

    Returns:
        tuple[np.ndarray]: Two-sided p-values, U-statistic, z-scores and fold change.
        Each np.ndarray of shape (n_groups, n_genes)

    Author: Rémy Dubois

    """
    # Convert to F-order for faster column access and sorting later
    chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, grpc.ovr_inclusion_indices)

    # Get ranks and tie sums
    tie_sum = np.empty(chunk.shape[1], dtype=np.float64)
    ranksums = np.zeros(shape=(grpc.counts.size, chunk.shape[1]), dtype=np.float64)
    included_groups_indicator = grpc.encoded_groups[grpc.ovr_inclusion_indices]
    for j in range(chunk.shape[1]):
        idxs = np.argsort(chunk[:, j])
        col_tie_sum, _ = _accumulate_group_ranksums_from_argsort(
            chunk[:, j], idxs, included_groups_indicator, ranksums[:, j]
        )
        tie_sum[j] = col_tie_sum

    # Compute U stats
    n = chunk.shape[0]
    n_ref = np.expand_dims(n - grpc.counts, -1)  # (g, 1)
    n_tgt = np.expand_dims(grpc.counts, -1)  # (g, 1)
    statistics = ranksums - n_tgt * (n_tgt + 1) / 2
    mu = n_ref * n_tgt / 2.0
    # Compute pvals
    n_selected_groups = grpc.selected_group_ids.size
    pvals = np.empty(shape=(n_selected_groups, chunk.shape[1]), dtype=np.float64)
    zscores = np.empty(shape=(n_selected_groups, chunk.shape[1]), dtype=np.float64)
    for j in range(chunk.shape[1]):
        for k, grp_id in enumerate(grpc.selected_group_ids):
            pvals[k, j], zscores[k, j] = compute_pval(
                n_ref=n_ref[grp_id, 0],
                n_tgt=n_tgt[grp_id, 0],
                n=n,
                tie_sum=tie_sum[j] if tie_correct else 0.0,
                U=statistics[grp_id, j],
                mu=mu[grp_id, 0],
                contin_corr=0.5 if use_continuity else 0.0,
                alternative=alternative,
            )

    # Get fold change
    # Note: it would be a bit cumbersome to have dense_fold_change handle itself all the shennanigans
    # groups and subsetting. I find clearer to have it here.
    # TODO: actually idk, bc I ended up doing it in the sparse path.
    group_agg_counts = np.zeros(shape=(grpc.counts.size, X.shape[1]), dtype=np.float64)
    # Sum expressions per group
    if is_log1p and not exp_post_agg:
        _add_at_vec(group_agg_counts, grpc.encoded_groups[grpc.ovr_inclusion_indices], np.expm1(chunk))
    else:
        _add_at_vec(group_agg_counts, grpc.encoded_groups[grpc.ovr_inclusion_indices], chunk)
    fold_change = fold_change_from_summed_expr(
        group_agg_counts, grpc, exp_post_agg=exp_post_agg & is_log1p, sum_over_selected_groups_only=True
    )

    # Now filter on the groups to return, if needed
    if n_selected_groups < grpc.counts.size:
        fold_change = fancy_indexing_axis0(fold_change, grpc.selected_group_ids)
        statistics = fancy_indexing_axis0(statistics, grpc.selected_group_ids)

    return pvals, statistics, zscores, fold_change
