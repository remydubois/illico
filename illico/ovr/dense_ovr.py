"""Runs in 3.4 seconds for 100 genes, so roughly a 3.4 x 180 = 9 minutes for the whole H1."""

from typing import Literal

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import chunk_and_fortranize, compute_pval, dense_fold_change
from illico.utils.ranking import _accumulate_group_ranksums_from_argsort
from illico.utils.registry import KernelDataFormat, Test, dispatcher_registry


# TODO: check if njit this or not: on my mbp, it is 2 faster when not jitted
@dispatcher_registry.register(Test.OVR, KernelDataFormat.DENSE)
@njit(nogil=True, fastmath=True, cache=False)
def dense_ovr_mwu_kernel_over_contiguous_col_chunk(
    X: np.ndarray,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    tie_correct: bool = True,
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
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis. Defaults to "two-sided".

    Returns:
        tuple[np.ndarray]: Two-sided p-values, U-statistic and fold change.
        Each np.ndarray of shape (n_groups, n_genes)

    Author: Rémy Dubois
    """
    # Convert to F-order for faster column access and sorting later
    chunk = chunk_and_fortranize(X, chunk_lb, chunk_ub, None)

    # Get ranks and tie sums
    tie_sum = np.empty(chunk.shape[1], dtype=np.float64)
    ranksums = np.zeros(shape=(grpc.counts.size, chunk.shape[1]), dtype=np.float64)
    for j in range(chunk.shape[1]):
        idxs = np.argsort(chunk[:, j])
        col_tie_sum = _accumulate_group_ranksums_from_argsort(chunk[:, j], idxs, grpc.encoded_groups, ranksums[:, j])
        tie_sum[j] = col_tie_sum

    # Compute U stats
    n = chunk.shape[0]
    n_ref = np.expand_dims(n - grpc.counts, -1)  # (g, 1)
    n_tgt = np.expand_dims(grpc.counts, -1)  # (g, 1)
    statistics = (n_ref * n_tgt + n_tgt * (n_tgt + 1) / 2) - ranksums
    mu = n_ref * n_tgt / 2.0
    # Compute pvals
    pvals = np.empty(shape=(grpc.counts.size, chunk.shape[1]), dtype=np.float64)
    for j in range(chunk.shape[1]):
        for k in range(grpc.counts.size):
            pvals[k, j] = compute_pval(
                n_ref=n_ref[k, 0],
                n_tgt=n_tgt[k, 0],
                n=n,
                tie_sum=tie_sum[j] if tie_correct else 0.0,
                U=statistics[k, j],
                mu=mu[k, 0],
                contin_corr=0.5 if use_continuity else 0.0,
                alternative=alternative,
            )

    # Get fold change
    fold_change = dense_fold_change(chunk, grpc=grpc, is_log1p=is_log1p)

    return pvals, statistics, fold_change
