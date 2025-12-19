"""Runs in 3.4 seconds for 100 genes, so roughly a 3.4 x 180 = 9 minutes for the whole H1."""

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import chunk_and_fortranize, compute_pval, dense_fold_change
from illico.utils.ranking import _accumulate_group_ranksums_from_argsort


# TODO: check if njit this or not: on my mbp, it is 2 faster when not jitted
@njit(nogil=True, fastmath=True, cache=False)
def dense_ovr_mwu_kernel_over_contiguous_col_chunk(
    X: np.ndarray,
    chunk_lb: int,
    chunk_ub: int,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
) -> tuple[np.ndarray]:
    """Compute OVR ranksum test on a dense matrix of expression counts.

    Args:
        X (np.ndarray): Input dense raw counts matrix
        grpc (GroupContainer): GroupContainer
        use_continuity (bool, optional): Apply continuity factor or not. Defaults to True.
        is_log1p (bool, optional): User-indicated flag telling if data underwent log1p
        transformation or not. Defaults to False.

    Returns:
        tuple[np.ndarray]: Two-sided p-values, U-statistic and fold change.
        Each np.ndarray of shape (n_groups, n_genes)

    Author: Rémy Dubois
    """
    # TODO: handle tie correction properly
    contin_corr = 0.5 if use_continuity else 0.0
    if chunk_lb < 0 or chunk_ub > X.shape[1] or chunk_lb > chunk_ub:
        raise ValueError((chunk_lb, chunk_ub))

    # TODO: check if converting to fortran helps here
    # This comes at the cost of copying the array but it is more than twice faster so worth it. Maybe make the batch size smaller.
    # chunk = X[:, chunk_lb:chunk_ub]
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
    U = (n_ref * n_tgt + n_tgt * (n_tgt + 1) / 2) - ranksums
    mu = n_ref * n_tgt / 2.0
    # Compute pvals
    # TODO: if the not jitted, maybe this double loop can be shelled inside a njit function
    pvals = np.empty(shape=(grpc.counts.size, chunk.shape[1]), dtype=np.float64)
    for j in range(chunk.shape[1]):
        for k in range(grpc.counts.size):
            pvals[k, j] = compute_pval(
                n_ref=n_ref[k, 0],
                n_tgt=n_tgt[k, 0],
                n=n,
                tie_sum=tie_sum[j],
                U=U[k, j],
                mu=mu[k, 0],
                contin_corr=contin_corr,
            )

    # Get fold change
    fold_change = dense_fold_change(chunk, grpc=grpc, is_log1p=is_log1p)

    # TODO: unify U names across funcs
    return pvals, U, fold_change
