from typing import Literal

import numpy as np
from scipy import sparse as py_sparse

from illico.ovr.dense_ovr import dense_ovr_mwu_kernel_over_contiguous_col_chunk
from illico.ovr.sparse_ovr import (
    csc_ovr_mwu_kernel_over_contiguous_col_chunk,
    csr_ovr_mwu_kernel_over_contiguous_col_chunk,
)
from illico.utils.groups import GroupContainer
from illico.utils.sparse.csc import CSCMatrix
from illico.utils.sparse.csr import CSRMatrix
from illico.utils.type import scipy_to_nb


def ovr_mwu_over_col_contiguous_chunk(
    X: np.ndarray | py_sparse.csc_matrix | py_sparse.csr_matrix,
    chunk_lb: int,
    chunk_ub: int,
    group_container: GroupContainer,
    is_log1p: bool,
    use_continuity: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[np.ndarray]:
    """Dispatcher for the OVR ranksum test. Routes toward the optimized
    implementation depending on input data format.

    Args:
        X (np.ndarray | py_sparse.csc_matrix | py_sparse.csr_matrix): Input expression count matrix.
        chunk_lb (int): Vertical slicing (column) lower bound
        chunk_ub (int): Vertical slicing (column) upper bound
        group_container (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if expression counts underwent log1p transfo.
        use_continuity (bool): Whether to use continuity correction when computing p-values.
        alternative (Literal["two-sided", "less", "greater"]): Type of alternative hypothesis.

    Raises:
        ValueError: If input is neither dense, csc or csr.

    Returns:
        tuple[np.ndarray]: Two-sided p-values, U-statistics and fold change.
        Each np.ndarray of shape (n_groups, n_genes)

    Author: Rémy Dubois
    """

    group_container = GroupContainer(*group_container)

    # This one can not be jitted due to types branching but thats fine the work done here is minimal
    # Convert to numba friendly dtype
    X_nb = scipy_to_nb(X)

    if isinstance(X_nb, np.ndarray):
        pvalues, statistics, fold_change = dense_ovr_mwu_kernel_over_contiguous_col_chunk(
            X=X_nb,
            chunk_lb=chunk_lb,
            chunk_ub=chunk_ub,
            grpc=group_container,
            is_log1p=is_log1p,
            use_continuity=use_continuity,
            alternative=alternative,
        )
    elif isinstance(X_nb, CSCMatrix):
        pvalues, statistics, fold_change = csc_ovr_mwu_kernel_over_contiguous_col_chunk(
            X=X_nb,
            chunk_lb=chunk_lb,
            chunk_ub=chunk_ub,
            grpc=group_container,
            is_log1p=is_log1p,
            use_continuity=use_continuity,
            alternative=alternative,
        )
    elif isinstance(X_nb, CSRMatrix):
        pvalues, statistics, fold_change = csr_ovr_mwu_kernel_over_contiguous_col_chunk(
            X=X_nb,
            chunk_lb=chunk_lb,
            chunk_ub=chunk_ub,
            grpc=group_container,
            is_log1p=is_log1p,
            use_continuity=use_continuity,
            alternative=alternative,
        )
    else:
        raise ValueError(type(X_nb))

    return pvalues, statistics, fold_change
