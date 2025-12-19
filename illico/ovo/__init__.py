import numpy as np
from scipy import sparse as py_sparse

from illico.ovo.dense_ovo import dense_ovo_mwu_kernel_over_contiguous_col_chunk
from illico.ovo.sparse_ovo import (
    csc_ovo_mwu_kernel_over_contiguous_col_chunk,
    csr_ovo_mwu_kernel_over_contiguous_col_chunk,
)
from illico.utils.groups import GroupContainer
from illico.utils.sparse.csc import CSCMatrix
from illico.utils.sparse.csr import CSRMatrix
from illico.utils.type import scipy_to_nb


def ovo_mwu_over_contiguous_col_chunk(
    X: np.ndarray | py_sparse.csc_matrix | py_sparse.csr_matrix,
    chunk_lb: int,
    chunk_ub: int,
    group_container: GroupContainer,
    is_log1p: bool,
):
    """Dispatcher for the OVO ranksum test. Routes toward the optimized
    implementation depending on input data format.

    Args:
        X (np.ndarray | py_sparse.csc_matrix | py_sparse.csr_matrix): Input expression count matrix.
        chunk_lb (int): Vertical slicing (column) lower bound
        chunk_ub (int): Vertical slicing (column) upper bound
        group_container (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if expression counts underwent log1p transfo.

    Raises:
        ValueError: If input is neither dense, csc or csr.

    Returns:
        tuple[np.ndarray]: Two-sided p-values, U-statistics and fold change.
        Each np.ndarray of shape (n_groups, n_genes)

    Author: Rémy Dubois
    """
    # Convert to numba friendly dtype
    X_nb = scipy_to_nb(X)

    # Check all possible input types
    if isinstance(X, np.ndarray):
        # TODO: could sort reference here
        pvalues, statistics, fold_change = dense_ovo_mwu_kernel_over_contiguous_col_chunk(
            X,
            chunk_lb,
            chunk_ub,
            group_container,
            is_log1p=is_log1p,
            use_continuity=True,
        )
    elif isinstance(X_nb, CSCMatrix):
        pvalues, statistics, fold_change = csc_ovo_mwu_kernel_over_contiguous_col_chunk(
            X_nb,
            chunk_lb,
            chunk_ub,
            group_container,
            is_log1p=is_log1p,
            use_continuity=True,
        )
    elif isinstance(X_nb, CSRMatrix):
        pvalues, statistics, fold_change = csr_ovo_mwu_kernel_over_contiguous_col_chunk(
            X_nb,
            chunk_lb,
            chunk_ub,
            group_container,
            is_log1p=is_log1p,
            use_continuity=True,
        )
    else:
        raise ValueError(X_nb)

    return pvalues, statistics, fold_change
