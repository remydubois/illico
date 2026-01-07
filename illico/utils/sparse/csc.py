from collections import namedtuple
from typing import Any

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import _add_at_scalar, _add_at_vec, fold_change_from_summed_expr

CSCMatrix = namedtuple("CSCMatrix", ["data", "indices", "indptr", "shape"])
CSRMatrix = namedtuple("CSRMatrix", ["data", "indices", "indptr", "shape"])


@njit(nogil=True, cache=False)
def _assert_is_csc(matrix: Any) -> None:
    """Assert a matrix is CSC.

    Args:
        obj Any: The matrix to check.

    Author: Rémy Dubois
    """
    try:
        n_parcels = matrix.indptr.size - 1
        _, n_cols = matrix.shape
    except:  # noqa: E722
        raise AssertionError("Matric is not CSC: missing attributes.")
    assert n_parcels == n_cols, "Matric is not CSC: indptr and number of columns don't match."


@njit(nogil=True, fastmath=True, cache=False)
def csc_count_nonzeros(csc_matrix: CSCMatrix, axis: int | None = None) -> np.ndarray:
    """Cout number of non zero values in a CSC matrix.

    Args:
        csc_matrix (CSCMatrix): Input matrix
        axis (int | None, optional): Axis along which to compute. Defaults to None.

    Raises:
        ValueError: If axis is not intelligible.

    Returns:
        np.ndarray: Number of non-zero values, always a 1-d array for compilation purposes.

    Author: Rémy Dubois
    """
    if axis is None:
        nnz = np.empty((1,), dtype=np.int32)
        nnz[0] = csc_matrix.data.size
        return nnz
    elif axis == 1:
        nnz = np.zeros(csc_matrix.shape[0], dtype=np.int32)
        _add_at_scalar(nnz, csc_matrix.indices, 1)
        return nnz
    elif axis == 0:
        # Idk why, np.diff crashes
        nnz = np.zeros(csc_matrix.shape[1], dtype=np.int32)
        for i in range(1, csc_matrix.indptr.size):
            nnz[i - 1] = csc_matrix.indptr[i] - csc_matrix.indptr[i - 1]
        return nnz
    else:
        raise ValueError(axis)


@njit(nogil=True, fastmath=True, cache=False)
def csc_to_csr(csc_matrix: CSCMatrix) -> CSRMatrix:
    """Convert a CSC matrix to CSR.

    Args:
        csc_matrix (CSCMatrix): Input CSC matrix

    Returns:
        CSRMatrix: The resulting CSR matrix.

    Author: Rémy Dubois
    """
    nnz = csc_matrix.data.size
    csr_indptr = np.zeros(csc_matrix.shape[0] + 1, dtype=csc_matrix.indptr.dtype)
    csr_indices = np.empty(nnz, dtype=csc_matrix.indices.dtype)
    csr_data = np.empty(nnz, dtype=csc_matrix.data.dtype)

    # Pass 1: count number of entries per row
    csr_indptr[1:] = np.cumsum(csc_count_nonzeros(csc_matrix, 1))
    next_row = csr_indptr.copy()

    # Pass 2: fill CSC structure
    for col in range(csc_matrix.shape[1]):
        col_start = csc_matrix.indptr[col]
        col_end = csc_matrix.indptr[col + 1]
        for idx in range(col_start, col_end):
            row = csc_matrix.indices[idx]
            dest = next_row[row]
            csr_indices[dest] = col
            csr_data[dest] = csc_matrix.data[idx]
            next_row[row] += 1
    return CSRMatrix(csr_data, csr_indices, csr_indptr, csc_matrix.shape)


@njit(nogil=True, fastmath=True, cache=False)
def csc_get_cols(csc_matrix: CSCMatrix, indices: np.ndarray) -> CSCMatrix:
    """Perform vertical slicing of a CSC matrix.

    Equivalent of `scipy.sparse.csc_matrix[:, indices]`.

    Args:
        csc_matrix (CSCMatrix): Input CSC matrix
        indices (np.ndarray): Indices of the columns to get.

    Raises:
        ValueError: If indices are not intelligible

    Returns:
        CSCMatrix: The resulting (sliced) CSC matrix

    Author: Rémy Dubois
    """
    if indices.min() < 0 or indices.max() > csc_matrix.shape[0]:
        raise ValueError(indices.min(), indices.max())
    new_indptr = np.empty(indices.size + 1, dtype=csc_matrix.indptr.dtype)
    new_indptr[0] = 0
    for i in range(indices.size):
        col_idx = indices[i]
        col_size = csc_matrix.indptr[col_idx + 1] - csc_matrix.indptr[col_idx]
        new_indptr[i + 1] = col_size + new_indptr[i]

    new_data = np.empty(new_indptr[-1], csc_matrix.data.dtype)
    new_indices = np.empty(new_indptr[-1], csc_matrix.indices.dtype)
    for i in range(indices.size):
        col_idx = indices[i]
        new_data[new_indptr[i] : new_indptr[i + 1]] = csc_matrix.data[
            csc_matrix.indptr[col_idx] : csc_matrix.indptr[col_idx + 1]
        ]
        new_indices[new_indptr[i] : new_indptr[i + 1]] = csc_matrix.indices[
            csc_matrix.indptr[col_idx] : csc_matrix.indptr[col_idx + 1]
        ]
    return CSCMatrix(new_data, new_indices, new_indptr, (csc_matrix.shape[0], indices.size))


@njit(nogil=True, fastmath=True, cache=False)
def csc_get_contig_cols_into_csr(csc_matrix: CSCMatrix, chunk_lb: int, chunk_ub: int) -> CSRMatrix:
    """Perform contiguous vertical slicing of a CSC matrix and stores it into a CSR matrix.

    This function is the equivalent of `scipy.sparse.csc_matrix[:, chunk_lb:chunk_ub].to_csr()`.

    Args:
        csc_matrix (CSCMatrix): Input CSC matrix
        chunk_lb (int): Lower bound of the vertical slicing
        chunk_ub (int): Upper bound of the vertical slicing

    Raises:
        ValueError: If bounds are not intelligible.

    Returns:
        CSRMatrix: Resulting CSR matrix

    Author: Rémy Dubois
    """
    if chunk_lb < 0 or chunk_ub > csc_matrix.shape[1] or chunk_lb > chunk_ub:
        raise ValueError((chunk_lb, chunk_ub))

    csr_nnz = np.zeros(csc_matrix.shape[0], dtype=np.int64)
    _add_at_scalar(
        csr_nnz,
        csc_matrix.indices[csc_matrix.indptr[chunk_lb] : csc_matrix.indptr[chunk_ub]],
        1,
    )
    # Of shape nrows+1, with a leading 0
    csr_indptr = np.cumsum(np.roll(np.append(csr_nnz, 0), 1))

    # Allocate placeholders for data and indices
    new_data = np.empty(csr_indptr[-1], csc_matrix.data.dtype)
    new_indices = np.empty(csr_indptr[-1], csc_matrix.indices.dtype)
    pointer = csr_indptr.copy()  # Keep one pointer per row, indicating where to store the next value
    for j in range(chunk_lb, chunk_ub):
        col_start = csc_matrix.indptr[j]
        col_end = csc_matrix.indptr[j + 1]
        for idx in range(col_start, col_end):
            row_idx = csc_matrix.indices[idx]
            v = csc_matrix.data[idx]
            new_data[pointer[row_idx]] = v
            new_indices[pointer[row_idx]] = j - chunk_lb
            pointer[row_idx] += 1
    return CSRMatrix(new_data, new_indices, csr_indptr, (csc_matrix.shape[0], chunk_ub - chunk_lb))


@njit(nogil=True, fastmath=True, cache=False)
def csc_fold_change(X: CSCMatrix, grpc: GroupContainer, is_log1p: bool) -> np.ndarray:
    """Compute fold change from a CSC matrix of expression counts.

    Args:
        X (CSCMatrix): Input expression counts CSC matrix
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data was log1p or not.

    Returns:
        np.ndarray: Fold change of change (n_groups, n_genes)

    Author: Rémy Dubois
    """
    _assert_is_csc(X)
    group_agg_counts = np.zeros(shape=(grpc.counts.size, X.shape[1]), dtype=np.float64)
    # Sum expressions per group
    for j in range(X.shape[1]):
        start = X.indptr[j]
        end = X.indptr[j + 1]
        row_indices = X.indices[start:end]
        row_data = np.expm1(X.data[start:end]) if is_log1p else X.data[start:end]
        group_id = grpc.encoded_groups[row_indices]
        _add_at_vec(group_agg_counts[:, j], group_id, row_data)
    fold_change = fold_change_from_summed_expr(group_agg_counts, grpc)
    return fold_change
