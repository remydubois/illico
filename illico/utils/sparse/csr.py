from collections import namedtuple
from typing import Any

import numpy as np
from numba import njit

from illico.utils.groups import GroupContainer
from illico.utils.math import (
    _add_at_scalar,
    _add_at_vec,
    diff,
    fold_change_from_summed_expr,
)

CSCMatrix = namedtuple("CSCMatrix", ["data", "indices", "indptr", "shape"])
CSRMatrix = namedtuple("CSRMatrix", ["data", "indices", "indptr", "shape"])


@njit(nogil=True, cache=False)
def _assert_is_csr(matrix: Any) -> None:
    """Assert a matrix is CSR.

    Args:
        matrix: Input matrix to check.

    Author: Rémy Dubois
    """
    try:
        n_parcels = matrix.indptr.size - 1
        n_rows, _ = matrix.shape
    except:  # noqa: E722
        raise AssertionError("Object is not CSR: missing attributes.")
    assert n_parcels == n_rows, "Object is not CSR: indptr and number of rows don't match."


@njit(nogil=True, fastmath=True, cache=False)
def csr_count_nonzeros(csr_matrix: CSRMatrix, axis: int | None = None) -> np.ndarray:
    """Count number of non zeros in a CSR matrix.

    Args:
        csr_matrix (CSRMatrix): Input matrix
        axis (int | None, optional): Axis along which to compute. Defaults to None.

    Raises:
        ValueError: If axis is not intelligible.

    Returns:
        np.ndarray: Number of non-zero values, always a 1-d array for compilation purposes.

    Author: Rémy Dubois
    """
    if axis is None:
        nnz = np.empty((1,), dtype=np.int32)
        nnz[0] = csr_matrix.data.size
        return nnz
    elif axis == 0:
        nnz = np.zeros(csr_matrix.shape[1], dtype=np.int32)
        _add_at_scalar(nnz, csr_matrix.indices, 1)
        return nnz
    elif axis == 1:
        nnz = diff(csr_matrix.indptr)
        return nnz
    else:
        raise ValueError(axis)


@njit(nogil=True, fastmath=True, cache=False)
def csr_to_csc(csr_matrix: CSRMatrix) -> CSCMatrix:
    """Convert a CSR matrix to CSC.

    Args:
        csr_matrix (CSRMatrix): Input CSR matrix

    Returns:
        CSCMatrix: The resulting CSC matrix.

    Author: Rémy Dubois
    """
    nnz = csr_matrix.data.size

    # Allocate placeholders
    csc_indptr = np.zeros(csr_matrix.shape[1] + 1, dtype=csr_matrix.indptr.dtype)
    csc_indices = np.empty(nnz, dtype=csr_matrix.indices.dtype)
    csc_data = np.empty(nnz, dtype=csr_matrix.data.dtype)

    # Pass 1: count number of entries per column
    csc_indptr[1:] = np.cumsum(csr_count_nonzeros(csr_matrix, 0))
    next_col = csc_indptr.copy()

    # Pass 2: fill CSC structure
    for row in range(csr_matrix.shape[0]):
        row_start = csr_matrix.indptr[row]
        row_end = csr_matrix.indptr[row + 1]
        for idx in range(row_start, row_end):
            col = csr_matrix.indices[idx]
            dest = next_col[col]
            csc_indices[dest] = row
            csc_data[dest] = csr_matrix.data[idx]
            next_col[col] += 1
    return CSCMatrix(csc_data, csc_indices, csc_indptr, csr_matrix.shape)


@njit(nogil=True, fastmath=True, cache=False)
def csr_get_rows_into_csc(csr_matrix: CSRMatrix, indices: np.ndarray) -> CSCMatrix:
    """Performs horizontal slicing of a CSR matrix and stores it in a CSC matrix.

    Equivalent of `scipy.sparse.csr_matrix[indices, :].to_csc()`.

    Args:
        csr_matrix (CSRMatrix): Input CSR matrix
        indices (np.ndarray): Indices of the rows to get.

    Returns:
        CSCMatrix: The resulting CSC matrix holding data of the sliced rows.

    Author: Rémy Dubois
    """
    # Count non zeros per col
    csc_nnz = np.zeros(csr_matrix.shape[1], dtype=np.int64)
    for row_idx in indices:
        _add_at_scalar(
            csc_nnz,
            csr_matrix.indices[csr_matrix.indptr[row_idx] : csr_matrix.indptr[row_idx + 1]],
            1,
        )
    # Of shape ncols +1, with leading 0
    csc_indptr = np.cumsum(np.roll(np.append(csc_nnz, 0), 1))
    new_data = np.empty(csc_indptr[-1], csr_matrix.data.dtype)
    new_indices = np.empty(csc_indptr[-1], csr_matrix.indices.dtype)
    pointer = csc_indptr.copy()  # Keep one pointer per row, indicating where to store the next value
    for i in range(indices.size):
        row_idx = indices[i]
        row_start = csr_matrix.indptr[row_idx]
        row_end = csr_matrix.indptr[row_idx + 1]
        for idx in range(row_start, row_end):
            col_idx = csr_matrix.indices[idx]
            v = csr_matrix.data[idx]
            new_data[pointer[col_idx]] = v
            new_indices[pointer[col_idx]] = i
            pointer[col_idx] += 1
    return CSCMatrix(new_data, new_indices, csc_indptr, (indices.size, csr_matrix.shape[1]))


@njit(nogil=True, fastmath=True, cache=False)
def csr_get_contig_cols_into_csr(csr_matrix: CSRMatrix, lb: int, ub: int) -> CSRMatrix:
    """Perform contiguous vertical slicing of a CSR matrix and stores it into a CSR matrix.

    Args:
        csr_matrix (CSRMatrix): Input CSR matrix.
        lb (int): Lower bound of the vertical slicing.
        ub (int): Upper bound of the vertical slicing.

    Raises:
        ValueError: If bounds are not intelligible.

    Returns:
        CSRMatrix: The CSR matrix holding data of the sliced columns.

    Author: Rémy Dubois
    """
    if lb < 0 or ub > csr_matrix.shape[1] or lb > ub:
        raise ValueError((lb, ub))

    # Start by counting non zeros, and store each column's parcel in the csr arrays in the bounds var
    n_nzeros = np.zeros(csr_matrix.shape[0], dtype=np.int32)
    bounds = np.empty((csr_matrix.shape[0], 2), dtype=np.int32)
    for i in range(csr_matrix.shape[0]):
        start, end = csr_matrix.indptr[i], csr_matrix.indptr[i + 1]
        # Log instead of linear search, this is the only real optimization compared to `sparse.csc_matrix(csr_mat[:, slice])`
        # did not really pay attention to `side` but it seems to work as it is now.
        cb, rb = np.searchsorted(csr_matrix.indices[start:end], [lb, ub])
        bounds[i, 0] = cb
        bounds[i, 1] = rb
        n_nzeros[i] += rb - cb

    # # TODO: unify this with indptr
    chunk_offset_per_col = np.roll(n_nzeros, 1)
    chunk_offset_per_col[0] = 0
    chunk_offset_per_col = np.cumsum(chunk_offset_per_col)

    # # Local counter of how many elements where stored in result's columns
    data_placeholder = np.empty(n_nzeros.sum(), dtype=csr_matrix.data.dtype)
    indices_placeholder = np.empty(n_nzeros.sum(), dtype=csr_matrix.indices.dtype)
    counter = 0
    # Go through all the nonzero elements
    for i in range(csr_matrix.shape[0]):
        start, end = csr_matrix.indptr[i], csr_matrix.indptr[i + 1]
        cb, rb = bounds[i, 0], bounds[i, 1]
        chunk_data = csr_matrix.data[start + cb : start + rb]
        data_placeholder[counter : counter + chunk_data.size] = chunk_data.copy()
        chunk_indices = csr_matrix.indices[start + cb : start + rb] - lb
        indices_placeholder[counter : counter + chunk_indices.size] = chunk_indices.copy()
        counter += rb - cb

    indptr = np.append(chunk_offset_per_col, data_placeholder.size).astype(csr_matrix.indptr.dtype)
    return CSRMatrix(data_placeholder, indices_placeholder, indptr, (csr_matrix.shape[0], ub - lb))


@njit(nogil=True, fastmath=True, cache=False)
def csr_get_contig_cols_into_csc(csr_matrix: CSRMatrix, chunk_lb: int, chunk_ub: int) -> CSCMatrix:
    """Perform contiguous vertical slicing of a CSR matrix and stores it into a CSC matrix.

    Args:
        csr_matrix (CSRMatrix): Input CSR matrix.
        lb (int): Lower bound of the vertical slicing.
        ub (int): Upper bound of the vertical slicing.

    Raises:
        ValueError: If bounds are not intelligible.

    Returns:
        CSCMatrix: The CSC matrix holding data of the sliced columns.

    Author: Rémy Dubois
    """

    if chunk_lb < 0 or chunk_ub > csr_matrix.shape[1] or chunk_lb > chunk_ub:
        raise ValueError((chunk_lb, chunk_ub))
    # Start by counting non zeros, and store each column's parcel in the csr arrays in the bounds var
    n_nzeros = np.zeros(chunk_ub - chunk_lb, dtype=csr_matrix.indptr.dtype)
    bounds = np.empty((csr_matrix.shape[0], 2), dtype=np.int64)
    for i in range(csr_matrix.shape[0]):
        start, end = csr_matrix.indptr[i], csr_matrix.indptr[i + 1]
        # Log instead of linear search, this is the only real optimization compared to `sparse.csc_matrix(csr_mat[:, slice])`
        # did not really pay attention to `side` but it seems to work as it is now.
        cb, rb = np.searchsorted(csr_matrix.indices[start:end], [chunk_lb, chunk_ub])
        bounds[i, 0] = cb
        bounds[i, 1] = rb
        for j in range(start + cb, start + rb):
            n_nzeros[csr_matrix.indices[j] - chunk_lb] += 1

    # # TODO: unify this with indptr
    chunk_offset_per_col = np.roll(n_nzeros, 1)
    chunk_offset_per_col[0] = 0
    chunk_offset_per_col = np.cumsum(chunk_offset_per_col)
    # Local counter of how many elements where stored in result's columns
    data_placeholder = np.empty(n_nzeros.sum(), dtype=csr_matrix.data.dtype)
    indices_placeholder = np.empty(n_nzeros.sum(), dtype=csr_matrix.indices.dtype)
    col_counter = np.zeros(csr_matrix.shape[1], dtype=np.int64)
    # Go through all the nonzero elements
    for i in range(csr_matrix.shape[0]):
        start, end = csr_matrix.indptr[i], csr_matrix.indptr[i + 1]
        cb, rb = bounds[i, 0], bounds[i, 1]
        for j in range(start + cb, start + rb):
            col_idx = csr_matrix.indices[j] - chunk_lb
            ph_idx = col_counter[col_idx] + chunk_offset_per_col[col_idx]
            data_placeholder[ph_idx] = csr_matrix.data[j]
            indices_placeholder[ph_idx] = i
            col_counter[col_idx] += 1

    indptr = np.append(chunk_offset_per_col, data_placeholder.size).astype(csr_matrix.indptr.dtype)
    return CSCMatrix(
        data_placeholder,
        indices_placeholder,
        indptr,
        (csr_matrix.shape[0], chunk_ub - chunk_lb),
    )


# TODO: move this in the same file as its subroutines, so that caching as no risk of staling
@njit(nogil=True, fastmath=True, cache=False)
def csr_fold_change(X: CSRMatrix, grpc: GroupContainer, is_log1p: bool) -> np.ndarray:
    """Compute fold change from a CSR matrix of expression counts.

    Args:
        X (CSRMatrix): Input expression counts CSR matrix
        grpc (GroupContainer): GroupContainer
        is_log1p (bool): User-indicated flag telling if data was log1p or not.

    Returns:
        np.ndarray: Fold change of change (n_groups, n_genes)

    Author: Rémy Dubois
    """
    _assert_is_csr(X)
    group_agg_counts = np.zeros(shape=(grpc.counts.size, X.shape[1]), dtype=np.float64)
    # Sum expressions per group
    for i in range(X.shape[0]):
        start = X.indptr[i]
        end = X.indptr[i + 1]
        col_indices = X.indices[start:end]
        row_data = np.expm1(X.data[start:end]) if is_log1p else X.data[start:end]
        group_id = grpc.encoded_groups[i]
        _add_at_vec(group_agg_counts[group_id], col_indices, row_data)
    fold_change = fold_change_from_summed_expr(group_agg_counts, grpc)
    return fold_change
