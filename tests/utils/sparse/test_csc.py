import numpy as np
import pytest
from scipy import sparse as sc_sparse

from illico.utils.sparse.csc import (
    _assert_is_csc,
    csc_count_nonzeros,
    csc_get_cols,
    csc_get_contig_cols_into_csr,
    csc_to_csr,
)


def test_assert_is_csc(rand_csc):
    _assert_is_csc(rand_csc)  # Should not raise


@pytest.mark.xfail(raises=AssertionError)
def test_assert_is_csc_invalid(rand_csc):
    _assert_is_csc(rand_csc)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_csc_count_nonzeros(rand_csc, axis):
    nnz = csc_count_nonzeros(rand_csc, axis=axis)
    py_mat = sc_sparse.csc_matrix((rand_csc.data, rand_csc.indices, rand_csc.indptr), shape=rand_csc.shape).toarray()
    expected = np.count_nonzero(py_mat, axis=axis)
    np.testing.assert_allclose(nnz, expected)


def test_csc_get_cols(rand_csc):
    rng = np.random.RandomState(0)
    indices = rng.randint(0, rand_csc.shape[1], size=rand_csc.shape[1] // 2, dtype=np.int32)
    sub_csc = csc_get_cols(rand_csc, indices)
    py_sub_csc = sc_sparse.csc_matrix((rand_csc.data, rand_csc.indices, rand_csc.indptr), shape=rand_csc.shape)[
        :, indices
    ]
    np.testing.assert_allclose(sub_csc.data, py_sub_csc.data)
    np.testing.assert_allclose(sub_csc.indices, py_sub_csc.indices)
    np.testing.assert_allclose(sub_csc.indptr, py_sub_csc.indptr)
    np.testing.assert_allclose(sub_csc.shape, py_sub_csc.shape)


def test_csc_to_csr(rand_csc):
    csr_matrix = csc_to_csr(rand_csc)
    py_csc = sc_sparse.csc_matrix((rand_csc.data, rand_csc.indices, rand_csc.indptr), shape=rand_csc.shape)
    py_csr = sc_sparse.csr_matrix(py_csc)
    np.testing.assert_allclose(py_csr.data, csr_matrix.data)
    np.testing.assert_allclose(py_csr.indices, csr_matrix.indices)
    np.testing.assert_allclose(py_csr.indptr, csr_matrix.indptr)
    np.testing.assert_allclose(py_csr.shape, csr_matrix.shape)


def test_csc_get_contig_cols_into_csr(rand_csc):
    chunk_lb = rand_csc.shape[1] // 3
    chunk_ub = 2 * rand_csc.shape[1] // 3
    csr_chunk = csc_get_contig_cols_into_csr(rand_csc, chunk_lb, chunk_ub)
    py_csc = sc_sparse.csc_matrix((rand_csc.data, rand_csc.indices, rand_csc.indptr), shape=rand_csc.shape)
    py_chunk = py_csc[:, chunk_lb:chunk_ub].tocsr()
    np.testing.assert_allclose(py_chunk.data, csr_chunk.data)
    np.testing.assert_allclose(py_chunk.indices, csr_chunk.indices)
    np.testing.assert_allclose(py_chunk.indptr, csr_chunk.indptr)
    np.testing.assert_allclose(py_chunk.shape, csr_chunk.shape)


# def test_csc_to_dense(rand_csc):
#     dense_mat = csc_to_dense(rand_csc)
#     py_csc = sc_sparse.csc_matrix(
#         (rand_csc.data, rand_csc.indices, rand_csc.indptr), shape=rand_csc.shape
#     )
#     py_dense = py_csc.toarray()
#     np.testing.assert_allclose(py_dense, dense_mat)
