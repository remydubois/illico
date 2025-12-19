import numpy as np
import pytest
from scipy import sparse as sc_sparse

from illico.utils.sparse.csr import (
    _assert_is_csr,
    csr_count_nonzeros,
    csr_get_contig_cols_into_csc,
    csr_get_contig_cols_into_csr,
    csr_get_rows_into_csc,
    csr_to_csc,
)


def test_assert_is_csr(rand_csr):
    _assert_is_csr(rand_csr)  # Should not raise


@pytest.mark.xfail(raises=AssertionError)
def test_assert_is_csr_invalid(rand_csr):
    _assert_is_csr(rand_csr)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_csr_count_nonzeros(rand_csr, axis):
    nnz = csr_count_nonzeros(rand_csr, axis=axis)
    py_mat = sc_sparse.csr_matrix((rand_csr.data, rand_csr.indices, rand_csr.indptr), shape=rand_csr.shape).toarray()
    expected = np.count_nonzero(py_mat, axis=axis)
    np.testing.assert_allclose(nnz, expected)


def test_csr_get_rows(rand_csr):
    rng = np.random.RandomState(0)
    indices = rng.randint(0, rand_csr.shape[0], size=rand_csr.shape[0] // 2, dtype=np.int32)
    sub_csc = csr_get_rows_into_csc(rand_csr, indices)
    py_sub_csr = sc_sparse.csr_matrix((rand_csr.data, rand_csr.indices, rand_csr.indptr), shape=rand_csr.shape)[
        indices, :
    ].tocsc()
    np.testing.assert_allclose(sub_csc.data, py_sub_csr.data)
    np.testing.assert_allclose(sub_csc.indices, py_sub_csr.indices)
    np.testing.assert_allclose(sub_csc.indptr, py_sub_csr.indptr)
    np.testing.assert_allclose(sub_csc.shape, py_sub_csr.shape)


def test_csr_to_csc(rand_csr):
    csc_matrix = csr_to_csc(rand_csr)
    py_csr = sc_sparse.csr_matrix((rand_csr.data, rand_csr.indices, rand_csr.indptr), shape=rand_csr.shape)
    py_csc = sc_sparse.csc_matrix(py_csr)
    np.testing.assert_allclose(py_csc.data, csc_matrix.data)
    np.testing.assert_allclose(py_csc.indices, csc_matrix.indices)
    np.testing.assert_allclose(py_csc.indptr, csc_matrix.indptr)
    np.testing.assert_allclose(py_csc.shape, csc_matrix.shape)


def test_csr_get_contig_cols_into_csc(rand_csr):
    chunk_lb = rand_csr.shape[1] // 3
    chunk_ub = 2 * rand_csr.shape[1] // 3
    csc_chunk = csr_get_contig_cols_into_csc(rand_csr, chunk_lb, chunk_ub)
    py_csr = sc_sparse.csr_matrix((rand_csr.data, rand_csr.indices, rand_csr.indptr), shape=rand_csr.shape)
    py_chunk = py_csr[:, chunk_lb:chunk_ub].tocsc()
    np.testing.assert_allclose(py_chunk.data, csc_chunk.data)
    np.testing.assert_allclose(py_chunk.indices, csc_chunk.indices)
    np.testing.assert_allclose(py_chunk.indptr, csc_chunk.indptr)
    np.testing.assert_allclose(py_chunk.shape, csc_chunk.shape)


def test_csr_get_contig_cols_into_csr(rand_csr):
    chunk_lb = rand_csr.shape[1] // 3
    chunk_ub = 2 * rand_csr.shape[1] // 3
    csr_chunk = csr_get_contig_cols_into_csr(rand_csr, chunk_lb, chunk_ub)
    py_csr = sc_sparse.csr_matrix((rand_csr.data, rand_csr.indices, rand_csr.indptr), shape=rand_csr.shape)
    py_chunk = py_csr[:, chunk_lb:chunk_ub]
    np.testing.assert_allclose(py_chunk.data, csr_chunk.data)
    np.testing.assert_allclose(py_chunk.indices, csr_chunk.indices)
    np.testing.assert_allclose(py_chunk.indptr, csr_chunk.indptr)
    np.testing.assert_allclose(py_chunk.shape, csr_chunk.shape)
