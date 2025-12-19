import numpy as np
import pytest

from illico.utils.math import _add_at_scalar, _add_at_vec, _warn_log1p, diff


def test_add_at_scalar():
    a = np.zeros(5, dtype=np.float64)
    b = np.array([0, 2, 4, 2, 3, 0], dtype=np.int64)
    c = 1.0
    _add_at_scalar(a, b, c)
    expected = np.zeros(5, dtype=np.float64)
    np.add.at(expected, b, c)
    np.testing.assert_allclose(a, expected)


def test_add_at_vec():
    a = np.zeros(5, dtype=np.float64)
    b = np.array([0, 2, 4, 2, 3, 0], dtype=np.int64)
    c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    _add_at_vec(a, b, c)
    expected = np.zeros(5, dtype=np.float64)
    np.add.at(expected, b, c)
    np.testing.assert_allclose(a, expected)


def test_diff():
    x = np.array([1, 3, 6, 10, 15], dtype=np.int64)
    result = diff(x)
    expected = np.diff(x)
    np.testing.assert_array_equal(result, expected)


""" --- IGNORE --- as warn_log1p is no longer used."""
# def test_warn_log1p_1(rand_adata, data_format):
#     # Case 1: Data is raw counts, is_log1p=True
#     with pytest.warns(UserWarning):
#         _warn_log1p(rand_adata.layers[data_format], is_log1p=True)


# def test_warn_log1p_2(rand_adata, data_format):
#     # Case 2: Data is raw counts, is_log1p=False (no warning)
#     _warn_log1p(rand_adata.layers[data_format], is_log1p=False)


# def test_warn_log1p_3(rand_adata, data_format):
#     # Case 3: Data is log1p transformed, is_log1p=False
#     rand_adata.layers[data_format] = np.log1p(rand_adata.layers[data_format])
#     with pytest.warns(UserWarning):
#         _warn_log1p(rand_adata.layers[data_format], is_log1p=False)


# def test_warn_log1p_4(rand_adata, data_format):
#     # Case 4: Data is log1p transformed, is_log1p=True (no warning)
#     _warn_log1p(rand_adata.layers[data_format], is_log1p=True)
