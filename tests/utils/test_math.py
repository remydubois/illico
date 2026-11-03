import numpy as np
import pytest

from illico.utils.math import _add_at_scalar, _add_at_vec, compute_batch_bounds, diff


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


@pytest.mark.parametrize("target", [256, 512, 1024])
def test_compute_batch_bounds_approx_targets(target):
    n_genes = 4000
    n_threads = 4
    bounds, eff_batch_size = compute_batch_bounds(n_genes, f"~{target}", n_threads)

    # Check bounds form a contiguous partition from 0 to n_genes
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n_genes
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]

    # Number of batches should be a multiple of n_threads (no worker starvation)
    assert len(bounds) % n_threads == 0

    # Effective batch size should be reasonably close to target
    assert abs(eff_batch_size - target) < target * 0.5


def test_compute_batch_bounds_small_inputs():
    # n_genes < target
    bounds, size = compute_batch_bounds(200, "~512", n_threads=4)
    assert bounds == [(0, 200)]
    assert size == 200

    # n_genes < n_threads
    bounds, size = compute_batch_bounds(3, "~512", n_threads=4)
    assert bounds == [(0, 3)]
    assert size == 3


def test_compute_batch_bounds_explicit_integer():
    n_genes = 1000
    bounds, size = compute_batch_bounds(n_genes, 200, n_threads=1)
    assert bounds == [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    assert size == 200


@pytest.mark.parametrize("invalid_batch_size", ["invalid", "~abc", "~0", "~-10", "~", 0, -5, None, True])
def test_compute_batch_bounds_invalid_values(invalid_batch_size):
    with pytest.raises((ValueError, TypeError)):
        compute_batch_bounds(1000, invalid_batch_size, n_threads=4)


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
