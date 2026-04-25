import numpy as np
import pytest
from scipy import sparse as sc_sparse
from scipy.stats import rankdata

from illico.utils.ranking import (
    _accumulate_group_ranksums_from_argsort,
    _sort_csc_columns_inplace,
    rank_sum_and_ties_from_sorted,
)
from illico.utils.sparse.csc import CSCMatrix


@pytest.mark.parametrize("format", ["dense", "sparse"])
def test_rank_sum_and_ties_from_sorted(format):
    rng = np.random.RandomState(0)
    A = rng.randint(-10, 10, size=20)
    A[:2] = 0  # Add some zeros manually
    B = rng.randint(-10, 10, size=15)
    B[:3] = 0  # Add some zeros manually

    # First compute real ranksum and tie sum
    combined = np.concatenate([A, B])
    ranks = rankdata(combined, method="average")
    ranksum_B_manual = ranks[len(A) :].sum()
    _, tie_counts = np.unique(combined, return_counts=True)
    manual_tie_sum = (tie_counts**3 - tie_counts).sum()

    if format == "sparse":
        n_zeros_A = (A == 0).sum()
        n_zeros_B = (B == 0).sum()
        n_zeros = n_zeros_A + n_zeros_B
        A, B = A[A != 0], B[B != 0]  # Keep only positive values to have ties
    else:
        n_zeros_A = n_zeros_B = n_zeros = 0
    A.sort()
    B.sort()

    ranksum_B, tie_sum, zero_pos = rank_sum_and_ties_from_sorted(A, B, n_zeros)
    # Add contributions of zeros to the ranksum
    ranksum_B += n_zeros_B * (zero_pos + (n_zeros + 1) / 2.0)
    # Add contributions of zeros to the tie sum
    tie_sum += n_zeros * (n_zeros**2 - 1)
    # Check
    np.testing.assert_allclose(ranksum_B, ranksum_B_manual)
    np.testing.assert_allclose(tie_sum, manual_tie_sum)


@pytest.mark.parametrize("format", ["dense", "sparse"])
def test_group_ranksum_accumulation(format):
    rng = np.random.RandomState(0)
    arr = rng.rand(30)
    arr[:5] = 0  # Add some zeros manually
    groups = rng.randint(0, 3, size=30)

    # Manually compute ranks and tie sums on the whole array
    ranks = rankdata(arr, method="average")
    manual_ranksums = np.zeros(3, dtype=np.float64)
    for i in range(len(arr)):
        manual_ranksums[groups[i]] += ranks[i]
    _, tie_counts = np.unique(arr, return_counts=True)
    manual_tie_sum = (tie_counts**3 - tie_counts).sum()

    # Now compute them with illico utils
    if format == "sparse":
        n_zeros = (arr == 0).sum()
        nz_per_group = np.array([((groups == g) & (arr == 0)).sum() for g in range(3)])
        groups = groups[arr != 0]
        arr = arr[arr != 0]
    else:
        n_zeros = 0
        nz_per_group = np.zeros(3, dtype=np.float64)

    idx = np.argsort(arr)
    ranksums = np.zeros(3, dtype=np.float64)
    tie_sum, zero_pos = _accumulate_group_ranksums_from_argsort(arr, idx, groups, ranksums, n_zeros)

    # Add contributions of zeros to the ranksums
    ranksums += nz_per_group * (zero_pos + (n_zeros + 1) / 2.0)
    # Add contributions of zeros to the tie sum
    tie_sum += n_zeros * (n_zeros**2 - 1)

    # Check
    np.testing.assert_allclose(ranksums, manual_ranksums)
    np.testing.assert_allclose(tie_sum, manual_tie_sum)


def test_sort_csc_columns_inplace():

    data = np.array([3, 1, 2, 5, 4], dtype=np.float64)
    indices = np.array([0, 2, 1, 0, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3, 5], dtype=np.int64)  # 3 columns
    csc_matrix = CSCMatrix(data=data, indices=indices, indptr=indptr, shape=(3, 3))

    _sort_csc_columns_inplace(csc_matrix)

    for j in range(csc_matrix.shape[1]):
        col_data = csc_matrix.data[csc_matrix.indptr[j] : csc_matrix.indptr[j + 1]]
        assert np.all(col_data[:-1] <= col_data[1:])
