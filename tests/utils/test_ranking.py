import numpy as np
from scipy import sparse as sc_sparse
from scipy.stats import rankdata

from illico.utils.ranking import (
    _accumulate_group_ranksums_from_argsort,
    _sort_csc_columns_inplace,
    rank_sum_and_ties_from_sorted,
)
from illico.utils.sparse.csc import CSCMatrix


def test_rank_sum_and_ties_from_sorted():
    rng = np.random.RandomState(0)
    A = rng.randint(0, 10, size=20)
    B = rng.randint(0, 10, size=15)
    A.sort()
    B.sort()

    ranksum_B, tie_sum = rank_sum_and_ties_from_sorted(A, B)

    # Now manually compute ranksum
    combined = np.concatenate([A, B])
    ranks = rankdata(combined, method="average")
    ranksum_B_manual = ranks[len(A) :].sum()
    # Manually compute tie sum
    _, tie_counts = np.unique(combined, return_counts=True)
    manual_tie_sum = (tie_counts**3 - tie_counts).sum()

    # Check
    np.testing.assert_allclose(ranksum_B, ranksum_B_manual)
    np.testing.assert_allclose(tie_sum, manual_tie_sum)


def test_group_ranksum_accumulation():
    rng = np.random.RandomState(0)
    arr = rng.rand(30)
    groups = rng.randint(0, 3, size=30)
    idx = np.argsort(arr)

    ranksums = np.zeros(3, dtype=np.float64)
    tie_sum = _accumulate_group_ranksums_from_argsort(arr, idx, groups, ranksums)

    # Manually compute ranks
    ranks = rankdata(arr, method="average")
    manual_ranksums = np.zeros(3, dtype=np.float64)
    for i in range(len(arr)):
        manual_ranksums[groups[i]] += ranks[i]

    # Manually compute tie sum
    _, tie_counts = np.unique(arr, return_counts=True)
    manual_tie_sum = (tie_counts**3 - tie_counts).sum()

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
