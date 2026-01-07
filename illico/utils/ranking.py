import numpy as np
from numba import njit

from illico.utils.sparse.csc import CSCMatrix, _assert_is_csc


@njit(nogil=True, cache=False, fastmath=True)
def _accumulate_group_ranksums_from_argsort(
    arr: np.ndarray, idx: np.ndarray, groups: np.ndarray, ranksums: np.ndarray
) -> np.ndarray:
    """
    From a given array of values, indices of sorted values (result of np.argsort) and groups,
    accumulate group rank sums in the placegolder `ranksums`.

    Args:
        arr (np.ndarray): Array of non sorted values
        idx (np.ndarray): Array of indices of sorted values
        groups (np.ndarray): Array of group indicator
        ranksums (np.ndarray): Plaeholder of shape (n_groups, ) where to accumulate rank sums

    Returns:
        np.ndarray: tie sums

    Author: Rémy Dubois
    """
    # if ranks is None:
    #     ranks = np.empty(arr.size, dtype=np.float64)
    n = idx.size
    i = 0
    tie_sum = 0.0
    while i < n:
        # find tie block
        j = i + 1
        while j < n and arr[idx[j]] == arr[idx[i]]:
            j += 1
        # average rank for indices i..j-1 (1-based)
        avg_rank = 0.5 * (i + 1 + j)  # mean of [i+1, ..., j]
        for k in range(i, j):
            if groups is not None:
                ranksums[groups[idx[k]]] += avg_rank
            else:
                # If no group is specified, ranks are sum reduced into one value for the whole column
                ranksums += avg_rank
        tie_count = j - i
        # Tie sum is the same for all froups
        tie_sum += tie_count**3 - tie_count
        i = j

    return tie_sum


@njit(nogil=True)
def rank_sum_and_ties_from_sorted(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray]:
    """Compute rank sums and tie sums from two 1-d sorted arrays.

    This routine is similar to the leetcode "merge two sorted arrays", except it
    never returns to sorted array, instead it accumulate rank sums of the second array
    and tie sums for the combined arrays.

    This routine sits at the core of the one-versus-one (or one-versus-control) asymptotic
    wilcoxon rank sum test as it allows to sort controls only once.
    Args:
        A (np.ndarray): The first sorted array (controls)
        B (np.ndarray): The second sorted array (perturbed)

    Returns:
        tuple[np.ndarray]: Ranks sum from the second array, and tie sums for the combined
        arrays.

    Author: Rémy Dubois
    """
    nA = len(A)
    nB = len(B)

    i = 0
    j = 0
    k = 0  # number of items processed so far (0-based)

    sum_ranks_B = 0.0
    tie_sum = 0.0

    # main sweep
    while i < nA and j < nB:
        # pick the smallest current value
        if A[i] < B[j]:
            v = A[i]
        else:
            v = B[j]

        # count occurrences in A
        tA = 0
        ii = i
        while ii < nA and A[ii] == v:
            tA += 1
            ii += 1

        # count occurrences in B
        tB = 0
        jj = j
        while jj < nB and B[jj] == v:
            tB += 1
            jj += 1

        t = tA + tB
        avg_rank = k + 0.5 * (t + 1)

        if t > 1:
            tie_sum += t * t * t - t

        sum_ranks_B += tB * avg_rank

        k += t
        i = ii
        j = jj

    # Drain remaining A
    while i < nA:
        v = A[i]

        # count in A
        tA = 0
        ii = i
        while ii < nA and A[ii] == v:
            tA += 1
            ii += 1

        t = tA
        avg_rank = k + 0.5 * (t + 1)

        # no B contribution because tB=0
        if t > 1:
            tie_sum += t * t * t - t

        k += t
        i = ii

    # Drain remaining B
    while j < nB:
        v = B[j]

        # count in B
        tB = 0
        jj = j
        while jj < nB and B[jj] == v:
            tB += 1
            jj += 1

        t = tB
        avg_rank = k + 0.5 * (t + 1)

        sum_ranks_B += tB * avg_rank
        if t > 1:
            tie_sum += t * t * t - t

        k += t
        j = jj

    return sum_ranks_B, tie_sum


@njit(nogil=True, cache=False)
def _sort_csc_columns_inplace(csc_matrix: CSCMatrix) -> None:
    """Sort CSC columns in place.

    Args:
        csc_matrix (CSCMatrix): Input CSC matrix.

    Author: Rémy Dubois
    """
    _assert_is_csc(csc_matrix)
    for j in range(csc_matrix.shape[1]):
        csc_matrix.data[csc_matrix.indptr[j] : csc_matrix.indptr[j + 1]].sort()


@njit(nogil=True, cache=False)
def sort_along_axis(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sort a dense array along a given axis.

    Args:
        X (np.ndarray): Input dense array.
        axis (int, optional): Axis along which to sort. Defaults to 0.

    Returns:
        np.ndarray: Sorted array.

    Author: Rémy Dubois
    """
    sorted_X = np.empty_like(X)
    if axis == 0:
        for j in range(X.shape[1]):
            sorted_X[:, j] = np.sort(X[:, j])
    elif axis == 1:
        for i in range(X.shape[0]):
            sorted_X[i, :] = np.sort(X[i, :])
    else:
        raise ValueError(f"Axis {axis} is not supported.")
    return sorted_X


@njit(nogil=True, cache=False)
def _sort_along_axis_inplace(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sort a dense array along a given axis.

    Args:
        X (np.ndarray): Input dense array.
        axis (int, optional): Axis along which to sort. Defaults to 0.

    Returns:
        np.ndarray: Sorted array.

    Author: Rémy Dubois
    """
    if axis == 0:
        for j in range(X.shape[1]):
            X[:, j].sort()
    elif axis == 1:
        for i in range(X.shape[0]):
            X[i, :].sort()
    else:
        raise ValueError(f"Axis {axis} is not supported.")


@njit(nogil=True, cache=False)
def check_if_sorted(arr: np.ndarray) -> bool:
    """Check if an array is sorted. O(n)

    Parameters
    ----------
    arr : np.ndarray
        1-d array to check

    Returns
    -------
    bool
        If sorted or not.

    Author: Rémy Dubois
    """
    for i in range(1, arr.size):
        if arr[i] < arr[i - 1]:
            return False
    return True


@njit(nogil=True, cache=False)
def check_indices_sorted_per_parcel(
    indices: np.ndarray,
    indptr: np.ndarray,
) -> bool:
    """Check if indices of a sparse array are sorted.

    This is esssential if input data is CSR. Indeed, chunking makes use of
    binary search on indices, which requires sorted indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices
    indptr : np.ndarray
        Indptr

    Returns
    -------
    bool
        True if all indices subarrays are sorted. False otherwise.
    """
    for k in range(indptr.size - 1):
        start = indptr[k]
        end = indptr[k + 1]
        indices_slice = indices[start:end]
        if not check_if_sorted(indices_slice):
            return False
    return True
