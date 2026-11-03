import numpy as np
from numba import njit, prange

from illico.utils.sparse.csc import CSCMatrix, _assert_is_csc


@njit(nogil=True, cache=False, fastmath=True)
def _accumulate_group_ranksums_from_argsort(
    arr: np.ndarray,
    idx: np.ndarray,
    groups: np.ndarray,
    ranksums: np.ndarray,
    zero_values_offset: int = 0,
) -> tuple[float, int]:
    """From a given array of values, indices of sorted values (result of np.argsort) and groups, accumulate group rank
    sums in the placegolder `ranksums`.

    Args:
        arr (np.ndarray): Array of non sorted values
        idx (np.ndarray): Array of indices of sorted values
        groups (np.ndarray): Array of group indicator
        ranksums (np.ndarray): Plaeholder of shape (n_groups, ) where to accumulate rank sums
        zero_values_offset (int): If > 0, it means that there are zeros not present in the input arrays but that
        they should be accounted for. This is only used when the input adata is sparse, and ranksum is computed
        on non zero values.

    Returns:
        float: tie sums
        int: position of zeros in the sorted array (if zero_values_offset > 0)

    Author: Rémy Dubois

    """
    zero_pos = -1
    n = idx.size
    i = 0
    rank = 0
    tie_sum = 0.0

    while i < n:
        # find tie block
        j = i + 1
        if zero_values_offset > 0 and arr[idx[i]] > 0:
            zero_pos = i
            rank += zero_values_offset
            zero_values_offset = 0
        while j < n and arr[idx[j]] == arr[idx[i]]:
            j += 1
        avg_rank = rank + 0.5 * (j - i + 1)
        for k in range(i, j):
            if groups is not None:
                ranksums[groups[idx[k]]] += avg_rank
            else:
                # If no group is specified, ranks are sum reduced into one value for the whole column
                ranksums += avg_rank
        tie_count = j - i
        # Tie sum is the same for all groups
        tie_sum += tie_count**3 - tie_count
        i = j
        rank += tie_count

    if zero_pos == -1 and zero_values_offset > 0:
        zero_pos = n

    return tie_sum, zero_pos


@njit(nogil=True)
def rank_sum_and_ties_from_sorted(A: np.ndarray, B: np.ndarray, zero_values_offset: int = 0) -> tuple[np.ndarray]:
    """Compute rank sums and tie sums from two 1-d sorted arrays.

    This routine is similar to the leetcode "merge two sorted arrays", except it
    never returns to sorted array, instead it accumulate rank sums of the second array
    and tie sums for the combined arrays.

    This routine sits at the core of the one-versus-one (or one-versus-control) asymptotic
    wilcoxon rank sum test as it allows to sort controls only once.
    Args:
        A (np.ndarray): The first sorted array (controls)
        B (np.ndarray): The second sorted array (perturbed)
        zero_values_offset (int): If > 0, it means that there are zeros not present in the input arrays but that
        they should be accounted for. This is only used when the input adata is sparse, and ranksum is computed
        on non zero values.

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
    zero_pos = -1  # Default setting

    sum_ranks_B = 0.0
    tie_sum = 0.0

    # main sweep
    while i < nA and j < nB:
        # pick the smallest current value
        if A[i] < B[j]:
            v = A[i]
        else:
            v = B[j]

        if v > 0 and zero_values_offset > 0:
            zero_pos = k
            k += zero_values_offset
            zero_values_offset = 0

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

        if v > 0 and zero_values_offset > 0:
            zero_pos = k
            k += zero_values_offset
            zero_values_offset = 0

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

        if v > 0 and zero_values_offset > 0:
            zero_pos = k
            k += zero_values_offset
            zero_values_offset = 0

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

    # if zero_pos is still not set, it means there are no positive values in the entire column.
    # so zeros come at the end of it
    if zero_pos == -1 and zero_values_offset > 0:
        zero_pos = k

    return sum_ranks_B, tie_sum, zero_pos


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


@njit(nogil=True, cache=True, parallel=True)
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
    is_sorted = np.empty(indptr.size - 1, dtype=np.bool_)
    for k in prange(indptr.size - 1):
        start = indptr[k]
        end = indptr[k + 1]
        indices_slice = indices[start:end]
        is_sorted[k] = check_if_sorted(indices_slice)
    return np.all(is_sorted)


@njit(nogil=True, fastmath=True, parallel=False, cache=False)
def unique_from_sorted(x: np.ndarray, uniques: np.ndarray, counts: np.ndarray):
    """Compute unique values and counts from a sorted array.

    Parameters
    ----------
    x : np.ndarray
        Sorted input array.
    uniques : np.ndarray
        Preallocated array to store unique values.
    counts : np.ndarray
        Preallocated array to store counts of unique values.

    Returns
    -------
    uniques : np.ndarray
        Array of unique values.
    counts : np.ndarray
        Array of counts corresponding to unique values.
    k : int
        Number of unique values.

    """
    if x.size == 0:
        return uniques[:0], counts[:0], 0
    prev_val = x[0]
    count = 0
    k = 0
    for val in x:
        if val == prev_val:
            count += 1
        else:
            uniques[k] = prev_val
            counts[k] = count
            k += 1
            count = 1
            prev_val = val

    uniques[k] = prev_val
    counts[k] = count
    return uniques[: k + 1], counts[: k + 1], k + 1


@njit(nogil=True, fastmath=True, parallel=False, cache=False, inline="always")
def tie_sum_delta(a, b):
    """Compute the tie sum delta for a block of values.

    This func allows to increment the tie sum in an online fashion. Without it, one would need to store all counts
    independantly and compute the tie sum at the end, which is not memory efficient.

    """
    return b * (3 * (a**2) + 3 * a * b + b**2 - 1)


@njit(nogil=True, fastmath=True, parallel=False, cache=False)
def left_binsearch(arr: np.ndarray, x: float, lo: int = 0, hi: int | None = None) -> int:
    """Perform a left binary search on a sorted array.

    Parameters
    ----------
    arr : np.ndarray
        Sorted input array.
    x : float
        Value to search for.

    Returns
    -------
    int
        Index of the first occurrence of x in arr, or the index where x would be inserted to maintain sorted order.

    """
    lo = max(lo, 0)
    if hi is None:
        hi = arr.size
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(fastmath=True)
def rank_sum_and_ties_from_binsearch(
    ctrl_values: np.ndarray,
    n_uniques: int,
    offsets: np.ndarray,
    prt_values: np.ndarray,
    zero_values_offset: int = 0,
):
    """Compute rank sums and tie sums from a sorted control array and a perturbed array using binary search.

    This function computes (perturbed) rank sum and

    """
    n_pert = prt_values.shape[0]

    # Prepare variables for accumulation
    tie_sum = 0.0
    rank_sum = 0.0

    # Prepare a variable to track the position of zeros in the sorted array
    zero_pos = -1  # Default setting

    # Guardrail #1: if no control values are present, fallback to the linear merge that actually only executes the final exhaust loop (not merge)
    if n_uniques == 0:
        rs, ts, zpos = rank_sum_and_ties_from_sorted(ctrl_values, prt_values, zero_values_offset)
        return float(rs), ts, zpos
    # Guardrail #2: if no perturbed values are present, by def the ranksum is zero, and the participation of perturbed values in the tie sum is also zero
    if n_pert == 0:
        if zero_values_offset > 0:
            zero_index = 0 if ctrl_values[0] >= 0 else left_binsearch(ctrl_values, 0.0, 0, n_uniques)
            zero_pos = offsets[zero_index]
        else:
            zero_pos = 0
        return 0.0, 0, zero_pos

    # Initialize variables for the block comparison
    val = prt_values[0]
    count = 0
    lo = left_binsearch(ctrl_values, val, 0, n_uniques)
    prev_lo = 0

    # Iterate over the target array to compute rank sums and tie sums
    for i in range(prt_values.shape[0]):
        x = prt_values[i]

        # Get position of this x in control values
        # lo = left_binsearch(ctrl_values, x, lo, n_uniques)

        # Record position of zeros in the sorted array if needed
        if x > 0 and zero_pos < 0 and zero_values_offset > 0:
            # Guard against the commmon case where no value is negative
            if ctrl_values[0] < 0:
                lo_zero = left_binsearch(ctrl_values, 0.0, 0, n_uniques)
            else:
                lo_zero = 0
            zero_pos = offsets[lo_zero] + i

        if x != val:  # If different, we are done with this value's block
            lo = left_binsearch(ctrl_values, x, lo, n_uniques)
            if (
                prev_lo < n_uniques and ctrl_values[prev_lo] == val
            ):  # If the value is present in controls, increment only of the delta
                ctrl_cnt = offsets[prev_lo + 1] - offsets[prev_lo]
                block_size = count + ctrl_cnt
                tie_sum += tie_sum_delta(ctrl_cnt, count)
            else:  # If the value is not present in controls
                tie_sum += float(count) ** 3 - float(count)
                block_size = count

            first_rank = offsets[prev_lo] + (i - count)
            # if the zero has been crossed, the first rank is offset
            if zero_pos >= 0 and val > 0:
                first_rank += zero_values_offset
            rank_sum += count * (first_rank + (block_size + 1) / 2)

            count = 1
            val = x
        else:
            count += 1

        prev_lo = lo

    # finalize the last block
    if prev_lo < n_uniques and ctrl_values[prev_lo] == val:
        ctrl_cnt = offsets[prev_lo + 1] - offsets[prev_lo]
        block_size = count + ctrl_cnt
        tie_sum += tie_sum_delta(ctrl_cnt, count)
    else:
        block_size = count
        tie_sum += float(count) ** 3 - float(count)
    first_rank = offsets[prev_lo] + (i + 1 - count)
    if zero_pos >= 0 and val > 0:
        first_rank += zero_values_offset
    rank_sum += count * (first_rank + 0.5 * (block_size + 1))

    # if zero_pos is still not set, it means no perturbed value is positive
    # but we still need to find the position of the first positive control value
    if zero_pos == -1:
        if zero_values_offset > 0:
            zero_index_ctrl = 0 if ctrl_values[0] >= 0 else left_binsearch(ctrl_values, 0.0, 0, n_uniques)
            zero_pos = n_pert + offsets[zero_index_ctrl]
        else:
            zero_pos = 0

    return rank_sum, tie_sum, zero_pos
