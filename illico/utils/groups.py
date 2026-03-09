from collections import namedtuple
from typing import Any

import numpy as np

GroupContainer = namedtuple(
    "GroupContainer",
    [
        "encoded_groups",
        "counts",
        "indices",
        "indptr",
        "encoded_ref_group",
    ],
)


def encode_and_count_groups(groups: np.ndarray, ref_group: Any) -> tuple[np.ndarray, GroupContainer]:
    """Build the GroupContainer holding all group-related information.

    GroupContainer holds:
    - original group information
    - reference group (control)
    - encoded groups
    - unique raw groups
    - counts (of cell, per group)
    - indices, indptr in a RLE format
    - encoded reference group (control)

    Args:
        groups (np.ndarray): 1-d array holding group labels, one per cell
        ref_group (Any): Flag

    Returns:
        unique_groups (np.ndarray): Array of unique group labels
        GroupContainer: GroupContainer holding all group-related information.

    Author: Rémy Dubois
    """
    if ref_group not in groups and ref_group is not None:
        raise ValueError(f"Reference group `{ref_group}` is not present in the group labels.")
    # Determine group indices
    group_indices = np.argsort(groups, kind="stable").astype(np.uint64)

    # Count occcurrences of each group
    sorted_groups = groups[group_indices]
    change_idx = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    group_counts = np.diff(np.r_[0, change_idx, sorted_groups.size]).astype(np.uint64)

    # Find unique groups
    unique_groups = sorted_groups[np.r_[0, change_idx]]

    # Encode groups as integers
    encoded_groups = np.searchsorted(unique_groups, groups).astype(np.uint64)

    # Build indptr
    group_indptr = np.cumsum(np.insert(group_counts, 0, 0)).astype(np.uint64)

    return unique_groups, GroupContainer(
        encoded_groups=encoded_groups,
        counts=group_counts.astype(np.uint64),
        indices=group_indices,
        indptr=group_indptr,
        encoded_ref_group=(
            -1 if ref_group is None else int(np.searchsorted(unique_groups, ref_group))
        ),  # Weirdly enough, this must be -1 and not None, otherwise Numba fails to compile various functions, especially branching
    )
