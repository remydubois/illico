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
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    groups_mapping = {g: i for i, g in enumerate(unique_groups)}
    encoded_groups = np.array([groups_mapping[g] for g in groups])
    # This should sort them in the same order as the first np.unique
    # TODO: np.lexsort so that indices are sorted within group as well. Just to be safer.^
    group_indices = np.argsort(groups)
    group_indptr = np.cumsum(np.insert(group_counts, 0, 0))

    return unique_groups, GroupContainer(
        encoded_groups=encoded_groups,
        counts=group_counts,
        indices=group_indices,
        indptr=group_indptr,
        encoded_ref_group=(
            -1 if ref_group is None else groups_mapping[ref_group]
        ),  # Weirdly enough, this must be -1 and not None, otherwise Numba fails to compile various functions, especially branching
    )
