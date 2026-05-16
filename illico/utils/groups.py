from collections import namedtuple
from typing import Any, Iterable

import numpy as np
from loguru import logger

GroupContainer = namedtuple(
    "GroupContainer",
    [
        "selected_group_ids",
        "non_excluded_group_ids",
        "encoded_groups",
        "counts",
        "indices",
        "ovr_inclusion_indices",
        "indptr",
        "encoded_ref_group",
    ],
)


def sanitize_groups(
    groups: np.ndarray, ref_group: Any, group_subset: Iterable[Any] | None = None, exclude: Iterable[Any] | None = None
) -> tuple[np.ndarray, GroupContainer]:
    # Check that subset and exclude dont contradict each other
    # Check that exclude is not specified if OVO
    pass


def encode_and_count_groups(
    groups: np.ndarray, ref_group: Any, group_subset: Iterable[Any] | None = None, exclude: Iterable[Any] | None = None
) -> tuple[np.ndarray, GroupContainer]:
    """Build the GroupContainer holding all group-related information.

    GroupContainer holds:
    - number of valid groups (i.e. groups that are selected for testing, including the reference group)
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
        group_subset (Iterable[Any] | None, optional): Subset of groups to test. If None, all groups are tested. Defaults to None.
        exclude (Iterable[Any] | None, optional): Groups to exclude from OVR testing. If None, no group is excluded. Defaults to None.

    Returns:
        unique_groups (np.ndarray): Array of unique group labels
        GroupContainer: GroupContainer holding all group-related information.

    Author: Rémy Dubois

    """
    if ref_group not in groups and ref_group is not None:
        raise ValueError(f"Reference group `{ref_group}` is not present in the group labels.")

    # Sort groups, putting selected groups first, then non-selected groups
    group_indices = np.argsort(groups, kind="stable")
    sorted_groups = groups[group_indices]

    change_idx = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    group_counts = np.diff(np.r_[0, change_idx, sorted_groups.size]).astype(np.uint64)

    # Find unique groups
    unique_groups = sorted_groups[np.r_[0, change_idx]]

    # Gather labels of groups listed in "groups"
    if group_subset is not None:
        group_subset = np.asarray(group_subset)
        if not len(group_subset):
            raise ValueError("Group subset can not be empty.")
        if not (_m := np.isin(group_subset, groups)).all():
            missing_groups = ",".join(map(str, group_subset[~_m]))
            raise ValueError(
                f"All groups in `group_subset` must be present in the group labels. {missing_groups} could not be found in the group labels."
            )
        selected_groups = np.isin(unique_groups, group_subset)
        if ref_group is not None:
            selected_groups |= (
                unique_groups == ref_group
            )  # Ensure that the reference group is included, even if not in group_subset
        selected_group_ids = np.where(selected_groups)[0]
    else:
        selected_groups = np.ones(unique_groups.shape[0], dtype=bool)
        selected_group_ids = np.arange(unique_groups.size, dtype=np.uint64)

    encoded_groups = np.searchsorted(unique_groups, groups).astype(np.uint64)

    # Build indptr
    group_indptr = np.cumsum(np.insert(group_counts, 0, 0)).astype(np.uint64)

    # Gather indices of non-excluded groups
    if exclude is not None and ref_group is None:
        exclude = np.asarray(exclude)
        if not len(exclude):
            raise ValueError("Exclude can not be empty.")
        if not (_m := np.isin(exclude, unique_groups)).all():
            missing_groups = ",".join(map(str, exclude[~_m]))
            logger.warning(
                f"Some groups in `exclude` are not present in the group labels. {missing_groups} could not be found in the group labels."
            )
        if not (_m := np.isin(exclude, unique_groups[selected_group_ids])).all():
            missing_groups = ",".join(map(str, exclude[~_m]))
            logger.warning(
                f"Some groups in `exclude` are present in the group labels but not in `groups`. {missing_groups} could not be found."
            )
        inclusion_mask = ~np.isin(groups, exclude)
        ovr_inclusion_indices = np.flatnonzero(inclusion_mask)
        # If the test is OVR, by definition only non-excluded groups are selected
        non_excluded_group_ids = np.where(~np.isin(unique_groups, exclude))[0]
        # Restrict the selected groups: excluded groups can not be selected.
        selected_group_ids = np.where((~np.isin(unique_groups, exclude)) & selected_groups)[0]
    else:
        if exclude is not None:
            logger.warning(
                "`exclude` is not relevant when a reference group is specified, as the test is OVO. Excluded groups will be ignored."
            )
        non_excluded_group_ids = np.arange(unique_groups.size, dtype=np.uint64)
        ovr_inclusion_indices = np.arange(groups.size, dtype=np.uint64)

    grpc = GroupContainer(
        selected_group_ids=selected_group_ids,
        non_excluded_group_ids=non_excluded_group_ids,  # Useless, only used in counts but could rely on included_cell_indices for that.
        encoded_groups=encoded_groups,
        counts=group_counts.astype(np.uint64),
        indices=group_indices,
        ovr_inclusion_indices=ovr_inclusion_indices,
        indptr=group_indptr,
        encoded_ref_group=(
            -1 if ref_group is None else int(np.searchsorted(unique_groups, ref_group))
        ),  # Weirdly enough, this must be -1 and not None, otherwise Numba fails to compile various functions, especially branching
    )

    return unique_groups, grpc
