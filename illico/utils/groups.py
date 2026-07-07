from collections import namedtuple
from typing import Any, Iterable

import numpy as np
from loguru import logger

GroupContainer = namedtuple(
    "GroupContainer",
    [
        "n_selected_groups",
        "encoded_groups",
        "counts",
        "indices",
        "included_cell_indices",
        "indptr",
        "encoded_ref_group",
    ],
)


def isin_fast(labels: np.ndarray, values: Iterable[Any]) -> np.ndarray:
    """Faster equivalent of npisin."""
    # Convert values to a set for O(1) lookups
    value_set = set(values)
    return np.array([label in value_set for label in labels], dtype=bool)


def sanitize_group_args(
    groups: np.ndarray, ref_group: Any, group_subset: Iterable[Any] | None = None, exclude: Iterable[Any] | None = None
) -> tuple[np.ndarray, GroupContainer]:
    """Check the consistency of group-related arguments and sanitize them."""
    # Make all sanity checks much faster
    unique_groups = np.array(sorted(set(groups)))

    if exclude is not None:
        exclude = np.array(sorted(set(exclude)))  # Remove duplicates, if any
        # Check that exclude is not specified if OVO
        if ref_group is not None:
            logger.warning(
                "`exclude` is not relevant when a reference group is specified. Excluded groups will be ignored."
            )
            exclude = None
        else:
            # Check that no excluded group is in the subset of groups to test, if specified
            if group_subset is not None:
                if (_m := np.isin(exclude, group_subset)).any():  # This is Ok fast bc both small
                    excluded_in_subset = np.array(exclude)[_m]
                    raise ValueError(
                        f"Groups {excluded_in_subset} are listed in both `group_subset` and `exclude`. Please remove them from one of the two lists."
                    )
            # Check that no group listed in exclude is absent from the .obs group labels
            if not (_m := np.isin(exclude, unique_groups)).all():
                missing_groups = np.array(exclude)[~_m]
                logger.warning(
                    f"Groups {missing_groups} are listed in `exclude` but are not present in the .obs group labels. They will be ignored."
                )
            # Check that at least two groups remain after exclusion, otherwise OVR test is not defined
            if np.isin(unique_groups, exclude).sum() >= unique_groups.size - 1:
                raise ValueError("Less than 2 groups left to compare. Remove some groups from `exclude`.")

    if group_subset is not None:
        group_subset = np.array(sorted(set(group_subset)))  # Remove duplicates, if any

        if ref_group is not None and ref_group not in group_subset:
            group_subset = np.append(group_subset, ref_group)
            logger.warning(
                f"Reference group `{ref_group}` was not included in `group_subset`, but it is required for testing. It has been automatically added to `group_subset`."
            )

        # Check that at least one group is indicated
        if not group_subset.size:
            raise ValueError("Group subset can not be empty.")

        # Check that all groups in group_subset are present in the .obs group labels
        if not (_m := np.isin(group_subset, unique_groups)).all():
            missing_groups = np.array(group_subset)[~_m]
            raise ValueError(
                f"Groups {missing_groups} are listed in `group_subset` but are not present in the .obs group labels. Please remove them from `group_subset` or correct the group labels."
            )

    return group_subset, exclude


def encode_and_count_groups(
    groups: np.ndarray, ref_group: Any, group_subset: Iterable[Any] | None = None, exclude: Iterable[Any] | None = None
) -> tuple[np.ndarray, GroupContainer]:
    """Build the GroupContainer holding all group-related information.

    GroupContainer holds:
    - number of valid groups (i.e. groups that are selected for testing, including the reference group)
    - encoded groups
    - counts (of cell, per group)
    - indices, indptr in a RLE format
    - included cell indices (i.e. indices of cells that are not excluded, if exclude is specified)
    - encoded reference group (control)
    - original group labels

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

    # Check edge cases and contradictions in the input arguments
    group_subset, exclude = sanitize_group_args(
        groups=groups, ref_group=ref_group, group_subset=group_subset, exclude=exclude
    )

    # Gather labels of groups listed in "groups"
    if group_subset is not None:
        is_selected = isin_fast(groups, group_subset)
        # Put non selected groups last, their pvalues will not be computed and their placeholders trimmed
        group_indices = np.lexsort((groups, ~is_selected))
    else:
        group_indices = np.argsort(groups, kind="stable")

    sorted_groups = groups[group_indices]

    # Gather indices of non-excluded groups
    if exclude is not None and ref_group is None:
        # The strategy is simple: discard cells of excluded groups, and all other fields
        # (counts, indices, encoded_groups, etc) are defined w.r.t this new "filtered" X.
        # Note that filtering happens inside the kernels using included_cell_indices.
        exclusion_mask = isin_fast(groups, exclude)
        sorted_mask = exclusion_mask[group_indices]
        sorted_groups = sorted_groups[~sorted_mask]
        group_indices = group_indices[~sorted_mask]
        groups = groups[~exclusion_mask]
        included_cell_indices, *_ = np.where(~exclusion_mask)
    else:
        included_cell_indices = np.arange(groups.size, dtype=np.uint64)
    change_idx = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    group_counts = np.diff(np.r_[0, change_idx, sorted_groups.size]).astype(np.uint64)

    # Find unique groups
    unique_groups = sorted_groups[np.r_[0, change_idx]]

    # Encode groups as integers
    # Note: can no longer use searchsorted because we eventually relied on lexsort.
    # We could use searchlexsorted, if it existed.
    label_to_int = {label: i for i, label in enumerate(unique_groups)}
    encoded_groups = np.array([label_to_int[label] for label in groups], dtype=np.uint64)

    # Build indptr
    group_indptr = np.cumsum(np.insert(group_counts, 0, 0)).astype(np.uint64)

    # Count selected groups
    n_selected_groups = (
        np.isin(unique_groups, group_subset).sum().item() if group_subset is not None else unique_groups.size
    )

    return unique_groups, GroupContainer(
        n_selected_groups=np.uint64(n_selected_groups),
        encoded_groups=encoded_groups,
        counts=group_counts.astype(np.uint64),
        indices=group_indices.astype(np.uint64),
        included_cell_indices=included_cell_indices.astype(np.uint64),
        indptr=group_indptr,
        encoded_ref_group=(
            -1 if ref_group is None else label_to_int[ref_group]
        ),  # Weirdly enough, this must be -1 and not None, otherwise Numba fails to compile various functions, especially branching
    )
