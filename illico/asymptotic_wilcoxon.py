from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from numba import set_num_threads
from scipy import sparse
from tqdm.auto import tqdm

from illico.ovo import single_group_sparse_ovo_mwu_kernel
from illico.utils.compile import _precompile
from illico.utils.groups import GroupContainer, encode_and_count_groups
from illico.utils.math import compute_batch_bounds
from illico.utils.memory import log_memory_usage
from illico.utils.ranking import (
    _sort_csc_columns_inplace,
    check_indices_sorted_per_parcel,
)
from illico.utils.registry import (
    DataHandler,
    KernelDataFormat,
    Test,
    data_handler_registry,
    nb_dispatcher_registry,
    rs_dispatcher_registry,
)
from illico.utils.scanpy import format_illico_results_for_scanpy
from illico.utils.sparse.csr import csr_sum_along_cols, csr_to_csc

__all__ = ["asymptotic_wilcoxon"]


@delayed
def all_purpose_operator(
    data_handler: DataHandler,
    lb: int,
    ub: int,
    group_container: GroupContainer,
    is_log1p: bool,
    use_continuity: bool,
    alternative: str,
    tie_correct: bool,
    exp_post_agg: bool,
    use_rust: bool,
    results: np.ndarray,
):
    """Delayed operator. Not user-facing.

    This operator is the default operator handling all data formats (Dense, CSC, CSR, in RAM or lazy) and both OVO and
    OVR tests, except one: the OVO test on lazy CSR datasets.

    The reason why the OVO test on lazy CSR datasets is not handled by this operator is that it can be performed in a
    much more efficient way by taking advantage of the fact that the data is in CSR format, and performing the test
    group by group instead of gene by gene. This allows to not load the whole dataset in RAM at once, which would defeat
    the purpose of using a lazy format in the first place. For this specific scenario, a dedicated operator
    `ovo_lazy_csr_operator` has been implemented.

    """
    if group_container.encoded_ref_group == -1:
        test = Test.OVR
    else:
        test = Test.OVO
    # Grab the adapted kernel
    if not use_rust:
        dispatcher = nb_dispatcher_registry.get(test, data_handler.kernel_data_format())
    else:
        dispatcher = rs_dispatcher_registry.get(test, data_handler.kernel_data_format())

    # Quick safety check
    if lb < 0 or ub > data_handler.data.shape[1] or lb > ub:
        raise ValueError(f"Invalid chunk bounds: {(lb, ub)} for data with {data_handler.data.shape[1]} columns.")

    # Fetch the data from disk if in backed mode
    # The reason to be of not applying X[:, lb:ub] in all cases (backed or not) is that if the data is whole in RAM, the CSR chunking is optimized, and
    # if the data is not in RAM, CSR chunking is not implemented
    fetched_data, bounds = data_handler.fetch_cols(lb, ub)
    # Convert to numba-compatible format
    X = data_handler.to_nb(fetched_data)
    # Call the dispatcher
    pvalues, statistics, zscores, fold_change = dispatcher(
        X,
        *bounds,
        group_container,
        is_log1p,
        use_continuity,
        tie_correct,
        exp_post_agg,
        alternative,
    )
    # Copy results into the shared array, do it thread-wise for cleaner garbage collection and speed
    # Note: there might be a little speedup in passing results to the dispatchers and writing in it directly, it would
    # allow to not allocate chunks of results. However, this is would be very non-Rusty as the same Python-managed memory block would be shared across several threads.
    # The copy should be GIL-free
    results[:, lb:ub, 0] = pvalues[: group_container.n_selected_groups, :]
    results[:, lb:ub, 1] = statistics[: group_container.n_selected_groups, :]
    results[:, lb:ub, 2] = zscores[: group_container.n_selected_groups, :]
    results[:, lb:ub, 3] = fold_change[
        : group_container.n_selected_groups, :
    ]  # Technically Rust returns f32, but numpy handles casting here
    return (lb, ub)


def preprocess_group(
    data_handler: DataHandler, grpc: GroupContainer, group_id: int, is_log1p: bool, exp_post_agg: bool
) -> tuple[np.ndarray]:
    """Preprocess the group (chunk of rows) for the OVO test on (lazy) CSR data.

    This function fetches the data (rows) for the given group, computes the summed expression for each gene, and sorts
    the data in CSC format. This function is used in the `ovo_lazy_csr_operator` operator, and is not user-facing.

    Parameters
    ----------
    data_handler : DataHandler
        Data handler for the input data matrix.
    grpc : GroupContainer
        Group container with the encoded groups information.
    group_id : int
        Id of the group to preprocess.
    is_log1p : bool
        Whether the data is log1p transformed. This is used for fold change computation.
    exp_post_agg : bool
        Whether to exponentiate the fold change after aggregation. This is relevant if the input data is log1p.
        See documentation for details. Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and
        exponentiates post aggregation by default.

    Returns
    -------
    tuple[np.ndarray]
        A tuple containing the preprocessed data for the given group:
        - X: The data matrix for the given group, in sorted CSC format.
        - mu: The summed expression for each gene in the given group, used for fold change computation.

    """
    assert data_handler.kernel_data_format() == KernelDataFormat.CSR, "This operator is specific for CSR datasets."
    assert data_handler.is_lazy, "This operator is specific for lazy datasets."

    # Fetch the data from disk
    start = grpc.indptr[group_id]
    end = grpc.indptr[group_id + 1]
    indices = grpc.indices[start:end]
    X = data_handler.fetch_rows(indices)
    X = data_handler.to_nb(X)
    # Compute mean expression for fold change computation
    mu = csr_sum_along_cols(X, expm1=is_log1p & (not exp_post_agg)) / X.shape[0]
    # Convert it to CSC
    X = csr_to_csc(X, include_indices=False)
    # Sort it
    _sort_csc_columns_inplace(X)
    return X, mu


@delayed
def ovo_lazy_csr_operator(
    data_handler: DataHandler,
    grpc: GroupContainer,
    group_id: int,
    X_control: np.ndarray,
    mu_control: np.ndarray,
    is_log1p: bool,
    use_continuity: bool,
    alternative: str,
    tie_correct: bool,
    exp_post_agg: bool,
    use_rust: bool,
    results: np.ndarray,
):
    """Delayed operator for lazy CSR datasets. Not user-facing.

    This operator is specific because instead of operating on chunks of columns, it takes the advantage of the fact that
    the OVO test can also be performed group by group (i.e on chunks of rows). This is useful only in the scenario where
    the data is in lazy CSR format, as it allows to not load the whole dataset in RAM at once, which would defeat the
    purpose of using a lazy format in the first place. In this scenario, the operator takes as input a group id, and the
    pre- computed sorted control and mean expression for this group, and performs OVO tests between this group and the
    reference group for all genes at once. This way, only one chunk of rows is loaded in RAM at once, which is much more
    memory-efficient than loading chunks of columns for CSR matrices.

    """
    assert data_handler.kernel_data_format() == KernelDataFormat.CSR, "This operator is specific for CSR datasets."
    assert data_handler.is_lazy, "This operator is specific for lazy datasets."

    if use_rust:
        raise ValueError("There is no Rust kernel for the lazy CSR format yet.")

    # Grab the adapted kernel, Numba only for now
    dispatcher = single_group_sparse_ovo_mwu_kernel

    # Preprocess this group's cells
    X_tgt, mu_tgt = preprocess_group(data_handler, grpc, group_id, is_log1p, exp_post_agg)

    # Call the dispatcher
    pvalues, statistics, zscores = dispatcher(X_control, X_tgt, use_continuity, tie_correct, alternative)

    # Compute fold change separately as it does not require to load the whole dataset in RAM at once
    fc = np.full(X_tgt.shape[1], fill_value=np.inf)
    mask = (mu_control != 0) & np.isfinite(mu_control)
    if is_log1p and exp_post_agg:
        fc[mask] = np.expm1(mu_tgt[mask]) / np.expm1(mu_control[mask])
    else:
        fc[mask] = mu_tgt[mask] / mu_control[mask]

    # Assign results to the shared array
    results[group_id, :, 0] = pvalues
    results[group_id, :, 1] = statistics
    results[group_id, :, 2] = zscores
    results[group_id, :, 3] = fc
    return group_id


def asymptotic_wilcoxon(
    adata: ad.AnnData,
    is_log1p: bool,
    group_keys: str,
    reference: str | None = None,
    n_threads: int = 1,
    batch_size: int | Literal["auto"] = "auto",
    alternative: str = "two-sided",
    use_continuity: bool = True,
    tie_correct: bool = True,
    exp_post_agg: bool = False,
    layer: str | None = None,
    groups: list[str] | None = None,
    exclude_from_ovr: list[str] | None = None,
    precompile: bool = True,
    use_rust: bool = True,
    return_as_scanpy: bool = False,
    n_genes: int | None = None,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
) -> pd.DataFrame | dict:
    """Perform asymptotic Mann-Whitney tests for differential gene expression.

    Mann-Whitney test is the same as Wilcoxon rank-sum test.
    This function takes as input an AnnData object of shape (n_cells, n_genes) with a group
    (e.g., perturbation) variable stored in .obs. It performs either one-versus-rest (OVR) or
    one-versus-one (OVO) Wilcoxon-Mann-Whitney tests for each gene, depending on whether a
    reference group is provided.

    It supports all the combinations of data formats (dense, sparse CSR, sparse CSC, in RAM or lazy)
    and tests (OVO, OVR), except one: the OVR test on lazy CSR datasets. The reason why this combo is
    not supported is because OVR test requires each column to be entirely in RAM as it has to be sorted,
    but CSR matrices requires the entirety of the `.indices` to be in RAM to chunk vertically, which
    defeats the purpose of using a lazy format in the first place.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix of shape (n_cells, n_genes).
    is_log1p : bool
        Whether the data is log1p transformed.
    group_keys : str
        Key in `adata.obs` specifying the group variable.
    reference : str or None, default=None
        Name of the reference group for OVO tests. If `None`, OVR tests are performed.
    n_threads : int, default=1
        Number of threads to use for parallel computation.
    batch_size : int or "auto", default="auto"
        Number of genes to process in each batch. If "auto", automatically determines
        optimal batch size aiming for approximately 256 genes per chunk.
    alternative : str, default="two-sided"
        Type of alternative hypothesis. One of 'two-sided', 'less', or 'greater'.
    use_continuity : bool, default=True
        Whether to apply continuity correction.
    tie_correct : bool, default=True
        Whether to apply tie correction in the test statistic.
    exp_post_agg : bool, default=False
        Whether to exponentiate the fold change after aggregation. This is relevant if the input data is log1p. See documentation for details.
        Note that `scanpy.rank_genes_groups` assumes the data to be log1p, and exponentiates post aggregation by default.
    layer : str or None, default=None
        Layer in `adata.layers` to use for the data. If `None`, uses `adata.X`.
    groups : list of values or None, default=None
        Subset of groups to test. If `None`, tests all groups. This arguments serves the same purpose as scanpy's `groups` argument in `rank_genes_groups`.
        It is used to filter which groups to compare against the reference in the OVO scenario, or which groups to compare against the rest in the OVR scenario.
        Note that in the OVR scenario, each comparison still happens against the entirety of the other groups, not just the ones listed in this argument.
        Note that in the OVO scenario, the reference group is automatically added.
        Order of the values in this list has no impact on the end results, duplicates will be trimmed away.
    exclude_from_ovr : list of values or None, default=None
        Subset of groups to exclude from the rest group in the OVR scenario (when reference=None). This argument is ignored in the OVO scenario.
        This can be useful if, for instance, one of the groups is corrupted and contains meaningless data, and we don't want it to be part of the comparisons in the OVR scenario.
        Order of the values in this list has no impact on the end results, duplicates will be trimmed away.
    precompile : bool, default=True
        Whether to precompile necessary functions for performance. It is recommended to set this to `True`.
    use_rust : bool, default=True
        Whether to use the Rust implementation of the test. If `False`, uses the Numba implementation.
    return_as_scanpy : bool, default=False
        Whether to return results in a format compatible with Scanpy's `rank_genes_groups` function.
        If yes, the output is a dictionary that can be attached to the `adata` object like this:
        `adata.uns['rank_genes_groups'] = asymptotic_wilcoxon(..., return_as_scanpy=True)`
    n_genes : int or None, default=None
        Number of top genes to return per group, sorted by z-score. If `None`, returns all genes. This is relevant only if `return_as_scanpy=True`,
        as Scanpy's `rank_genes_groups` function expects the results to be sorted by significance. If `return_as_scanpy=False`, the results are
        not sorted and `n_genes` is ignored.
    corr_method: str, default="benjamini-hochberg"
        Method to use for multiple testing correction. One of 'benjamini-hochberg' or 'bonferroni'.


    Returns
    -------
    Either one of pd.DataFrame or Dict, depending on the value of `return_as_scanpy`:
        A DataFrame with MultiIndex (pert, feature) containing columns:
        - 'p_value': P-value from the Mann-Whitney test
        - 'statistic': Test statistic (U-statistic)
        - 'z-scores': Test z-score
        - 'fold_change': Fold change between groups
        Or a dictionary formatted for Scanpy's `rank_genes_groups` results, containing:
        - 'params': Dictionary of parameters used for the test
        - 'names': Record array of gene names sorted by significance for each group
        - 'scores': Record array of test statistics sorted by significance for each group
        - 'pvals': Record array of p-values sorted by significance for each group
        - 'pvals_adj': Record array of adjusted p-values sorted by significance for each group
        - 'logfoldchanges': Record array of log2 fold changes sorted by significance for each group

    Raises
    ------
    ValueError
        If input data matrix indices are not sorted (for sparse CSR matrices).
        If batch_size is not 'auto' or an integer.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # Create example AnnData object
    >>> n_cells, n_genes = 1000, 500
    >>> X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    >>> obs = pd.DataFrame({'cell_type': np.random.choice(['A', 'B', 'C'], n_cells)})
    >>> var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    >>> adata = ad.AnnData(X=X, obs=obs, var=var)
    >>>
    >>> # Perform one-versus-rest tests
    >>> results = asymptotic_wilcoxon(
    ...     adata,
    ...     is_log1p=False,
    ...     group_keys='cell_type',
    ...     n_threads=4
    ... )
    >>> print(results.head())
    >>>
    >>> # Perform one-versus-one tests against reference
    >>> results_ovo = asymptotic_wilcoxon(
    ...     adata,
    ...     is_log1p=False,
    ...     group_keys='cell_type',
    ...     reference='A',
    ...     n_threads=4,
    ...     alternative='greater'
    ... )
    >>>
    >>> # Filter significant results
    >>> significant = results[results['p_value'] < 0.05]
    >>> print(f"Found {len(significant)} significant tests")

    Notes
    -----
    The function automatically handles both dense and sparse matrices. For sparse CSR matrices,
    indices must be sorted per row to ensure correct results.

    Author: Rémy Dubois

    """

    # Get expression matrix
    if layer is not None:
        logger.info(f"Using layer '{layer}' for differential expression.")
        X = adata.layers[layer]
    else:
        X = adata.X
    data_handler = data_handler_registry.get(X)

    # Check that the input CSR is sorted.
    if isinstance(X, sparse.csr_matrix):
        set_num_threads(n_threads)  # Set the number of threads for Numba to use in the check function
        if not check_indices_sorted_per_parcel(X.indices, X.indptr):
            raise ValueError(
                "Input data matrix indices are not sorted. This is very unusual and may lead to incorrect results. "
                "This can be the result of operations like `adata[:, np.random.choice(…)]` that do not preserve sorting."
                "Please make sure that indices used to chunk the adata or the expression matrix have been sorted "
                "prior to computing DE genes."
            )

    # Precompile if requested
    if precompile:
        if use_rust:
            logger.info("No precompilation needed for Rust kernels.")
        else:
            # TODO: this branch is not reached if use_rust and special lazy ovo CSR, yet it should as Numba is used in this scenario.
            _precompile(data_handler, reference)

    # Process the groups information
    unique_raw_groups, group_container = encode_and_count_groups(
        groups=adata.obs[group_keys].values,
        ref_group=reference,
        group_subset=groups,
        exclude=exclude_from_ovr,
    )
    logger.info(
        f"Found {group_container.counts.size} unique groups (min size: {group_container.counts.min()} cells; "
        f"max size: {group_container.counts.max()} cells), with reference group: {reference}"
    )
    _, n_genes_total = X.shape

    # Allocate the results dataframes
    cols = pd.Series(adata.var_names, name="feature", dtype=str)
    rows = pd.Series(unique_raw_groups[: group_container.n_selected_groups], name="pert", dtype=str)
    results = np.empty((len(rows), len(cols), 4), dtype=np.float64)

    # Go through all the possible combinations
    n_tests = n_genes_total * group_container.counts.size
    logger.trace(f"Performing a total of {n_tests:,d} tests.")
    with Parallel(n_threads, prefer="threads", return_as="generator_unordered") as pool:
        with tqdm(total=n_tests, smoothing=0.0, unit="it", unit_scale=True, unit_divisor=1000) as pbar:
            if (
                data_handler.is_lazy
                and data_handler.kernel_data_format() is KernelDataFormat.CSR
                and reference is not None
            ):
                if use_rust:
                    use_rust = False
                    logger.info("There is no Rust kernel for the lazy CSR format. Falling back to the Numba kernel.")

                logger.trace(
                    f"Performing OVO test on lazy-loaded CSR data. Processing data group by group with {n_threads} threads."
                )

                # Preprocess control cells
                X_ctrl, mu_ctrl = preprocess_group(
                    data_handler, group_container, group_container.encoded_ref_group, is_log1p, exp_post_agg
                )

                # Process all perturbations one by one
                for _ in pool(ovo_lazy_csr_operator(data_handler, group_container, grp_id, X_ctrl,  mu_ctrl, is_log1p, use_continuity, alternative, tie_correct, exp_post_agg, use_rust, results) for grp_id in range(group_container.counts.size)): # fmt: skip
                    pbar.update(adata.n_vars)
            else:
                # Compute the batch bounds for each thread
                iterator, batch_size = compute_batch_bounds(n_genes_total, batch_size, n_threads)
                logger.trace(
                    f"Processing {n_genes_total} genes through {len(iterator)} batches with {n_threads} threads."
                )

                # Compute estimated mem footprint
                _ = log_memory_usage(data_handler, group_container, batch_size, n_threads)

                # Process chunks of columns one by one
                for lb, ub in pool(all_purpose_operator(data_handler, lb, ub, group_container, is_log1p, use_continuity, alternative, tie_correct, exp_post_agg, use_rust, results) for lb, ub in iterator):  # fmt: skip
                    pbar.update(group_container.counts.size * (ub - lb))

    if not return_as_scanpy:
        if n_genes is not None:
            logger.warning(
                "Argument `n_genes` is ignored when `return_as_scanpy=False`, as the results are not sorted. Returning all genes."
            )
        # Return a pd.DataFrame to index results
        results = pd.DataFrame(
            data=results.reshape(-1, 4),
            index=pd.MultiIndex.from_product([rows, cols], names=["pert", "feature"]),
            columns=["p_value", "statistic", "z_score", "fold_change"],
        )
    else:
        # Return a dict formatted for Scanpy's rank_genes_groups results
        results = format_illico_results_for_scanpy(
            adata=adata,
            unique_groups=unique_raw_groups[: group_container.n_selected_groups],
            reference=reference,
            group_keys=group_keys,
            layer=layer,
            values=results,
            n_genes=n_genes,
            corr_method=corr_method,
        )

    return results
