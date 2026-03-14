import math
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy import sparse
from tqdm.auto import tqdm

from illico.utils.compile import _precompile
from illico.utils.groups import GroupContainer, encode_and_count_groups
from illico.utils.memory import log_memory_usage
from illico.utils.ranking import check_indices_sorted_per_parcel
from illico.utils.registry import (
    DataHandler,
    Test,
    data_handler_registry,
    nb_dispatcher_registry,
    rs_dispatcher_registry,
)
from illico.utils.scanpy import format_illico_results_for_scanpy

__all__ = ["asymptotic_wilcoxon"]


@delayed
def operator(
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
    """Delayed operator. Not user-facing."""
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
    fetched_data, bounds = data_handler.fetch(lb, ub)
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
    results[:, lb:ub, 0] = pvalues
    results[:, lb:ub, 1] = statistics
    results[:, lb:ub, 2] = zscores
    results[:, lb:ub, 3] = fold_change  # Technically Rust returns f32, but numpy handles casting here
    return (lb, ub)


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
    precompile: bool = True,
    use_rust: bool = True,
    return_as_scanpy: bool = False,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
) -> pd.DataFrame | dict:
    """Perform asymptotic Mann-Whitney tests for differential gene expression.

    Mann-Whitney test is the same as Wilcoxon rank-sum test.
    This function takes as input an AnnData object of shape (n_cells, n_genes) with a group
    (e.g., perturbation) variable stored in .obs. It performs either one-versus-rest (OVR) or
    one-versus-one (OVO) Wilcoxon-Mann-Whitney tests for each gene, depending on whether a
    reference group is provided.

    It supports in-RAM dense, sparse CSC and sparse CSR matrices, as well as backed dense and sparse CSC matrices.

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
    precompile : bool, default=True
        Whether to precompile necessary functions for performance. It is recommended to set this to `True`.
    use_rust : bool, default=True
        Whether to use the Rust implementation of the test. If `False`, uses the Numba implementation.
    return_as_scanpy : bool, default=False
        Whether to return results in a format compatible with Scanpy's `rank_genes_groups` function.
        If yes, the output is a dictionary that can be attached to the `adata` object like this:
        `adata.uns['rank_genes_groups'] = asymptotic_wilcoxon(..., return_as_scanpy=True)`
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
            _precompile(data_handler, reference)

    # Process the groups information
    unique_raw_groups, group_container = encode_and_count_groups(
        groups=adata.obs[group_keys].values, ref_group=reference
    )
    logger.info(
        f"Found {group_container.counts.size} unique groups (min size: {group_container.counts.min()} cells; max size: {group_container.counts.max()} cells), with reference group: {reference}"
    )
    _, n_genes = X.shape

    # Allocate the results dataframes
    cols = pd.Series(adata.var_names, name="feature", dtype=str)
    rows = pd.Series(unique_raw_groups, name="pert", dtype=str)
    results = np.empty((len(rows), len(cols), 4), dtype=np.float64)

    # Adapt batch size to leverage multithreading regarding the number of genes, if requested
    if n_genes < 256:
        batch_size = n_genes  # No batching for small number of genes
        n_threads = 1  # No multithreading for small number of genes
        iterator = [[0, n_genes]]
    elif isinstance(batch_size, int):
        batch_size = min(batch_size, math.ceil(n_genes / n_threads))
        bounds = np.append(np.arange(0, n_genes, batch_size), n_genes)
        iterator = list(zip(bounds[:-1], bounds[1:]))
    elif batch_size == "auto":
        n_dispatches = max(int(n_genes / 256 / n_threads), 1)  # Aim for approximately 256 genes per chunk
        splits = np.array_split(np.arange(n_genes + 1), indices_or_sections=n_threads * n_dispatches)
        iterator = [split[[0, -1]] for split in splits]
        batch_size = int(np.ceil(n_genes / (n_dispatches * n_threads)))
    else:
        raise ValueError(f"Invalid batch_size value: {batch_size}. Must be 'auto' or an integer.")
    logger.trace(f"Using batch size of {batch_size} for {n_threads} threads and {n_genes} genes.")

    # Compute estimated mem footprint
    _ = log_memory_usage(data_handler, group_container, batch_size, n_threads)

    # Go through all the possible combinations
    n_tests = n_genes * group_container.counts.size
    logger.trace(f"Performing a total of {n_tests:,d} tests.")
    with Parallel(n_threads, prefer="threads", return_as="generator_unordered") as pool:
        with tqdm(total=n_tests, smoothing=0.0, unit="it", unit_scale=True, unit_divisor=1000) as pbar:
            for lb, ub in pool(
                operator(
                    data_handler,
                    lb,
                    ub,
                    group_container,
                    is_log1p,
                    use_continuity,
                    alternative,
                    tie_correct,
                    exp_post_agg,
                    use_rust,
                    results,
                )
                for lb, ub in iterator
            ):
                pbar.update(group_container.counts.size * (ub - lb))

    if not return_as_scanpy:
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
            reference=reference,
            group_keys=group_keys,
            layer=layer,
            values=results,
            corr_method=corr_method,
        )

    return results
