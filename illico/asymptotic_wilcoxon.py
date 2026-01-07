import gc
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
    dispatcher_registry,
)

# from illico.utils.math import _warn_log1p

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
):
    """Delayed operator. Not user-facing."""
    if group_container.encoded_ref_group == -1:
        test = Test.OVR
    else:
        test = Test.OVO
    # Grab the adapted kernel
    dispatcher = dispatcher_registry.get(test, data_handler.kernel_data_format())

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
    pvalues, statistics, fold_change = dispatcher(
        X,
        *bounds,
        group_container,
        is_log1p,
        use_continuity,
        tie_correct,
        alternative,
    )
    return (pvalues, statistics, fold_change), (lb, ub)


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
    layer: str | None = None,
    precompile: bool = True,
):
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
    layer : str or None, default=None
        Layer in `adata.layers` to use for the data. If `None`, uses `adata.X`.
    precompile : bool, default=True
        Whether to precompile necessary functions for performance. It is recommended to set this to `True`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with MultiIndex (pert, feature) containing columns:
        - 'p_value': P-value from the Mann-Whitney test
        - 'statistic': Test statistic (U-statistic)
        - 'fold_change': Fold change between groups

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
        _precompile(data_handler, reference)

    # Process the groups information
    raw_groups = adata.obs[group_keys].tolist()
    unique_raw_groups, group_container = encode_and_count_groups(groups=raw_groups, ref_group=reference)
    logger.info(
        f"Found {group_container.counts.size} unique groups (min size: {group_container.counts.min()} cells; max size: {group_container.counts.max()} cells), with reference group: {reference}"
    )
    _, n_genes = X.shape

    # Allocate the results dataframes
    cols = pd.Series(adata.var_names, name="feature", dtype=str)
    rows = pd.Series(unique_raw_groups, name="pert", dtype=str)
    results = np.empty((len(rows), len(cols), 3), dtype=np.float64)

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
            for (pv, ustat, fc), (lb, ub) in pool(
                operator(data_handler, lb, ub, group_container, is_log1p, use_continuity, alternative, tie_correct)
                for lb, ub in iterator
            ):
                results[:, lb:ub, 0] = pv
                results[:, lb:ub, 1] = ustat
                results[:, lb:ub, 2] = fc
                pbar.update(group_container.counts.size * (ub - lb))

                # Cleanup memory
                del pv, ustat, fc
                gc.collect()

        # Return a pd.DataFrame to index results
        results = pd.DataFrame(
            data=results.reshape(-1, 3),
            index=pd.MultiIndex.from_product([rows, cols], names=["pert", "feature"]),
            columns=["p_value", "statistic", "fold_change"],
        )

    return results
