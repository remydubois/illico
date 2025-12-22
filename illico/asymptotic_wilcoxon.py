import gc
import math

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy import sparse
from tqdm.auto import tqdm

from illico.ovo import ovo_mwu_over_contiguous_col_chunk
from illico.ovr import ovr_mwu_over_col_contiguous_chunk
from illico.utils.compile import _precompile
from illico.utils.groups import encode_and_count_groups
from illico.utils.memory import _log_memory_usage
from illico.utils.ranking import check_indices_sorted_per_parcel

# from illico.utils.math import _warn_log1p

__all__ = ["asymptotic_wilcoxon"]


def asymptotic_wilcoxon(
    adata: ad.AnnData,
    is_log1p: bool,
    group_keys: str,
    reference_group: str | None = None,
    n_threads: int = 1,
    batch_size: int = 256,
    alternative: str = "two-sided",
    use_continuity: bool = True,
    layer: str | None = None,
    precompile: bool = True,
):
    """Perform asymptotic Mann-Whitney tests for differential gene expression.

    Mann-Whitney test is the same as Wilcoxon rank-sum test.

    This function takes as input an AnnData object of shape (n-cells, n-genes) with a group
    (e.g., perturbation) variable stored in .obs. It performs either one-versus-rest (OVR) or
    one-versus-one (OVO) Wilcoxon-Mann-Whitney tests for each gene, depending on whether a
    reference group is provided.

    Parameters
    ----------
    adata:
        Annotated data matrix of shape (n-cells, n-genes).
    is_log1p
        Whether the data is log1p transformed.
    group_keys
        Key in `adata.obs` specifying the group variable.
    reference_group
        Name of the reference group for OVO tests. If `None`, OVR tests are performed.
    n_threads
        Number of threads to use for parallel computation.
    batch_size
        Number of genes to process in each batch.
    alternative
        Type of alternative hypothesis. One of 'two-sided', 'less', or 'greater'.
    use_continuity
        Whether to apply continuity correction.
    layer
        Layer in `adata.layers` to use for the data. If `None`, uses `adata.X`.
    precompile
        Whether to precompile necessary functions for performance.

    Returns
    -------
    pd.DataFrame
        A DataFrame with MultiIndex (group, gene) containing p-values, statistics, and fold changes.

    Author: Rémy Dubois
    """
    # TODO: add a sparsity warning inviting user to convert to sparse if possible
    # TODO: rename group_keys

    if layer is not None:
        logger.info(f"Using layer '{layer}' for differential expression.")
        X = adata.layers[layer]
    else:
        X = adata.X

    # Check that the input CSR is sorted.
    if isinstance(X, sparse.csr_matrix):
        if not check_indices_sorted_per_parcel(X.indices, X.indptr):
            raise ValueError(
                "Input data matrix indices are not sorted. This is very unusual and may lead to incorrect results. "
                "This can be the result of operations like `adata[:, np.random.choice(…)]` that do not preserve sorting."
                "Please make sure that indices used to chunk the adata or the expression matrix have been sorted "
                "prior to computing DE genes."
            )

    if precompile:
        _precompile(X, reference_group)

    # Process the groups information
    raw_groups = adata.obs[group_keys].tolist()
    unique_raw_groups, group_container = encode_and_count_groups(groups=raw_groups, ref_group=reference_group)
    logger.info(
        f"Found {group_container.counts.size} unique groups (min size: {group_container.counts.min()} cells; max size: {group_container.counts.max()} cells), with reference group: {reference_group}"
    )
    _, n_genes = X.shape

    # Allocate the results dataframes
    cols = pd.Series(adata.var_names, name="feature", dtype=str)
    rows = pd.Series(unique_raw_groups, name="pert", dtype=str)
    results = np.empty((len(rows), len(cols), 3), dtype=np.float64)

    # Adapt batch size to leverage multithreading regarding the number of genes, if requested
    batch_size = min(batch_size, math.ceil(n_genes / n_threads))
    logger.trace(f"Using batch size of {batch_size} for {n_threads} threads and {n_genes} genes.")

    # Compute estimated mem footprint
    _log_memory_usage(adata, group_container, batch_size, n_threads)

    # Generate batches of columns
    bounds = np.append(np.arange(0, n_genes, batch_size), n_genes)
    iterator = list(zip(bounds[:-1], bounds[1:]))
    n_tests = n_genes * group_container.counts.size
    logger.trace(f"Performing a total of {n_tests:,d} tests.")
    with Parallel(n_threads, prefer="threads", return_as="generator_unordered") as pool:
        with tqdm(total=n_tests, smoothing=0.0, unit="it", unit_scale=True, unit_divisor=1000) as pbar:
            if reference_group is None:  # ovr use case
                pbar.set_description("Running one-versus-all MannWhitney-U tests")
                op = delayed(lambda *args: (ovr_mwu_over_col_contiguous_chunk(*args), args))
            else:  # ovo use case
                pbar.set_description("Running one-versus-ref MannWhitney-U tests")
                op = delayed(lambda *args: (ovo_mwu_over_contiguous_col_chunk(*args), args))
            for (pv, ustat, fc), args in pool(
                op(X, lb, ub, group_container, is_log1p, use_continuity, alternative) for lb, ub in iterator
            ):
                # progress.update(task, advance=group_container.counts.size * (args[2] - args[1]))  # refresh after processing is done
                pbar.update(group_container.counts.size * (args[2] - args[1]))
                col_chunk = slice(*args[1:3])
                results[:, col_chunk, 0] = pv
                results[:, col_chunk, 1] = ustat
                results[:, col_chunk, 2] = fc

                # Cleanup memory
                del pv, ustat, fc
                gc.collect()

        # Return a pd.DataFrame to index results
        results = pd.DataFrame(
            data=results.reshape(-1, 3),
            index=pd.MultiIndex.from_product([rows, cols], names=["pert", "feature"]),
            columns=["p_value", "statistic", "fold_change"],
        )

        """
            This case is harder than the OVR use case, because:
            1. If input is dense, then it's easy we can slice easily over rows (or group of rows) or columns and parallelize the way we want
            2. If input is CSR, we can easily slice the perturbations, and easily slice contiguous columns
            3. If input is CSC, we can easily slice along the columns, BUT NOT ALONG THE PERTURBATIONS because perturbations are non-contiguous along axis 0

            So the answer is simple: if we want a unified parallelism scheme, it has to be along the columns. Now:
            - For the dense use case, there is no problem.
            - For the sparse use case, the OVO kernel requires data to be CSC, so:
                a) If input is CSC, it means we do overall: big CSC ->  vertical chunk of CSC -> horizontal chunk of vertical chunk of CSC -> test
                b) If input is CSR, it means we do: big CSR  ->  vertical chunk of CSR -> horizontal chunk of vertical chunk of CSR -> CSC -> test
            For a), the second chunking is suboptimal because we chunk along 0 a CSC matrix, but because it has little columns this is fast.
            For b), this is a lot of conversion but I believe they all happen on small matrices so this should be fine.

            Conclusion: runtime on H1 is ~25 seconds for CSC, 25 seconds for CSR so still very nice.
            TO KEEP IN MIND: there is **no** way to optimize any other parallelism than column-based actually, the current idea (parallelizing over the
            perturbations) will be very slow, or require a brand new CSR allocation (which we don't want), to function.
            """
    return results
