import anndata as ad
import numpy as np
from loguru import logger
from scipy import sparse

from illico.utils.groups import GroupContainer


def _log_memory_usage(adata: ad.AnnData, grpc: GroupContainer, batch_size: int, n_threads: int) -> None:
    """Log estimated memory usage of the whole routine.

    This function simply computes estimated memory footprint of the DE genes routine.

    Parameters
    ----------
    adata : ad.AnnData
        Input data
    grpc : GroupContainer
        GroupContainer
    batch_size : int
        Batch size used for processing genes
    n_threads : int
        Number of threads used for processing genes

    Author: Rémy Dubois
    """
    X = adata.X
    # Results hold fold change, p-values, statistics
    results_fp = grpc.counts.size * adata.n_vars * 3 * np.float64().nbytes
    if isinstance(X, np.ndarray):
        proc_fp = batch_size / X.shape[1] * n_threads * X.nbytes
    elif isinstance(X, sparse.spmatrix):
        proc_fp = batch_size / X.shape[1] * n_threads * (X.data.nbytes + X.indptr.nbytes + X.indices.nbytes)
    total_fp = results_fp + proc_fp
    logger.trace(
        f"Estimated RAM footprint: {total_fp / (1000**3):.2f} GB. (This includes {results_fp / (1000**3):.3f} GB for results and {proc_fp / (1000**3):.3f} GB for processing data.)"
    )
