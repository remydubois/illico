import numpy as np
from loguru import logger

from illico.utils.groups import GroupContainer
from illico.utils.registry import data_handler_registry


def log_memory_usage(data_handler, grpc: GroupContainer, batch_size: int, n_threads: int) -> None:
    """Log estimated memory usage of the whole routine.

    This function simply computes estimated memory footprint of the DE genes routine.

    Parameters
    ----------
    data_handler : DataHandler
    grpc : GroupContainer
        GroupContainer
    batch_size : int
        Batch size used for processing genes
    n_threads : int
        Number of threads used for processing genes

    Author: Rémy Dubois
    """
    X = data_handler.data
    data_handler = data_handler_registry.get(X)
    # Results hold fold change, p-values, statistics
    results_fp = grpc.counts.size * X.shape[1] * 3 * np.float64().nbytes
    # Instead of just doing batch_size*n_cells, we scale it by the total object size so that sparsity is taken into account,
    # and scale it by number of threads running concurrently
    proc_fp = batch_size / X.shape[1] * n_threads * data_handler.footprint()
    total_fp = results_fp + proc_fp
    logger.trace(
        f"Estimated RAM footprint: {total_fp / (1000**3):.2f} GB. (This includes {results_fp / (1000**3):.3f} GB for results and {proc_fp / (1000**3):.3f} GB for processing data.)"
    )
    return total_fp
