from collections import namedtuple

import numpy as np
from scipy import sparse

TestResults = namedtuple("TestResults", ["statistic", "pvalue"])
CSRMatrix = namedtuple("CSRMatrix", ["data", "indices", "indptr", "shape"])
CSCMatrix = namedtuple("CSCMatrix", ["data", "indices", "indptr", "shape"])


def scipy_to_nb(
    obj: np.ndarray | sparse.csc_matrix | sparse.csr_matrix,
) -> np.ndarray | CSRMatrix | CSCMatrix:
    """Convert python object into Numba-compat holder

    Returns
    -------
    np.ndarray | CSRMatrix | CSCMatrix
        The Numba-compatible object

    Raises
    ------
    ValueError
        If input is neither dense, csc or csr.

    Author: Rémy Dubois
    """
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, sparse.csc_matrix):
        return CSCMatrix(obj.data, obj.indices, obj.indptr, obj.shape)
    elif isinstance(obj, sparse.csr_matrix):
        return CSRMatrix(obj.data, obj.indices, obj.indptr, obj.shape)
    else:
        raise ValueError(type(obj))
