# from abc import ABC, abstractmethod
from collections import namedtuple

# import anndata as ad
# import h5py
# import numpy as np
# from scipy import sparse

TestResults = namedtuple("TestResults", ["statistic", "pvalue"])
# CSRMatrix = namedtuple("CSRMatrix", ["data", "indices", "indptr", "shape"])
# CSCMatrix = namedtuple("CSCMatrix", ["data", "indices", "indptr", "shape"])


# def scipy_to_nb(
#     obj: np.ndarray | sparse.csc_matrix | sparse.csr_matrix,
# ) -> np.ndarray | CSRMatrix | CSCMatrix:
#     """Convert python object into Numba-compat holder

#     Returns
#     -------
#     np.ndarray | CSRMatrix | CSCMatrix
#         The Numba-compatible object

#     Raises
#     ------
#     ValueError
#         If input is neither dense, csc or csr.

#     Author: Rémy Dubois
#     """
#     if isinstance(obj, np.ndarray):
#         return obj
#     elif isinstance(obj, sparse.csc_matrix):
#         return CSCMatrix(obj.data, obj.indices, obj.indptr, obj.shape)
#     elif isinstance(obj, sparse.csr_matrix):
#         return CSRMatrix(obj.data, obj.indices, obj.indptr, obj.shape)
#     else:
#         raise ValueError(type(obj))

# def fetch_if_backed(f):
#     """
#     Decorator to fetch chunks of backed dataset in RAM for backed mode.
#     Here, f is either:
#     - ovo_mwu_over_contiguous_col_chunk
#     - ovr_mwu_over_contiguous_col_chunk
#     """

#     def g(
#         X,
#         chunk_lb,
#         chunk_ub,
#         group_container,
#         is_log1p,
#         use_continuity,
#         alternative):
#         if isinstance(X, h5py.Dataset):
#             X = X[:, chunk_lb:chunk_ub]
#             chunk_ub = chunk_ub - chunk_lb
#             chunk_lb = 0
#         elif isinstance(X, ad._core.sparse_dataset._CSCDataset):
#             X = X[:, chunk_lb:chunk_ub]
#             chunk_ub = chunk_ub - chunk_lb
#             chunk_lb = 0
#         elif isinstance(X, ad._core.sparse_dataset._CSRDataset):
#             raise ValueError("CSR backed AnnData is not supported.")
#         return f(
#             X=X,
#             chunk_lb=chunk_lb,
#             chunk_ub=chunk_ub,
#             group_container=group_container,
#             is_log1p=is_log1p,
#             use_continuity=use_continuity,
#             alternative=alternative)

#     return g
