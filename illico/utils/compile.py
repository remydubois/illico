import os
import time
from typing import Any

import numpy as np
import scipy.sparse as py_sparse
from loguru import logger
from numba import types

from illico.ovo.dense_ovo import dense_ovo_mwu_kernel_over_contiguous_col_chunk
from illico.ovo.sparse_ovo import (
    csc_ovo_mwu_kernel_over_contiguous_col_chunk,
    csr_ovo_mwu_kernel_over_contiguous_col_chunk,
)
from illico.ovr.dense_ovr import dense_ovr_mwu_kernel_over_contiguous_col_chunk
from illico.ovr.sparse_ovr import (
    csc_ovr_mwu_kernel_over_contiguous_col_chunk,
    csr_ovr_mwu_kernel_over_contiguous_col_chunk,
)
from illico.utils.groups import GroupContainer
from illico.utils.type import CSCMatrix, CSRMatrix

DISPATCHER_MAP = {
    ("dense", "ovr"): dense_ovr_mwu_kernel_over_contiguous_col_chunk,
    ("csc", "ovr"): csc_ovr_mwu_kernel_over_contiguous_col_chunk,
    ("csr", "ovr"): csr_ovr_mwu_kernel_over_contiguous_col_chunk,
    ("dense", "ovo"): dense_ovo_mwu_kernel_over_contiguous_col_chunk,
    ("csc", "ovo"): csc_ovo_mwu_kernel_over_contiguous_col_chunk,
    ("csr", "ovo"): csr_ovo_mwu_kernel_over_contiguous_col_chunk,
}


def _precompile(input_data: np.ndarray | py_sparse.csc_matrix | py_sparse.csr_matrix, reference_group: Any | None):
    """Precompile the CPU dispatcher before the threads start rushing to it.

    Note: a simpler way to do it could be to use a threading.lock to make just the first thread compile, but
    there are still concurrency risks.
    Having this in a separate routine also allows to call it from tests so that memory and speed benchmarks are not impacted by compilation.

    Args:
        input_data (np.ndarray | py_sparse.csc_matrix | py_sparse.csr_matrix): Input data
        reference_group (Any | None): Reference group

    Raises:
        ValueError: If input data is neither dense, CSC nor CSR.

    Author: Rémy Dubois
    """
    if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1":
        logger.warning("Numba JIT is disabled, skipping precompilation.")
        return
    GroupContainerType = types.NamedTuple(
        [types.int64[::1], types.int64[::1], types.int64[::1], types.int64[::1], types.int64], GroupContainer
    )

    # This input signature corresponds to: lower bound, upper bvound, group container, is_log1p, use_continuity
    common_sig = (types.int64, types.int64, GroupContainerType, types.boolean, types.boolean)
    # This is the output: three float64 2D arrays
    out_sig = types.UniTuple(types.float64[:, ::1], 3)

    input_dtype = str(input_data.dtype)  # Sparse or dense expose .dtype
    if isinstance(input_data, np.ndarray):
        if input_data.flags["C_CONTIGUOUS"]:
            input_type = getattr(types, input_dtype)[:, ::1]
        else:
            input_type = getattr(types, input_dtype)[:, :]
        data_format = "dense"
    elif isinstance(input_data, py_sparse.spmatrix):
        # TODO: are inputs necessarily C-contiguous?
        input_type = getattr(types, input_dtype)[::1]
        indices_type = getattr(types, str(input_data.indices.dtype))[::1]
        indptr_type = getattr(types, str(input_data.indptr.dtype))[::1]
        if isinstance(input_data, py_sparse.csc_matrix):
            input_type = types.NamedTuple(
                [input_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSCMatrix
            )
            data_format = "csc"
        elif isinstance(input_data, py_sparse.csr_matrix):
            input_type = types.NamedTuple(
                [input_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSRMatrix
            )
            data_format = "csr"
        else:
            raise ValueError(f"Unsupported sparse matrix type {type(input_data)}")
    else:
        raise ValueError(f"Unsupported input data type {type(input_data)}")

    if reference_group is None:
        test_type = "ovr"
    else:
        test_type = "ovo"

    dispatcher = DISPATCHER_MAP[(data_format, test_type)]
    sig = out_sig(input_type, *common_sig)

    s = time.time()
    dispatcher.compile(sig)
    e = time.time()
    logger.trace(f"Precompilation of {data_format}-{test_type} dispatcher took {e - s:.1f}s")
    return dispatcher
