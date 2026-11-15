from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger
from numba import types

from illico.ovo.sparse_ovo import _single_group_sparse_ovo_mwu_kernel
from illico.utils.groups import GroupContainer
from illico.utils.registry import (
    DataHandler,
    KernelDataFormat,
    Test,
    nb_dispatcher_registry,
)
from illico.utils.sparse.types import CSCMatrix


def _precompile(data_handler: DataHandler, reference: Any | None):
    """Precompile the CPU dispatcher before the threads start rushing to it.

    Note: a simpler way to do it could be to use a threading.lock to make just the first thread compile, but
    there are still concurrency risks.
    Having this in a separate routine also allows to call it from tests so that memory and speed benchmarks are not impacted by compilation.

    Args:

        reference (Any | None): Reference group

    Raises:
        ValueError: If input data is neither dense, CSC nor CSR.

    Author: Rémy Dubois

    """
    if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1":
        logger.warning("Numba JIT is disabled, skipping precompilation.")
        return

    if reference is None:
        test_type = Test.OVR
    else:
        test_type = Test.OVO
    if data_handler.is_lazy and data_handler.kernel_data_format() is KernelDataFormat.CSR and test_type is Test.OVO:
        # This is the special lazy CSR OVO scenario (backed or Dask), handled by `ovo_lazy_csr_operator`.
        # Compile the exact signature it calls the kernel with, so that worker threads do not compile it.
        dispatcher = _single_group_sparse_ovo_mwu_kernel
        # The input to this dispatcher is CSC, not CSR, because data is converted in `preprocess_group`
        # Here, take advantage of the fact that CSCMatrix and CSRMatrix hold the exact same attributes
        csr_sig = data_handler.input_signature()
        csc_sig = types.NamedTuple([*csr_sig], CSCMatrix)
        # Number of zeros per column of the control group: same dtype as indptr
        n_zeros_sig = csr_sig[2].dtype[::1]
        # Results are written in place into strided slices of the shared results array
        result_slice_sig = types.float64[:]
        input_sig = (
            csc_sig,  # sorted control data
            csc_sig,  # sorted target data
            n_zeros_sig,  # n_zeros_ref
            types.uint32[::1],  # n_uniques_ref
            types.int64[::1],  # ref_block_offsets
            types.int64[::1],  # tie_sums_ref
            types.boolean,  # use_continuity
            types.boolean,  # tie_correct
            types.string,  # alternative
            result_slice_sig,  # p-values
            result_slice_sig,  # z-scores
            result_slice_sig,  # U-statistics
        )
        # Results are written in place
        out_sig = types.none
    else:
        GroupContainerType = types.NamedTuple(
            [
                types.uint64,
                types.uint64[::1],
                types.uint64[::1],
                types.uint64[::1],
                types.uint64[::1],
                types.uint64[::1],
                types.int64,
            ],
            GroupContainer,
        )
        # This input signature corresponds to: lower bound, upper bvound, group container, is_log1p, use_continuity
        input_sig = (
            data_handler.input_signature(),
            types.int64,
            types.int64,
            GroupContainerType,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.string,
        )
        # This is the output: four float64 2D arrays
        out_sig = types.UniTuple(types.float64[:, ::1], 4)

        dispatcher = nb_dispatcher_registry.get(test_type, data_handler.kernel_data_format())

    # Assemble final signature
    sig = out_sig(*input_sig)

    s = time.time()
    dispatcher.compile(sig)
    e = time.time()
    logger.trace(
        f"Precompilation of {data_handler.kernel_data_format().value}-{test_type.value} dispatcher took {e - s:.1f}s"
    )
    return dispatcher
