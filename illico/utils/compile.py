from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger
from numba import types

from illico.ovo.sparse_ovo import single_group_sparse_ovo_mwu_kernel
from illico.utils.groups import GroupContainer
from illico.utils.registry import (
    DaskArrayDataHandler,
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
    if (
        data_handler.is_lazy
        and data_handler.kernel_data_format() is KernelDataFormat.CSR
        and test_type is Test.OVO
        and not isinstance(data_handler, DaskArrayDataHandler)
    ):
        # This is the special lazy CSR OVO scenario
        dispatcher = single_group_sparse_ovo_mwu_kernel
        # The input to this dispatcher is CSC, not CSR, because data is converted in process_group
        # Here, take advantage of the fact that CSCMatrix and CSRMatrix hold the exact same attributes
        csc_sig = types.NamedTuple([*data_handler.input_signature()], CSCMatrix)
        input_sig = (csc_sig, csc_sig, types.boolean, types.boolean, types.string)
        # This kernel does not return fold change
        out_sig = types.UniTuple(types.float64[::1], 3)
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
