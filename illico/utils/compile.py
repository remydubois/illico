import os
import time
from typing import Any

from loguru import logger
from numba import types

from illico.utils.groups import GroupContainer
from illico.utils.registry import DataHandler, Test, dispatcher_registry


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
    GroupContainerType = types.NamedTuple(
        [types.int64[::1], types.int64[::1], types.int64[::1], types.int64[::1], types.int64], GroupContainer
    )

    # This input signature corresponds to: lower bound, upper bvound, group container, is_log1p, use_continuity
    common_sig = (
        types.int64,
        types.int64,
        GroupContainerType,
        types.boolean,
        types.boolean,
        types.boolean,
        types.string,
    )
    # This is the output: three float64 2D arrays
    out_sig = types.UniTuple(types.float64[:, ::1], 3)

    input_type = data_handler.input_signature()
    if reference is None:
        test_type = Test.OVR
    else:
        test_type = Test.OVO
    dispatcher = dispatcher_registry.get(test_type, data_handler.kernel_data_format())
    sig = out_sig(input_type, *common_sig)

    s = time.time()
    dispatcher.compile(sig)
    e = time.time()
    logger.trace(
        f"Precompilation of {data_handler.kernel_data_format().value}-{test_type.value} dispatcher took {e - s:.1f}s"
    )
    return dispatcher
