from illico.asymptotic_wilcoxon import asymptotic_wilcoxon

# Import kernel modules to trigger decorator registration
# These imports must come after the registry definitions above
from illico.ovo import (  # noqa: E402, F401
    csc_ovo_mwu_kernel_over_contiguous_col_chunk,
    csr_ovo_mwu_kernel_over_contiguous_col_chunk,
    dense_ovo_mwu_kernel_over_contiguous_col_chunk,
)
from illico.ovr import (  # noqa: E402, F401
    csc_ovr_mwu_kernel_over_contiguous_col_chunk,
    csr_ovr_mwu_kernel_over_contiguous_col_chunk,
    dense_ovr_mwu_kernel_over_contiguous_col_chunk,
)

# Now register the Rust kernels
from illico.rust_backend import (  # noqa: E402, F401
    csc_ovo_mwu_kernel_over_contiguous_col_chunk_rust,
    csc_ovr_mwu_kernel_over_contiguous_col_chunk_rust,
    csr_ovo_mwu_kernel_over_contiguous_col_chunk_rust,
    csr_ovr_mwu_kernel_over_contiguous_col_chunk_rust,
    dense_ovo_over_contiguous_col_chunk_rust,
    dense_ovr_over_contiguous_col_chunk_rust,
)
from illico.utils.registry import (
    KernelDataFormat,
    Test,
    rs_dispatcher_registry,
)

rs_dispatcher_registry.register(Test.OVO, KernelDataFormat.DENSE)(dense_ovo_over_contiguous_col_chunk_rust)
rs_dispatcher_registry.register(Test.OVR, KernelDataFormat.DENSE)(dense_ovr_over_contiguous_col_chunk_rust)
rs_dispatcher_registry.register(Test.OVO, KernelDataFormat.CSC)(csc_ovo_mwu_kernel_over_contiguous_col_chunk_rust)
rs_dispatcher_registry.register(Test.OVO, KernelDataFormat.CSR)(csr_ovo_mwu_kernel_over_contiguous_col_chunk_rust)
rs_dispatcher_registry.register(Test.OVR, KernelDataFormat.CSC)(csc_ovr_mwu_kernel_over_contiguous_col_chunk_rust)
rs_dispatcher_registry.register(Test.OVR, KernelDataFormat.CSR)(csr_ovr_mwu_kernel_over_contiguous_col_chunk_rust)


__all__ = ["asymptotic_wilcoxon"]
