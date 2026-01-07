from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import anndata as ad
import h5py
import numpy as np
from numba import types
from scipy import sparse as py_sparse

from illico.utils.sparse.csc import CSCMatrix
from illico.utils.sparse.csr import CSRMatrix


class Test(Enum):
    OVO = "ovo"
    OVR = "ovr"


class KernelDataFormat(Enum):
    DENSE = "dense"
    CSC = "csc"
    CSR = "csr"


class DispatcherRegistry(dict):
    def register(self, test: Test, data_format: KernelDataFormat):
        test = Test(test)
        data_format = KernelDataFormat(data_format)

        def decorator(obj):
            key = (test, data_format)
            self[key] = obj
            return obj

        return decorator

    def get(self, test: Test, data_format: KernelDataFormat):
        key = (Test(test), KernelDataFormat(data_format))
        try:
            return self[key]
        except KeyError as e:
            raise KeyError(f"No dispatcher registered for test {test} and data format {data_format}.") from e


class DataHandlerRegistry(dict):
    def register(self, data_format):
        def decorator(obj):
            self[data_format] = obj
            return obj

        return decorator

    def get(self, key):
        try:
            return self[type(key)](key)
        except KeyError as e:
            raise KeyError(f"Support for data type {type(key)} is not implemented.") from e


# How to fetch data from disk, if data is backed or lazy-loaded
data_handler_registry = DataHandlerRegistry()
# Which dispatcher to use depending on data format and test type
dispatcher_registry = DispatcherRegistry()


class DataHandler(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def input_signature(self, *args, **kwargs) -> tuple:
        """Return the numba input signature for this handler."""
        pass

    @abstractmethod
    def fetch(self, *args, **kwargs) -> tuple:
        """Fetch data from disk if needed."""
        pass

    @abstractmethod
    def to_nb(self, *args, **kwargs) -> Any:
        """Convert data to numba-compatible format."""
        pass

    @abstractmethod
    def kernel_data_format(self) -> KernelDataFormat:
        """Return the dispatcher kernel routine for this handler."""
        pass

    @abstractmethod
    def footprint(self) -> int:
        """Return estimated memory footprint of the data."""
        pass


class InRAMDataHandler(DataHandler):
    def fetch(self, lb: int, ub: int) -> tuple:
        """If the data is already in RAM, let the kernels do optimized slicing."""
        return self.data, (lb, ub)


@data_handler_registry.register(np.ndarray)
class DenseDataHandler(InRAMDataHandler):
    def input_signature(self) -> tuple:
        # Because this will be loaded by chunk, input type is necessarily contiguous
        input_type = getattr(types, str(self.data.dtype))[:, ::1]
        return input_type

    def kernel_data_format(self) -> KernelDataFormat:
        return KernelDataFormat.DENSE

    def footprint(self) -> int:
        return self.data.nbytes

    @classmethod
    def to_nb(cls, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return X


@data_handler_registry.register(py_sparse._csr.csr_matrix)
class CSRDataHandler(InRAMDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data.data.dtype))[::1]
        indices_type = getattr(types, str(self.data.indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data.indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSRMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csr_matrix) -> CSRMatrix:
        assert isinstance(X, py_sparse.csr.csr_matrix)
        return CSRMatrix(X.data, X.indices, X.indptr, X.shape)

    def kernel_data_format(self) -> KernelDataFormat:
        return KernelDataFormat.CSR

    def footprint(self) -> int:
        return self.data.data.nbytes + self.data.indptr.nbytes + self.data.indices.nbytes


@data_handler_registry.register(py_sparse._csc.csc_matrix)
class CSCDataHandler(InRAMDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data.data.dtype))[::1]
        indices_type = getattr(types, str(self.data.indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data.indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSCMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csc_matrix) -> CSCMatrix:
        assert isinstance(X, py_sparse.csc.csc_matrix)
        return CSCMatrix(X.data, X.indices, X.indptr, X.shape)

    def kernel_data_format(self) -> KernelDataFormat:
        return KernelDataFormat.CSC

    def footprint(self) -> int:
        return self.data.data.nbytes + self.data.indptr.nbytes + self.data.indices.nbytes


@data_handler_registry.register(h5py.Dataset)
class H5pyDatasetDataHandler(DenseDataHandler):
    def fetch(self, lb: int, ub: int) -> tuple:
        return self.data[:, lb:ub], (0, ub - lb)

    def footprint(self):
        return self.data.nbytes


@data_handler_registry.register(ad._core.sparse_dataset._CSCDataset)
class H5pyBackedCSCDataHandler(CSCDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data._data.dtype))[::1]
        indices_type = getattr(types, str(self.data._indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data._indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSCMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csc_matrix) -> CSCMatrix:
        assert isinstance(X, py_sparse.csc.csc_matrix)
        return CSCMatrix(X.data, X.indices, X.indptr, X.shape)

    def footprint(self) -> int:
        return self.data._data.nbytes + self.data._indptr.nbytes + self.data._indices.nbytes

    def fetch(self, lb: int, ub: int) -> tuple:
        return self.data[:, lb:ub], (0, ub - lb)


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
