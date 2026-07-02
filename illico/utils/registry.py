from abc import ABC, abstractmethod
from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import anndata as ad
import h5py
import numpy as np
from numba import types
from scipy import sparse as py_sparse

from illico.utils.sparse.csc import CSCMatrix
from illico.utils.sparse.csr import CSRMatrix

if TYPE_CHECKING:
    import dask.array as da


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
nb_dispatcher_registry = DispatcherRegistry()
# Register the same dispatchers for the Rust kernels
rs_dispatcher_registry = DispatcherRegistry()


class DataHandler(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def input_signature(self, *args, **kwargs) -> tuple:
        """Return the numba input signature for this handler."""
        pass

    @abstractmethod
    def fetch_rows(self, *args, **kwargs) -> np.ndarray:
        """Fetch data from disk if needed."""
        pass

    @abstractmethod
    def fetch_cols(self, *args, **kwargs) -> tuple:
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

    @property
    @abstractmethod
    def is_lazy(self) -> bool:
        """Return whether the data is lazy-loaded or backed on disk."""
        pass


class InRAMDataHandler(DataHandler):
    def fetch_rows(self, indices: np.ndarray) -> tuple:
        """If the data is already in RAM, let the kernels do optimized slicing."""
        raise NotImplementedError("Rows fetching should only be used for the OVO test on lazy loaded sparse CSR data.")

    def fetch_cols(self, lb: int, ub: int) -> tuple:
        """If the data is already in RAM, let the kernels do optimized slicing."""
        return self.data, (lb, ub)

    @property
    def is_lazy(self) -> bool:
        return False


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


@data_handler_registry.register(py_sparse._csr.csr_array)
@data_handler_registry.register(py_sparse._csr.csr_matrix)
class CSRDataHandler(InRAMDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data.data.dtype))[::1]
        indices_type = getattr(types, str(self.data.indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data.indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSRMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csr_matrix | py_sparse.csr_array) -> CSRMatrix:
        assert isinstance(X, (py_sparse.csr_matrix, py_sparse.csr_array))
        return CSRMatrix(X.data, X.indices, X.indptr, X.shape)

    def kernel_data_format(self) -> KernelDataFormat:
        return KernelDataFormat.CSR

    def footprint(self) -> int:
        return self.data.data.nbytes + self.data.indptr.nbytes + self.data.indices.nbytes


@data_handler_registry.register(py_sparse._csc.csc_array)
@data_handler_registry.register(py_sparse._csc.csc_matrix)
class CSCDataHandler(InRAMDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data.data.dtype))[::1]
        indices_type = getattr(types, str(self.data.indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data.indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSCMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csc_matrix | py_sparse.csc_array) -> CSCMatrix:
        assert isinstance(X, (py_sparse.csc_matrix, py_sparse.csc_array))
        return CSCMatrix(X.data, X.indices, X.indptr, X.shape)

    def kernel_data_format(self) -> KernelDataFormat:
        return KernelDataFormat.CSC

    def footprint(self) -> int:
        return self.data.data.nbytes + self.data.indptr.nbytes + self.data.indices.nbytes


@data_handler_registry.register(h5py.Dataset)
class H5pyDatasetDataHandler(DenseDataHandler):
    def fetch_cols(self, lb: int, ub: int) -> tuple:
        return self.data[:, lb:ub], (0, ub - lb)

    def fetch_rows(self, indices: np.ndarray) -> tuple:
        raise NotImplementedError("Rows fetching should only be used for the OVO test on lazy loaded sparse CSR data.")

    def footprint(self):
        return self.data.nbytes

    @property
    def is_lazy(self) -> bool:
        return True


@data_handler_registry.register(ad._core.sparse_dataset._CSCDataset)
class H5pyBackedCSCDataHandler(CSCDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data._data.dtype))[::1]
        indices_type = getattr(types, str(self.data._indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data._indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSCMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csc_matrix | py_sparse.csc_array) -> CSCMatrix:
        assert isinstance(X, (py_sparse.csc_matrix, py_sparse.csc_array))
        return CSCMatrix(X.data, X.indices, X.indptr, X.shape)

    def footprint(self) -> int:
        return self.data._data.nbytes + self.data._indptr.nbytes + self.data._indices.nbytes

    def fetch_cols(self, lb: int, ub: int) -> tuple:
        return self.data[:, lb:ub], (0, ub - lb)

    def fetch_rows(self, indices: np.ndarray) -> tuple:
        raise NotImplementedError(
            "Fetching rows from a CSC-backed dataset is slow and memory-costly. Instead, load the whole dataset in RAM at once."
        )

    @property
    def is_lazy(self) -> bool:
        return True


@data_handler_registry.register(ad._core.sparse_dataset._CSRDataset)
class H5pyBackedCSRDataHandler(CSRDataHandler):
    def input_signature(self) -> tuple:
        data_type = getattr(types, str(self.data._data.dtype))[::1]
        indices_type = getattr(types, str(self.data._indices.dtype))[::1]
        indptr_type = getattr(types, str(self.data._indptr.dtype))[::1]
        return types.NamedTuple([data_type, indices_type, indptr_type, types.UniTuple(types.int64, 2)], CSRMatrix)

    @classmethod
    def to_nb(cls, X: py_sparse.csr_matrix | py_sparse.csr_array) -> CSRMatrix:
        assert isinstance(X, (py_sparse.csr_matrix, py_sparse.csr_array))
        return CSRMatrix(X.data, X.indices, X.indptr, X.shape)

    def footprint(self) -> int:
        return self.data._data.nbytes + self.data._indptr.nbytes + self.data._indices.nbytes

    def fetch_cols(self, lb: int, ub: int) -> tuple:
        raise NotImplementedError(
            "Fetching columns from a CSR-backed dataset is slow and memory-costly. Instead, load the whole dataset in RAM at once."
        )

    def fetch_rows(self, indices: np.ndarray) -> tuple:
        return self.data[indices, :]

    @property
    def is_lazy(self) -> bool:
        return True


""" Dask Arrays. """


class DaskArrayDataHandler(DataHandler):
    def input_signature(self) -> tuple:
        # Because this will be loaded by chunk, input type is necessarily contiguous
        input_type = getattr(types, str(self.data.dtype.type))[:, ::1]
        return input_type

    def fetch_cols(self, lb: int, ub: int) -> tuple:
        return self.data[:, lb:ub].compute(), (0, ub - lb)

    def fetch_rows(self, indices: np.ndarray) -> tuple:
        raise ValueError("Row fetching is only implemented for lazy CSR arrays.")

    @property
    def is_lazy(self) -> bool:
        return True

    def footprint(self) -> int:
        return np.nan


class DenseDaskArrayDataHandler(DaskArrayDataHandler, DenseDataHandler):
    # Inherits everything from DaskArrayDataHandler and DenseDataHandler, no need to override anything
    pass


class CSCDaskArrayDataHandler(DaskArrayDataHandler, CSCDataHandler):
    # Inherits everything from DaskArrayDataHandler and CSCDataHandler, no need to override anything
    pass


class CSRDaskArrayDataHandler(DaskArrayDataHandler, CSRDataHandler):
    # Inherits almost everything from DaskArrayDataHandler and CSRDataHandler
    def fetch_rows(self, indices: np.ndarray) -> tuple:
        return self.data[indices, :].compute()

is_dask_installed = find_spec("dask") is not None

# Because all dask arrays are of type da.Array, we need to inspect the meta attribute to determine the underlying array type and dispatch to the correct handler
def _dask_handler_factory(x: da.Array) -> DataHandler:
    if not is_dask_installed:
        raise ImportError("Install dask via the extra `\"illico[dask]\"` to be able to use `dask.Array` inside `illico`")
    meta = x._meta
    if isinstance(meta, np.ndarray):
        return DenseDaskArrayDataHandler(x)
    elif isinstance(meta, (py_sparse.csr_matrix, py_sparse.csr_array)):
        return CSRDaskArrayDataHandler(x)
    elif isinstance(meta, (py_sparse.csc_matrix, py_sparse.csc_array)):
        return CSCDaskArrayDataHandler(x)
    else:
        raise TypeError(f"Unsupported dask array backing type: {type(meta)}")

if is_dask_installed:
    data_handler_registry[da.Array] = _dask_handler_factory
    data_handler_registry[ad._core.views.DaskArrayView] = _dask_handler_factory
