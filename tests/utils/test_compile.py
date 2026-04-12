import anndata as ad
import pytest

from illico.utils.compile import _precompile
from illico.utils.groups import encode_and_count_groups
from illico.utils.registry import KernelDataFormat, data_handler_registry
from illico.utils.sparse.csr import csr_to_csc


@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_precompile(rand_adata, test):
    # No need to test that exception is raised, as it is done in `test_asymptotic_wilcoxon` already
    data_handler = data_handler_registry.get(rand_adata.X)
    if isinstance(rand_adata.X, ad._core.sparse_dataset._CSRDataset) and test == "ovr":
        pytest.skip("OVR on CSR lazy data not supported for now.")

    # Run pre-compilation
    reference = rand_adata.obs.pert.iloc[0] if test == "ovo" else None
    dispatcher = _precompile(data_handler, reference)
    # Now compile it, and make sure it compiled nopython
    assert (len(leg_sig := dispatcher.nopython_signatures)) > 0, "Dispatcher should be compiled now."

    # Now check that re-running does not trigger another compilation
    if data_handler.is_lazy and data_handler.kernel_data_format() is KernelDataFormat.CSR and reference is not None:
        x_csr = data_handler.to_nb(data_handler.data[:])
        x_csc = csr_to_csc(x_csr)
        dispatcher(x_csc, x_csc, False, False, "two-sided")
    else:
        # Now run the dispatcher
        _, grpc = encode_and_count_groups(rand_adata.obs.pert.values, reference)
        X, bounds = data_handler.fetch_cols(0, rand_adata.X.shape[1])
        X_nb = data_handler.to_nb(X)
        dispatcher(X_nb, *bounds, grpc, False, True, True, False, "two-sided")

    # Assert no other signature was added
    assert len(dispatcher.nopython_signatures) == len(
        leg_sig
    ), f"Dispatcher should not have recompiled: {chr(10).join(map(str, dispatcher.nopython_signatures))}"
