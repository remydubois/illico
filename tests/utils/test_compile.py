import anndata as ad
import pytest

from illico.utils.compile import _precompile
from illico.utils.groups import encode_and_count_groups
from illico.utils.registry import data_handler_registry


@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_precompile(rand_adata, test):
    # No need to test that exception is raised, as it is done in `test_asymptotic_wilcoxon` already
    if isinstance(rand_adata.X, ad._core.sparse_dataset._CSRDataset):
        pytest.skip("CSR lazy data not supported for now.")

    reference = rand_adata.obs.pert.iloc[0] if test == "ovo" else None
    # Now compile it, and make sure it compiled nopython
    data_handler = data_handler_registry.get(rand_adata.X)
    dispatcher = _precompile(data_handler, reference)
    assert (len(leg_sig := dispatcher.nopython_signatures)) > 0, "Dispatcher should be compiled now."
    # Now run the dispatcher
    _, grpc = encode_and_count_groups(rand_adata.obs.pert.values, reference)
    X, bounds = data_handler.fetch(0, rand_adata.X.shape[1])
    X_nb = data_handler.to_nb(X)
    dispatcher(X_nb, *bounds, grpc, False, True, True, "two-sided")
    # Assert no other signature was added
    assert len(dispatcher.nopython_signatures) == len(
        leg_sig
    ), f"Dispatcher should not have recompiled: {chr(10).join(map(str, dispatcher.nopython_signatures))}"
