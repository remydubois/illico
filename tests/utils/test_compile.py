import pytest

from illico.utils.compile import _precompile
from illico.utils.groups import encode_and_count_groups
from illico.utils.type import scipy_to_nb


@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_precompile(rand_adata, test):
    reference = rand_adata.obs.pert.iloc[0] if test == "ovo" else None
    # Now compile it, and make sure it compiled nopython
    dispatcher = _precompile(rand_adata.X, reference)
    assert (len(leg_sig := dispatcher.nopython_signatures)) > 0, "Dispatcher should be compiled now."
    # Now run the dispatcher
    _, grpc = encode_and_count_groups(rand_adata.obs.pert.values, reference)
    dispatcher(scipy_to_nb(rand_adata.X), 0, rand_adata.X.shape[1], grpc, False, True, "two-sided")
    # Assert no other signature was added
    assert len(dispatcher.nopython_signatures) == len(
        leg_sig
    ), f"Dispatcher should not have recompiled: {chr(10).join(map(str, dispatcher.nopython_signatures))}"
