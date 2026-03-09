import anndata as ad
import memray
import pytest

from illico import asymptotic_wilcoxon
from illico.utils.compile import _precompile
from illico.utils.groups import encode_and_count_groups
from illico.utils.memory import log_memory_usage
from illico.utils.registry import data_handler_registry


@pytest.mark.skip(
    f"Memory footprint is actually too cumbersome to estimate, especially in very low data regimes like in this test."
)
@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_log_memory_usage(rand_adata, test, tmp_path):
    # No need to test that exception is raised, as it is done in `test_asymptotic_wilcoxon` already
    if isinstance(rand_adata.X, ad._core.sparse_dataset._CSRDataset):
        pytest.skip("CSR lazy data not supported for now.")

    data_handler = data_handler_registry.get(rand_adata.X)
    reference = rand_adata.obs.pert.iloc[0] if test == "ovo" else None
    _, grpc = encode_and_count_groups(groups=rand_adata.obs["pert"].tolist(), ref_group=reference)

    batch_size = 16
    n_threads = 1

    estimated_heap_size = log_memory_usage(data_handler, grpc, batch_size, n_threads)

    # precompile
    _precompile(data_handler, reference)

    with memray.Tracker((_f := tmp_path / "tracker.bin")) as tracker:
        asymptotic_wilcoxon(
            adata=rand_adata,
            is_log1p=False,
            group_keys="pert",
            reference=reference,
            n_threads=n_threads,
            batch_size=batch_size,
        )
    with memray.FileReader(_f) as reader:
        real_heap_size = reader.metadata.peak_memory
    # Allow some tolerance for overhead
    assert real_heap_size <= estimated_heap_size * 1.1, f"Estimated: {estimated_heap_size}, Real: {real_heap_size}"
