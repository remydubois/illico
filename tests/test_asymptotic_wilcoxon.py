import contextlib
import gc
import os
import re
import warnings
from pathlib import Path

import anndata as ad
import memray
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from numba import set_num_threads
from pdex import parallel_differential_expression
from pdex._single_cell import parallel_differential_expression_vec_wrapper
from scipy import sparse as py_sparse
from scipy.stats import mannwhitneyu

from illico.asymptotic_wilcoxon import asymptotic_wilcoxon
from illico.utils.compile import _precompile
from illico.utils.registry import data_handler_registry, dispatcher_registry

set_num_threads(1)  # Ensure single-threaded by default for testing consistency

ATOL = 0.0
RTOL = 1.0e-12


def scanpy_mannwhitneyu(adata, groupby_key, reference):
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby_key,
        method="wilcoxon",
        use_raw=False,
        tie_correct=True,
        reference=reference or "rest",
    )

    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names  # perturbed genes

    records = []

    for g in groups:
        records.append(
            pd.DataFrame(
                {
                    "target": g,
                    "feature": rg["names"][g],
                    "pval": rg["pvals"][g],
                    "pval_adj": rg["pvals_adj"][g],
                    "logfoldchange": rg["logfoldchanges"][g],
                    "ustat": rg["scores"][g],  # Wilcoxon statistic proxy
                }
            )
        )

    df = pd.concat(records, ignore_index=True)
    return df.set_index(["target", "feature"])


def scipy_mannwhitneyu(adata, groupby_key, reference, use_continuity, alternative, is_log1p=False):
    if reference is not None:
        ref_counts = adata[adata.obs[groupby_key].eq(reference)].X
        if not isinstance(ref_counts, np.ndarray):
            ref_counts = ref_counts.toarray()

    # Loop over perturbations
    results = []
    for pert in adata.obs[groupby_key].unique():
        if pert == reference:
            continue
        mask = adata.obs[groupby_key].eq(pert).values
        grp_counts = adata.X[mask]  # Grab the perturbed counts
        if reference is None:
            ref_counts = adata.X[~mask]  # Grab the perturbed counts

        # densify
        if not isinstance(grp_counts, np.ndarray):
            grp_counts = grp_counts.toarray()
        if not isinstance(ref_counts, np.ndarray):
            ref_counts = ref_counts.toarray()

        # Compute FC
        if is_log1p:
            grp_counts = np.expm1(grp_counts)
            ref_counts = np.expm1(ref_counts)
            fc = np.expm1(grp_counts).mean(axis=0) / np.expm1(ref_counts).mean(axis=0)
        else:
            fc = np.mean(grp_counts, axis=0) / np.mean(ref_counts, axis=0)

        stats, pvals = mannwhitneyu(
            ref_counts, grp_counts, axis=0, method="asymptotic", use_continuity=use_continuity, alternative=alternative
        )
        results.append(
            pd.DataFrame(
                {
                    "p_value": pvals,
                    "statistic": stats,
                    "fold_change": fc,
                    "target": pert,
                    "feature": adata.var_names,
                }
            )
        )
    results = pd.concat(results, axis=0).set_index(["target", "feature"])
    return results


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("tie_correct", [True, False], ids=["tie-correct", "no-tie-correct"])
@pytest.mark.parametrize("use_continuity", [True, False])
@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_asymptotic_wilcoxon(rand_adata, test, use_continuity, tie_correct, alternative):
    if not rand_adata.isbacked:
        cached = rand_adata.copy()

    if test == "ovo":
        reference = rand_adata.obs.pert.iloc[0]
    else:
        reference = None

    # If rand_adata is lazy and CSR, ensure we raise an error because this is not supported
    if isinstance(rand_adata.X, ad._core.sparse_dataset._CSRDataset):
        ctx = pytest.raises(
            KeyError,
            match="Support for data type <class 'anndata._core.sparse_dataset._CSRDataset'> is not implemented.",
        )
        should_raise = True
    else:
        ctx = contextlib.nullcontext()
        should_raise = False

    with ctx:
        asy_results = asymptotic_wilcoxon(
            adata=rand_adata,
            is_log1p=False,
            group_keys="pert",
            reference=reference,
            use_continuity=use_continuity,
            tie_correct=tie_correct,
            n_threads=1,
            batch_size=16,
            alternative=alternative,
        )

    if should_raise:
        return

    if not tie_correct:
        # We skip at this point, so that we make sure that at least the code runs
        pytest.skip(f"Skipping comparison with scipy when tie correction is disabled, as scipy does not support it.")

    scipy_results = scipy_mannwhitneyu(
        adata=rand_adata,
        groupby_key="pert",
        reference=reference,
        is_log1p=False,
        use_continuity=use_continuity,
        alternative=alternative,
    )
    # sc_results = scanpy_mannwhitneyu(adata=rand_adata, groupby_key="pert", reference=reference)

    # Test statistics exactly
    np.testing.assert_allclose(
        asy_results.loc[scipy_results.index].statistic.values,
        scipy_results.statistic.values,
        atol=0.0,
        rtol=0.0,
    )
    # Test p-values with low tolerance
    np.testing.assert_allclose(
        asy_results.loc[scipy_results.index].p_value.values,
        scipy_results.p_value.values,
        atol=0.0,
        rtol=1.0e-12,
    )
    # Test FC with mid tolerance
    np.testing.assert_allclose(
        asy_results.loc[scipy_results.index].fold_change.values,
        scipy_results.fold_change.values,
        atol=0.0,
        rtol=1.0e-6,
    )

    if not rand_adata.isbacked:
        # Test that the original adata is not modified, some sorting happen in-place so just making sure
        pd.testing.assert_frame_equal(rand_adata.obs, cached.obs)
        pd.testing.assert_frame_equal(rand_adata.var, cached.var)
        if isinstance(rand_adata.X, np.ndarray):
            np.testing.assert_array_equal(rand_adata.X, cached.X)
        else:
            np.testing.assert_array_equal(rand_adata.X.toarray(), cached.X.toarray())


# Do not sweep all the possible test params, alternative and all
@pytest.mark.parametrize("backed", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_backed_asymptotic_wilcoxon(eager_rand_adata, test, backed, tmp_path):
    # No need to test that exception is raised, as it is done in `test_asymptotic_wilcoxon` already
    if isinstance(eager_rand_adata.X, py_sparse.csr.csr_matrix) and backed:
        pytest.skip("CSR lazy data not supported for now.")

    if test == "ovo":
        reference = eager_rand_adata.obs.pert.iloc[0]
    else:
        reference = None

    # Precompile
    data_handler = data_handler_registry.get(eager_rand_adata.X)
    _precompile(data_handler, reference)

    # Run this with one thread and small batch size, this simply makes sure we never load
    adata_path = tmp_path / f"rand_adata_lazy.h5ad"
    # Make this anndata bigger, otherwise memory measurements are not significant
    bigger_eager_rand_adata = ad.concat([eager_rand_adata] * 100, axis=1)
    # Concatenation converts to CSR, so revert back to CSC
    if isinstance(eager_rand_adata.X, py_sparse.csc.csc_matrix):
        bigger_eager_rand_adata.X = py_sparse.csc_matrix(bigger_eager_rand_adata.X)
    bigger_eager_rand_adata.obs = eager_rand_adata.obs.copy()
    bigger_eager_rand_adata.var_names_make_unique()
    bigger_eager_rand_adata.write_h5ad(adata_path)

    # In order to track proper memory usage, we include the read_h5ad call within the memray context
    # Consequently, memory allocated to adata will show as heap memory, unlike memory tests below which only
    # tracked algorithm allocations
    with memray.Tracker(tmp_path / "memray-trace.bin", file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS):
        adata = ad.read_h5ad(adata_path, backed="r" if backed else None)
        _ = asymptotic_wilcoxon(
            adata=adata,
            is_log1p=False,
            group_keys="pert",
            reference=reference,
            use_continuity=True,
            n_threads=1,
            batch_size=16,
            alternative="two-sided",
        )
    max_rss, max_heap = 0, 0
    with memray.FileReader(tmp_path / "memray-trace.bin") as reader:
        for snapshot in reader.get_memory_snapshots():
            max_rss = max(max_rss, snapshot.rss)
            max_heap = max(max_heap, snapshot.heap)

    # print(f"Max heap memory usage: {max_heap/1_000_000:.1f} bytes")
    if backed:
        if max_heap > 10_000_000:  # 10 MB
            raise AssertionError(
                f"Expected low (<10MB) heap memory usage when running in backed mode, got {max_heap/1_000_000:.1f} MB."
            )
    else:
        if max_heap < 50_000_000:  # 50 MB
            raise AssertionError(
                f"Expected high (>50MB) heap memory usage when running in backed mode, got {max_heap/1_000_000:.1f} MB."
            )


def test_unsorted_indices_error(eager_rand_adata):
    """Test that an error is raised if input data matrix indices are not sorted."""
    if isinstance(eager_rand_adata.X, np.ndarray):
        pytest.skip("Test only relevant for sparse matrices.")
    # Unsort the indices of the csr matrix
    eager_rand_adata.X.indices[:] = eager_rand_adata.X.indices[::-1]
    with pytest.raises(ValueError):
        _ = asymptotic_wilcoxon(
            adata=eager_rand_adata,
            is_log1p=False,
            group_keys="pert",
            reference="non-targeting",
            n_threads=1,
            batch_size=16,
        )


def call_routine(data, method, test, num_threads):
    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "pdex":
                parallel_differential_expression(
                    data,
                    groupby_key="gene",
                    reference="non-targeting",
                    num_workers=num_threads,
                )
            elif method == "pdexp":
                parallel_differential_expression_vec_wrapper(
                    data,
                    groupby_key="gene",
                    reference="non-targeting",
                    num_workers=num_threads,
                )
            elif method == "illico":
                reference = "non-targeting" if test == "ovo" else None
                asymptotic_wilcoxon(
                    data,
                    is_log1p=False,
                    group_keys="gene",
                    reference=reference,
                    n_threads=num_threads,
                    batch_size=256,
                )
            elif method == "scanpy":
                reference = "non-targeting" if test == "ovo" else "rest"
                set_num_threads(num_threads)  # Scanpy does not set number of threads explicitely
                group_counts = data.obs["gene"].value_counts()
                valid_groups = group_counts.index[group_counts.values > 1].tolist()
                sc.tl.rank_genes_groups(
                    data,
                    groupby="gene",
                    groups=valid_groups,
                    reference=reference,
                    method="wilcoxon",
                    tie_correct=True,
                )
            else:
                raise ValueError(method)

    return run


@pytest.mark.speed_bench
@pytest.mark.parametrize("num_threads", [1, 2, 4, 8], ids=lambda v: f"nthreads={v}")
@pytest.mark.parametrize("test", ["ovo", "ovr"])
@pytest.mark.parametrize("method", ["illico", "scanpy", "pdex", "pdexp"])
def test_speed_benchmark(adata, method, test, num_threads, benchmark, request):
    """Not a test, just a speed benchmark."""
    if test != "ovo" and method in ["pdex", "pdexp"]:
        # This exits the test, not running the benchmark, and not raising an error
        pytest.skip("pdex only implements OVO test.")

    # Compile
    if method == "illico":
        _precompile(data_handler_registry.get(adata.X), reference="non-targeting" if test == "ovo" else None)

    params = re.match(".*\[(.*)\]", request.node.name).group(1).split("-")
    group_params = [p for i, p in enumerate(params) if i in [0, 1, 4]]
    benchmark.group = "-".join(group_params)
    _ = benchmark.pedantic(call_routine(adata, method, test, num_threads), iterations=1, warmup_rounds=0, rounds=1)


@pytest.mark.memory_bench
@pytest.mark.parametrize("num_threads", [8], ids=lambda v: f"nthreads={v}")
@pytest.mark.parametrize("test", ["ovo", "ovr"])
@pytest.mark.parametrize("method", ["illico", "scanpy", "pdex", "pdexp"])
def test_memory_benchmark(adata, method, test, num_threads, request):
    """Not a test, just a memory footprint benchmark."""
    if test != "ovo" and method == "pdex":
        # For memory benchmark, we raise here so that it does not show in the resulting summary
        # raise ValueError('PDEX can only run OVO ranksum test')
        pytest.skip("pdex only implements OVO test.")

    # Compile outside of the tracker context
    if method == "illico":
        _precompile(adata.X, reference="non-targeting" if test == "ovo" else None)

    test_params_string = re.match(".*\[(.*)\]", request.node.name).group(1)
    outdir = Path(os.environ.get("MEMRAY_RESULTS_DIR") or Path(__file__).parents[1])

    trace_filepath = lambda x: outdir / ".memray-trackings" / f"trace-{test_params_string}-{str(x).zfill(4)}.bin"
    run_increment = 0
    while (_fp := trace_filepath(run_increment)).exists():
        run_increment += 1
    _fp.parent.mkdir(exist_ok=True)

    try:
        with memray.Tracker(_fp, file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS):
            _ = call_routine(adata, method, test, num_threads)()
    except Exception as e:
        # Cleanup the file if an error happened
        _fp.unlink(missing_ok=True)
        raise e
