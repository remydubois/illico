import os
import re
from datetime import datetime
from pathlib import Path

import memray
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from numba import set_num_threads
from pdex import parallel_differential_expression
from pdex._single_cell import parallel_differential_expression_vec_wrapper
from scanpy.tools import _rank_genes_groups
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from illico.asymptotic_wilcoxon import asymptotic_wilcoxon
from illico.utils.compile import _precompile

set_num_threads(1)  # Ensure single-threaded by default for testing consistency

_rank_genes_groups.enumerate = lambda x: enumerate(tqdm(x))
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


def scipy_mannwhitneyu(adata, groupby_key, reference, is_log1p=False):
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

        stats, pvals = mannwhitneyu(ref_counts, grp_counts, axis=0, method="asymptotic")
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


@pytest.mark.parametrize("test", ["ovo", "ovr"])
def test_asymptotic_wilcoxon(rand_adata, test):
    cached = rand_adata.copy()

    if test == "ovo":
        reference = rand_adata.obs.pert.iloc[0]
    else:
        reference = None

    asy_results = asymptotic_wilcoxon(
        adata=rand_adata, is_log1p=False, group_keys="pert", reference_group=reference, n_threads=1, batch_size=16
    )

    scipy_results = scipy_mannwhitneyu(adata=rand_adata, groupby_key="pert", reference=reference, is_log1p=False)
    # sc_results = scanpy_mannwhitneyu(adata=rand_adata, groupby_key="pert", reference=reference)

    # Test statistics exactly
    np.testing.assert_allclose(
        asy_results.loc[scipy_results.index].statistic.values,
        scipy_results.statistic.values,
        atol=0.0,
        rtol=0.0,
    )
    # Test p-values with tolerance
    np.testing.assert_allclose(
        asy_results.loc[scipy_results.index].p_value.values,
        scipy_results.p_value.values,
        atol=0.0,
        rtol=1.0e-12,
    )
    # Test FC with tolerance
    np.testing.assert_allclose(
        asy_results.loc[scipy_results.index].fold_change.values,
        scipy_results.fold_change.values,
        atol=0.0,
        rtol=1.0e-6,
    )

    # Test that the original adata is not modified, some sorting happen in-place so just making sure
    pd.testing.assert_frame_equal(rand_adata.obs, cached.obs)
    pd.testing.assert_frame_equal(rand_adata.var, cached.var)
    if isinstance(rand_adata.X, np.ndarray):
        np.testing.assert_array_equal(rand_adata.X, cached.X)
    else:
        np.testing.assert_array_equal(rand_adata.X.toarray(), cached.X.toarray())


def test_unsorted_indices_error(rand_adata):
    """Test that an error is raised if input data matrix indices are not sorted."""
    if isinstance(rand_adata.X, np.ndarray):
        pytest.skip("Test only relevant for sparse matrices.")
    # Unsort the indices of the csr matrix
    rand_adata.X.indices[:] = rand_adata.X.indices[::-1]
    with pytest.raises(ValueError):
        _ = asymptotic_wilcoxon(
            adata=rand_adata,
            is_log1p=False,
            group_keys="pert",
            reference_group="non-targeting",
            n_threads=1,
            batch_size=16,
        )


def call_routine(data, method, test, num_threads):
    def run():
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
                reference_group=reference,
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

    _rank_genes_groups._CONST_MAX_SIZE = int(
        2**31
    )  # If not set, scanpy will chunk the input data making display and progress bars completely hectic

    # Compile
    if method == "illico":
        _precompile(adata.X, reference_group="non-targeting" if test == "ovo" else None)

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
        _precompile(adata.X, reference_group="non-targeting" if test == "ovo" else None)

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
