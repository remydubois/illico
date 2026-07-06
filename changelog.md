Changelog
=========

Version 0.6.0
------------
- Adds support for AnnData read as Dask arrays.
- Adds support for `groups` and `exclude` arguments:
    - `groups` is the same as `scanpy`'s `groups` argument, allowing to specify a subset of groups to test: only p-values of those groups will be computed. For the OVR test, there is no significant effect on execution time as ranks need to be computed for every cell anyways. For the OVO test: the smaller this subset is, the faster the test will be, as less groups are compared to the control.
    - `exclude` allows to specify a subset of groups to exclude from testing. Only applicable to the OVR test. The bigger this exclusion group, the faster the test, as there are less values to rank.
- Lowered anndata lower bound to 0.10.8.
- Made Dask an optional dependency

Version 0.5.1
------------
- Fixes bug when sparse matrices contain strictly negative values, cf [this discussion](https://scverse.zulipchat.com/#narrow/channel/557094-differential-gene-expression/topic/Nonnegative.20expectation/near/590691572).

Version 0.5.0rc2
------------
- For better compatibility with `scanpy`, lower bounds on dependencies have been relaxed.

Version 0.5.0rc1
------------
This version improves compatibility with `scanpy`:
- Added `n_genes` arguments allowing to return only the top N genes per group when `return_as_scanpy=True`. This allowed to match `scanpy`'s sorting method (partial sort) resulting in better reproducibility of scanpy results.
- Fixed genes ordering in the scanpy formatter, by removing redundant sorting of perturbation names as `encode_and_count_groups` already returns sorted unique perturbation names. This ensures that gene names are sorted the same way everywhere.
- Added explicit testing of genes ordering. In the PBMC dataset, lots of genes end up with identical z-scores but different logfoldchanges. This was not caught by previous tests.
- Fold change is now computed with `(numerator + 1.e-9) / (denominator + 1.e-9)` to avoid division by zero, and to be more consistent with scanpy's implementation. This has no     effect on the ranking of genes, but allows to get finite fold change values for all genes.

It also includes some performance improvements:
- Improved CSR chunking mechanism for the OVO test, resulting in faster execution and much smaller memory footprint. A direct implication is that `batch_size` can grow much larger now.
    - On TAHOE's `plate3` (in RAM) with `batch_size=1024`, this reduced memory footprint from 35GB to 1.5GB, and runtime from 1:17 to 0:50 with 8 CPUs.
    - The reduced footprint allows to scale more aggressively `n_threads`. With 32 threads, TAHOE's `plate3` runs in 21 seconds, while eating only 2.5GB of RAM.

Also, it adds support for OVO test on lazy CSR (h5-based) datasets, through a specific parallelization scenario where groups are processed one by one.

Version 0.4.0
------------
- Added option to return scanpy-friendly output with `return_as_scanpy` arg. `asymptotic_wilcoxon` returns either:
    - A `pandas.DataFrame` with columns `feature`, `p_value`, `fold_change`, and `statistic` (default), if `return_as_scanpy=False`
    - A dictionary containing the same keys as `scanpy.tl.rank_genes_groups`, if `return_as_scanpy=True`. Similarly as scanpy, genes are ordered by decreasing z-score.
- Improved the batching mechanism, fixed the 'auto' mode that was excluding the very last gene in previous versions.

Version 0.3.0
------------
- Rust backend is available for all tests. Compare Rust vs Numba with `poetry run pytest-benchmark compare 0003 0005`:
    - CSR OVO approx 20% faster
    - CSR OVR approx 80% faster
    - Dense OVO approx 70% faster
    - Dense OVR approx 100% faster (twice faster)
- Moved results allocation into thread operator for both Numba and Rust
    - Compare before/after with `poetry run pytest-benchmark compare 0003 0008`, approx 15% speedup on 8 threads.
    - Enables better scaling to larger machines: 32 threads is approximately 27 times faster than 1 thread.

Version 0.2.0
------------
- H5-based, disk-backed, CSC and dense datasets are now supported natively.
- Non tie-corrected tests are now supported as well.

Version 0.1.1
------------
- Changed `reference_group` to `reference` for better transparence with the `scanpy` API.

Version 0.1.0
------------
First version
