Changelog
=========

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
