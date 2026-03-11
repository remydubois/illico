Changelog
=========

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
