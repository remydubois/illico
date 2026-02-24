# What if my adata does not fit in memory?

Although not initially designed to run out-of-core rank-sum tests, `illico` supports **some** disk-backed expression matrices natively. The slowdown occurred by backing the dataset on disk is hard to estimate as it directly depends on your system's IO. Notably:

- h5-dense (np.ndarray) disk-backed dataset are natively supported
- h5-CSC (sparse along the columns) disk-backed datasets are natively supported
- :warning: **h5-CSR (sparse along the rows) disk-backed datasets are not supported**

If your data is backed through Dask or another backend, please open an issue as dense and CSC use cases should require very little rework to be supported.

Notes:

1. Supporting the CSR use case is highly non trivial, and running `adata[:, idxs]` on a backed CSR matrix will load (temporarily) the entirety of the indices in RAM, resulting in a memory footprint almost equivalent to loading everything at once, on top of being extremely slow.
2. Users struggling with out-of-core single cell RNASeq analyses should visit `rapids-singlecell`, which explicitely targets this use-case.
