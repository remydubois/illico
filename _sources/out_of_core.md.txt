# What if my adata does not fit in memory?

Although not initially designed to run out-of-core rank-sum tests, `illico` supports **some** disk-backed expression matrices natively. The slowdown occurred by backing the dataset on disk is hard to estimate as it directly depends on your system's IO. Notably:

- h5-dense (np.ndarray) disk-backed dataset are natively supported
- h5-CSC (sparse along the columns) disk-backed datasets are natively supported
- h5-CSR (sparse along the rows) disk-backed datasets are natively supported **only for OVO (perturbed vs controls) test**. If you want to perform OVR (each group vs the rest) tests, you are better off loading it entirely in memory, as OVR test requires each column to be entirely in RAM at once, and CSR format does not allow to load columns from disk without loading the entire `.indices` in RAM (without telling you).

If your data is backed through Dask or another backend, please open an issue as it should require little rework for it to be supported.

Summary:
|               Test               | Format | Storage | Supported ? | Remark |
|----------------------------------|--------|--------|--------|------|
| [OVO\|OVR]  | [Dense\|CSC\|CSR]  |  In RAM  | ✅   | - |
| OVO (reference="non-targeting")  | Dense  |  Lazy (H5)  | ✅   | - |
| OVO (reference="non-targeting")  | CSR  |  Lazy (H5)  | ✅   | Specific parallelization scheme |
| OVO (reference="non-targeting")  | CSC  |  Lazy (H5)  | ✅   | - |
| OVR (reference=None)  | Dense  |  Lazy (H5)  | ✅   | - |
| OVR (reference=None)  | CSR  |  Lazy (H5)  |  ❌   | Voluntarily not supported, better off loading in RAM  |
| OVR (reference=None)  | CSC  |  Lazy (H5)  | ✅   | - |



Notes:

1. Supporting the CSR use case is highly non trivial, and running `adata[:, idxs]` on a backed CSR matrix will load (temporarily) the entirety of the indices in RAM, resulting in a memory footprint almost equivalent to loading everything at once, on top of being extremely slow. That's why OVR test on lazy CSR is not supported.
2. Users struggling with out-of-core single cell RNASeq analyses should visit `rapids-singlecell`, which explicitely targets this use-case.
3. The "Specific parallelization scheme mentioned for the OVO lazy CSR use case simply relies on the fact that due to the nature of the OVR test, we can run it group by group, and thus only load one group at a time in RAM, which is not the case for OVR where we need to load all groups at once.
4. Note also that illico is expected to scale less well on lazy datasets, as most of the time the data loading part (such as the one of h5 datasets) is GIL-blocking.
