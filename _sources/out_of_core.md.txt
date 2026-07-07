# What if my adata does not fit in memory?

Although not initially designed to run out-of-core rank-sum tests, `illico` supports **some** disk-backed expression matrices natively. The slowdown occurred by backing the dataset on disk is hard to estimate as it directly depends on your system's IO. Notably:

- h5-dense (np.ndarray) disk-backed dataset are natively supported
- h5-CSC (sparse along the columns) disk-backed datasets are natively supported
- h5-CSR (sparse along the rows) disk-backed datasets are natively supported **only for OVO (perturbed vs controls) test**. If you want to perform OVR (each group vs the rest) tests, you are better off loading it entirely in memory, as OVR test requires each column to be entirely in RAM at once, and CSR format does not allow to load columns from disk without loading the entire `.indices` in RAM (without telling you).
- Dask-backed datasets are supported in the same scenarii as h5 datasets.

If your data is backed through another backend, please open an issue.

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
| OVO (reference="non-targeting")  | Dense  |  Lazy (Dask)  | ✅   | - |
| OVO (reference="non-targeting")  | CSR  |  Lazy (Dask)  | ✅   | - |
| OVO (reference="non-targeting")  | CSC  |  Lazy (Dask)  | ✅   | - |
| OVR (reference=None)  | Dense  |  Lazy (Dask)  | ✅   | - |
| OVR (reference=None)  | CSR  |  Lazy (Dask)  | ✅   | Setting `illico`'s batch size to 512 or 1024 will greatly reduce compute time for limited extra footprint.  |
| OVR (reference=None)  | CSC  |  Lazy (Dask)  | ✅   | - |



Notes:

1. Supporting the CSR use case is highly non trivial, and running `adata[:, idxs]` on a backed CSR matrix will load (temporarily) the entirety of the indices in RAM, resulting in a memory footprint almost equivalent to loading everything at once, on top of being extremely slow. That's why OVR test on lazy CSR is not supported.
2. Users struggling with out-of-core single cell RNASeq analyses should visit `rapids-singlecell`, which explicitely targets this use-case.
3. The "Specific parallelization scheme mentioned for the OVO lazy CSR use case simply relies on the fact that due to the nature of the OVR test, we can run it group by group, and thus only load one group at a time in RAM, which is not the case for OVR where we need to load all groups at once.
4. Note also that illico is expected to scale less well on lazy datasets, as most of the time the data loading part (such as the one of h5 datasets) is GIL-blocking.

## Note on Dask support
Dask support was added in `v0.6.0` and, although fully functional, it is still in early stages. Notably, efficiently parallelizing the tests on dask-backed datasets is not trivial, as many possibilities and design choices exist to do so. For in-RAM datasets, tests are parallelized over threads by splitting the dataset in vertical chunks (i.e. groups of genes). For dask-backed datasets, the dask scheduler parallelizes the data loading, but could also be used to parallelize the tests: in this case, the resulting memory footprint would be quite high. In order to not defeat the initial goal of lazy loading (maintaining a small RAM footprint), the design choices made on this side were **conservative**. Consequently, the parallelization scheme in this scenario is thread-based:
- One thread loads the chunk in RAM (with a `.compute()` call)
- One thread processes this chunk

Those two operations are ran in parallel as the chunk processing is non GIL-blocking. The fact that only one `.compute()` is called at a time allows to maintain low RAM footprint. The data loading is protected with a Semaphore to avoid multiple concurrent `.compute()` calls. Using two threads allows an efficient overlapping of loading & processing, making the most of the machine's resources. For more details and benchmarks on this question, please refer to the [this issue](https://github.com/remydubois/illico/issues/21#issuecomment-4693686202).
