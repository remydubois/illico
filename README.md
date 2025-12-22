# Illico
## Overview
*illico* is a python library performing blazing fast asymptotic wilcoxon rank-sum tests (same as `scanpy.tl.rank_genes_groups(… tie_correct=True)`), useful for single-cell RNASeq data analyses and processing. `illico`'s features are:
1. :rocket: Blazing fast: On K562 (essential) dataset (~300k cells, 8k genes, 2k perturbations), `illico` computes DE genes (with `reference="non-targeting"`) in a mere 30 seconds. That's more than 100 times faster than both `pdex` or `scanpy` with the same compute ressources (8 CPUs).
2. :diamond_shape_with_a_dot_inside: No compromise: on synthetic data, `illico`'s p-values matched `scipy.stats.mannwhitneyu` up to a relative difference of 1.e-12, and an absolute tolerance of 0.
3. :zap: Thread-first: `illico` eventually parallelizes the processing (if specified by the user) over **threads**, never processes. This saves you from all the fixed cost of multiprocessing, such as spanning processes, duplicating data across processes, and communication costs.
3. :beetle: Data format agnostic: whether your data is dense, sparse along rows, or sparse along columns, `illico` will deal with it while never converting the whole data to whichever format is more optimized.
4. 🪶 Lightweight: `illico` will process the input data in batches, making any memory allocation needed along the way much smaller than if it processed the whole data at once.
5. 📈 Scalable: Because thread-first and batchable, `illico` scales reasonably with your compute budget. Tests showed that spanning 8 threads brings a 7-fold speedup over spanning 1 single thread.
6. :fireworks: All-purpose: `illico` performs both one-versus-reference (useful for perturbation analyses) and one-versus-rest (useful for clustering analyses) wilcoxon rank-sum tests, both equally optimized and fast.

Approximate speed benchmarks ran on k562-essential can be found below. All the code used to generate those numbers can be found in `tests/test_asymptotic_wilcoxon.py::test_speed_benchmark`.

|               Test               | Format | Illico | Scanpy | pdex |
|----------------------------------|--------|--------|--------|------|
| OVO (reference="non-targeting")  | Dense  | <1min  | ~1h    | ~4h  |
| OVO (reference="non-targeting")  | Sparse | <1min  | ~1h30  | ~4h  |
| OVR (reference=None)             | Dense  | <1min  | ~11h   |  X   |
| OVR (reference=None)             | Sparse | <1min  | ~10h   |  X   |

:bulb: Note:
1. This library only performs tie-corrected wilcoxon rank-sum tests, also known as Mann-Whitney test, also performed by `scanpy.tl.rank_genes_groups(…, tie_correct=True)`. It **does not** perform wilcoxon signed-sum tests, those are less often used in for single-cell data analyses as it requires samples to be **paired**.
1. Exact benchmarks ran on a subset of the whole k562 can be found at the end of this readme.
2. OVO refers to one-versus-one: this test computes u-stats and p-values between control cells and perturbed cells. Equivalent to `scanpy`'s `rank_gene_groups(…, reference="non-targeting")`.
3. OVR refers to one-versus-rest: this test computes u-stats and p-values between each group cells, and all other cells, for each group. Equivalent to `scanpy`'s `rank_gene_groups(…, reference="rest")`.
4. This package is not intended at running out-of-core single cell data analyses like `rapids-singlecell`.

## Installation
`illico` can be installed via pip, compatible with Python 3.12 and onward:
```bash
pip install illico -U
```

## How to use
This library exposes one single function that returns a `pd.DataFrame` holding p-value, u-statistic and fold-change for each (group, gene). Except the few points below, the function and its arguments should be self-explanatory:
1. It is **required** to indicate if the data you run the tests on underwent log1p transform. This only impacts the fold-change calculation and not the test results (p-values, u-stats). The choice was made to not try to guess this information, as those often lead to error-prone and potentially harmful rules of thumb.
2. By default, `illico.asymptotic_wilcoxon` will use what lies in `adata.X` to compute DE genes. If you want a specific layer to be used to perform the tests, you must specify it.

### DE genes compared to control cells
If you are working on single cell perturbation data:
```python
from illico import asymptotic_wilcoxon

adata = ad.read_h5ad('dataset.h5ad') # (n_cells, n_genes)
de_genes = asymptotic_wilcoxon(
       adata,
       # layer="Y", # <-- If you want tests to run not on .X, but a specific layer
       group_keys="perturbation",
       reference="non-targeting",
       is_log1p=[False|True], # <-- Specify if your data underwent log1p or not
       )
```

The resulting dataframe contains `n_perturbations * n_genes` rows and three columns: `(p_value, statistic, fold_change)`. In this case, the wilcoxon rank-sum test is performed between cells perturbed with perturbation *p_i* and control cells, for each *p_i*.
### DE genes for clustering analyses
Let's say your `.obs` contains a clustering variable, assigning a label to each cell.
```python
from illico import asymptotic_wilcoxon

adata = ad.read_h5ad('dataset.h5ad') # (n_cells, n_genes)
adata.obs["cluster"] = ...
de_genes = asymptotic_wilcoxon(adata, group_keys="cluster", reference=None, is_log1p=[False|True])
```
In this case, the resulting dataframe contains `n_perturbations * n_genes` rows and the same three columns: `(p_value, statistic, fold_change)`. In this case, the wilcoxon rank-sum test is performed between cells belonging to cluster *c_i* and all the other cells (one-versus-the-rest), for all *c_i*.

<!-- ### I am used to `scanpy`, how to make use of `illico` ?
In this case, you can replace your usual `sc.tl.rank_genes_groups(adata, groupby="...", reference="...", method="wilcoxon", tie_correct=True)` by:
```python
from illico.utils.sc import scanpy_port_asymptotic_wilcoxon
scanpy_port_asymptotic_wilcoxon(adata, group_keys="perturbation", reference="non-targeting", is_log1p=[False|True])
```
:warning: As of version XXX, `scanpy` lets the user decide to tie correct or not. `illico` only implements tie-corrected wilcoxon rank-sum tests. -->

### `illico` is not faster than `scanpy` or `pdex`, is there a bug ?
`illico` relies on a few optimization tricks to be faster than other existing tools. It is very possible that for some reason, the specific layout of your dataset (very small control population, very low sparsity, very small amount of distinct values) result in those tricks being effect-less, or less effective than observed on the datasets used to develop & benchmark `illico`. It is also very possible that because of those, other solutions end up faster than `illico` ! If this is your case, please open a issue describing your situation.

### `illico`'s results (p-values or fold-change) does not match `pdex` or `scanpy`.
Please open an issue, but before that: make sure that you are running **asymptotic** wilcoxon rank-sum tests as this is the only test exposed by `illico`.
- `pdex` relies on `scipy.stats.mannwhitneyu` that runs exact (non asymptotic) only when there are 8 values in both groups combined, and no ties.
- `scanpy` offers the possibility to run non-tie-corrected wilcoxon rank-sum tests, make sure this is disabled by passing `tie_correct=True`.
- Also, `illico` uses continuity correction which is the best practice.

### What about normalization and log1p
1. `illico` does not care about your data being normalized or not, it is up to you to apply the preprocessing of your choice before running the tests. It is expected that `illico` is slower if ran on total-count normalized data by a factor ~2. This is because if applied on non total-count normalized data, sorting relies on radix sort which is faster than the usual quicksort (that is used if testing total-count normalized data).
2. In order to avoid any unintended conversion, or relie on failure-prone rules of thumb, **`illico` requires the user to indicate if the input data is log1p or not**. This is only used to compute appropriate fold-change, and does not impact test (p-value and statistic) results.

### What if my adata does not fit in memory ?
Optimizing this use case is highly non-trivial as efficiently chunking CSR or CSC matrices is much more complex than running `adata[:, idxs]`. Ran on a CSR matrix, this command will load (temporarily) the entirety of the indices in RAM, resulting in a memory footprint almost equivalent to loading everything at once, on top of being extremely slow.
1. If your adata holds the expression matrix in a dense array, `illico` will work on it transparently because batch-based by design.
2. If your adata holds the expression matrix in a sparse (CSC or CSR) array, you have no other choice than manually chunking your array before running `illico` on batches. But, again, in this case I would advice to fallback to other solutions like `rapids-singlecell`.

## How it works
The rank-sum tests performed by `illico` are classical, asymptotic, rank-sum tests. No approximation nor assumption is done. `Illico` relies on a few optimization tricks that are non-exhaustively listed below:
1. 🧀 Sparse first: if the input data is sparse, that can be a lot less values to sort. Instead of converting it to dense, `illico` will only sort and rank non-zero values, and adjust rank-sums and tie sums later on with missing zeros.
2. 🗑️ Memory-conscious: ranking and sorting values across groups often requires to slice and convert the data numerous times, especially for CSC or CSR data. Memory allocations are minimized and optimized so as to ensure better scalability and lower overall memory footprint.
3. :brain: Sort controls only once: for the one-versus-reference use case, `illico` takes care of not repeatdly sorting the control values. Controls are sorted only once, after what each "perturbation" chunk is sorted, and the two sorted sub-arrays are merged (linear cost). Because there are often much more control cells than perturbed cells, this is a huge economy of processing.
4. :loop: Vectorize everything: for the one-versus-ref use case, `illico` performs one single sorting of the whole batch (all groups combined) and sums ranks for each group in a vectorized manner. This allows to sort only once instead of repeatedly performing `scipy.stats.mannwhitneyu` on all-but-group-*g* and group-*g*, for all *g* - involving one sorting each.
4. :snake: Generally speaking, `illico` relies heavily on `numba`'s JIT kernels to ensure GIL-free operations and efficient vectorization.

## Benchmarks
### Benchmarking against other solutions
In order for benchmarks to run in a reasonable amount of time, the timings reported below were obtained by running each solution on **a subset of each cell line** (20% of the genes). All solutions were find to scale linearly with the number of genes (columns in the adata). Extrapolating (x5) the elapsed times below will approximate runtime of those solutions on the whole datasets. Numbers in parenthesis report the multiplicative factor versus the fastest solution of each benchmark. A "benchmark" is defined by:
1. The cell line (K562 essential, RPE1, Hep-G2, Jurkat) used as input.
1. The data format (CSR, or dense) used to contain the expression matrix.
2. The test performed: OVO (`reference="non-targeting"`) or OVR (`reference=None`).

:bulb: Keep in mind that `pdex` does not implement *OVR* test.
```bash
------------------------------- benchmark 'hepg2-csr-ovo': 3 tests ------------------------------
Name (time in s)                                                                   Mean
-------------------------------------------------------------------------------------------------
test_speed_benchmark[hepg2-csr-20%-illico-ovo-nthreads=8] (0003_illico-)         4.1165 (1.0)
test_speed_benchmark[hepg2-csr-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)       369.3903 (89.73)
test_speed_benchmark[hepg2-csr-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       1,545.4044 (375.42)
-------------------------------------------------------------------------------------------------

------------------------------- benchmark 'hepg2-csr-ovr': 2 tests ------------------------------
Name (time in s)                                                                   Mean
-------------------------------------------------------------------------------------------------
test_speed_benchmark[hepg2-csr-20%-illico-ovr-nthreads=8] (0003_illico-)         3.6700 (1.0)
test_speed_benchmark[hepg2-csr-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     3,687.2432 (>1000.0)
-------------------------------------------------------------------------------------------------

------------------------------- benchmark 'hepg2-dense-ovo': 3 tests ------------------------------
Name (time in s)                                                                     Mean
---------------------------------------------------------------------------------------------------
test_speed_benchmark[hepg2-dense-20%-illico-ovo-nthreads=8] (0003_illico-)         6.4324 (1.0)
test_speed_benchmark[hepg2-dense-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)       352.2373 (54.76)
test_speed_benchmark[hepg2-dense-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       1,843.8692 (286.65)
---------------------------------------------------------------------------------------------------

------------------------------- benchmark 'hepg2-dense-ovr': 2 tests ------------------------------
Name (time in s)                                                                     Mean
---------------------------------------------------------------------------------------------------
test_speed_benchmark[hepg2-dense-20%-illico-ovr-nthreads=8] (0003_illico-)         4.0817 (1.0)
test_speed_benchmark[hepg2-dense-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     4,194.9233 (>1000.0)
---------------------------------------------------------------------------------------------------

------------------------------ benchmark 'jurkat-csr-ovo': 3 tests -------------------------------
Name (time in s)                                                                    Mean
--------------------------------------------------------------------------------------------------
test_speed_benchmark[jurkat-csr-20%-illico-ovo-nthreads=8] (0003_illico-)         6.3750 (1.0)
test_speed_benchmark[jurkat-csr-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)     1,164.5936 (182.68)
test_speed_benchmark[jurkat-csr-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       3,204.1846 (502.62)
--------------------------------------------------------------------------------------------------

------------------------------ benchmark 'jurkat-csr-ovr': 2 tests -------------------------------
Name (time in s)                                                                    Mean
--------------------------------------------------------------------------------------------------
test_speed_benchmark[jurkat-csr-20%-illico-ovr-nthreads=8] (0003_illico-)         4.8208 (1.0)
test_speed_benchmark[jurkat-csr-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     6,489.7840 (>1000.0)
--------------------------------------------------------------------------------------------------

------------------------------ benchmark 'jurkat-dense-ovo': 3 tests -------------------------------
Name (time in s)                                                                      Mean
----------------------------------------------------------------------------------------------------
test_speed_benchmark[jurkat-dense-20%-illico-ovo-nthreads=8] (0003_illico-)         8.7321 (1.0)
test_speed_benchmark[jurkat-dense-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)       958.6772 (109.79)
test_speed_benchmark[jurkat-dense-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       2,903.1847 (332.47)
----------------------------------------------------------------------------------------------------

------------------------------ benchmark 'jurkat-dense-ovr': 2 tests -------------------------------
Name (time in s)                                                                      Mean
----------------------------------------------------------------------------------------------------
test_speed_benchmark[jurkat-dense-20%-illico-ovr-nthreads=8] (0003_illico-)         6.0360 (1.0)
test_speed_benchmark[jurkat-dense-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     7,892.3868 (>1000.0)
----------------------------------------------------------------------------------------------------

------------------------------ benchmark 'k562-csr-ovo': 3 tests -------------------------------
Name (time in s)                                                                  Mean
------------------------------------------------------------------------------------------------
test_speed_benchmark[k562-csr-20%-illico-ovo-nthreads=8] (0003_illico-)         5.4187 (1.0)
test_speed_benchmark[k562-csr-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)       906.4330 (167.28)
test_speed_benchmark[k562-csr-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       2,628.7324 (485.12)
------------------------------------------------------------------------------------------------

------------------------------ benchmark 'k562-csr-ovr': 2 tests -------------------------------
Name (time in s)                                                                  Mean
------------------------------------------------------------------------------------------------
test_speed_benchmark[k562-csr-20%-illico-ovr-nthreads=8] (0003_illico-)         7.0503 (1.0)
test_speed_benchmark[k562-csr-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     8,083.1556 (>1000.0)
------------------------------------------------------------------------------------------------

------------------------------ benchmark 'k562-dense-ovo': 3 tests -------------------------------
Name (time in s)                                                                    Mean
--------------------------------------------------------------------------------------------------
test_speed_benchmark[k562-dense-20%-illico-ovo-nthreads=8] (0003_illico-)         7.0818 (1.0)
test_speed_benchmark[k562-dense-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)       750.8397 (106.02)
test_speed_benchmark[k562-dense-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       2,872.8148 (405.66)
--------------------------------------------------------------------------------------------------

------------------------------ benchmark 'k562-dense-ovr': 2 tests -------------------------------
Name (time in s)                                                                    Mean
--------------------------------------------------------------------------------------------------
test_speed_benchmark[k562-dense-20%-illico-ovr-nthreads=8] (0003_illico-)         5.3919 (1.0)
test_speed_benchmark[k562-dense-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     8,554.6306 (>1000.0)
--------------------------------------------------------------------------------------------------

------------------------------ benchmark 'rpe1-csr-ovo': 3 tests -------------------------------
Name (time in s)                                                                  Mean
------------------------------------------------------------------------------------------------
test_speed_benchmark[rpe1-csr-20%-illico-ovo-nthreads=8] (0003_illico-)         5.1816 (1.0)
test_speed_benchmark[rpe1-csr-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)     1,059.5642 (204.49)
test_speed_benchmark[rpe1-csr-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       2,495.4590 (481.60)
------------------------------------------------------------------------------------------------

------------------------------ benchmark 'rpe1-csr-ovr': 2 tests -------------------------------
Name (time in s)                                                                  Mean
------------------------------------------------------------------------------------------------
test_speed_benchmark[rpe1-csr-20%-illico-ovr-nthreads=8] (0003_illico-)         3.8255 (1.0)
test_speed_benchmark[rpe1-csr-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     8,133.4382 (>1000.0)
------------------------------------------------------------------------------------------------

------------------------------ benchmark 'rpe1-dense-ovo': 3 tests -------------------------------
Name (time in s)                                                                    Mean
--------------------------------------------------------------------------------------------------
test_speed_benchmark[rpe1-dense-20%-illico-ovo-nthreads=8] (0003_illico-)         8.2337 (1.0)
test_speed_benchmark[rpe1-dense-20%-scanpy-ovo-nthreads=8] (0001_scanpy-)       989.9742 (120.23)
test_speed_benchmark[rpe1-dense-20%-pdex-ovo-nthreads=8] (0002_pdex-sp)       2,435.5715 (295.81)
--------------------------------------------------------------------------------------------------

------------------------------ benchmark 'rpe1-dense-ovr': 2 tests -------------------------------
Name (time in s)                                                                    Mean
--------------------------------------------------------------------------------------------------
test_speed_benchmark[rpe1-dense-20%-illico-ovr-nthreads=8] (0003_illico-)         4.6674 (1.0)
test_speed_benchmark[rpe1-dense-20%-scanpy-ovr-nthreads=8] (0001_scanpy-)     7,720.4164 (>1000.0)
--------------------------------------------------------------------------------------------------
```

### Scalability
TODO: this could clearly be improved with a smarter batching strategy
`illico` scales reasonably well with your compute budget. Find below the processing time of the K562-essential dataset for both OVO and OVR tests, while increasing the number of threads used. Similarly as before, a benchmark is defined by:
1. The data format (CSR, or dense) used to contain the expression matrix.
2. The test performed: OVO (`reference="non-targeting"`) or OVR (`reference=None`).

```bash
---------------------- benchmark 'k562-csr-ovo': 4 tests -----------------------
Name (time in s)                                                  Mean
--------------------------------------------------------------------------------
test_speed_benchmark[k562-csr-100%-illico-ovo-nthreads=8]      19.3962 (1.0)
test_speed_benchmark[k562-csr-100%-illico-ovo-nthreads=4]      31.2427 (1.61)
test_speed_benchmark[k562-csr-100%-illico-ovo-nthreads=2]      62.0832 (3.20)
test_speed_benchmark[k562-csr-100%-illico-ovo-nthreads=1]     129.3453 (6.67)
--------------------------------------------------------------------------------

---------------------- benchmark 'k562-csr-ovr': 4 tests ----------------------
Name (time in s)                                                 Mean
-------------------------------------------------------------------------------
test_speed_benchmark[k562-csr-100%-illico-ovr-nthreads=8]     14.9382 (1.0)
test_speed_benchmark[k562-csr-100%-illico-ovr-nthreads=4]     25.2831 (1.69)
test_speed_benchmark[k562-csr-100%-illico-ovr-nthreads=2]     48.5062 (3.25)
test_speed_benchmark[k562-csr-100%-illico-ovr-nthreads=1]     98.2664 (6.58)
-------------------------------------------------------------------------------

---------------------- benchmark 'k562-dense-ovo': 4 tests -----------------------
Name (time in s)                                                    Mean
----------------------------------------------------------------------------------
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=8]      29.6962 (1.0)
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=4]      53.4369 (1.80)
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=2]     100.3919 (3.38)
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=1]     208.2443 (7.01)
----------------------------------------------------------------------------------

---------------------- benchmark 'k562-dense-ovr': 4 tests -----------------------
Name (time in s)                                                    Mean
----------------------------------------------------------------------------------
test_speed_benchmark[k562-dense-100%-illico-ovr-nthreads=8]      19.3093 (1.0)
test_speed_benchmark[k562-dense-100%-illico-ovr-nthreads=4]      33.6427 (1.74)
test_speed_benchmark[k562-dense-100%-illico-ovr-nthreads=2]      63.1888 (3.27)
test_speed_benchmark[k562-dense-100%-illico-ovr-nthreads=1]     127.4927 (6.60)
----------------------------------------------------------------------------------
```
### Memory
TODO: Add memit for all solutions, remind that memory footprint grows linearly with number of threads for illico.
```
============================================================================== MEMRAY REPORT ===============================================================================
Allocation results for tests/test_asymptotic_wilcoxon.py::test_memory_benchmark[pdex-ovo-csc-nthreads=1] at the high watermark

         📦 Total memory allocated: 1.5GiB
         📏 Total allocations: 34
         📊 Histogram of allocation sizes: |█▄  ▂|
         🥇 Biggest allocating functions:
                - __init__:/Users/remydubois/.pyenv/versions/3.13.7/lib/python3.13/multiprocessing/shared_memory.py:117 -> 763.1MiB
                - _process_toarray_args:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/scipy/sparse/_base.py:1530 -> 763.1MiB
                - func:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/pandas/core/arrays/categorical.py:163 -> 31.9MiB
                - parallel_differential_expression:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/pdex/_single_cell.py:374 -> 25.4KiB


Allocation results for tests/test_asymptotic_wilcoxon.py::test_memory_benchmark[scanpy-ovo-csc-nthreads=1] at the high watermark

         📦 Total memory allocated: 524.3MiB
         📏 Total allocations: 260
         📊 Histogram of allocation sizes: |█▆ ▃▂|
         🥇 Biggest allocating functions:
                - _check_nonnegative_integers_in_mem:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/scanpy/_utils/__init__.py:823 -> 392.1MiB
                - _check_nonnegative_integers_in_mem:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/scanpy/_utils/__init__.py:823 -> 98.0MiB
                - select_groups:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/scanpy/_utils/__init__.py:849 -> 31.9MiB
                - take:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/pandas/core/algorithms.py:1239 -> 1.7MiB
                - _take_nd_ndarray:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/pandas/core/array_algos/take.py:155 -> 432.2KiB


Allocation results for tests/test_asymptotic_wilcoxon.py::test_memory_benchmark[illico-ovo-csc-nthreads=1] at the high watermark

         📦 Total memory allocated: 114.1MiB
         📏 Total allocations: 32
         📊 Histogram of allocation sizes: |▄█ ▄ |
         🥇 Biggest allocating functions:
                - ovo_mwu_over_contiguous_col_chunk:/Users/remydubois/Documents/perso/repos/illico/illico/ovo/__init__.py:36 -> 106.7MiB
                - _stack_arrays:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/pandas/core/internals/managers.py:2271 -> 1.8MiB
                - tolist:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/pandas/core/arrays/base.py:2078 -> 1.7MiB
                - _wrapit:/Users/remydubois/Documents/perso/repos/illico/.venv/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46 -> 1.7MiB
                - encode_and_count_groups:/Users/remydubois/Documents/perso/repos/illico/illico/utils/groups.py:25 -> 1.7MiB
```
## Why illico
The name *illico* is a wordplay inspired by the R package `presto` (now the Wilcoxon rank-sum test backend in Seurat). Aside from this naming reference, there is no affiliation or intended equivalence between the two. `illico` was developed independently, and although the statistical methodology may be similar, it was not designed to reproduce `presto`’s results.

# Contributing
All contributions are welcome through merge requests. Developers are highly encouraged to rely on `tox` as the testing process is quite cumbersome.
## Testing
The reason to be of this package is its speed, hence the need for extensive speed benchmarks in order to compare it exhaustively and accurately against existing solutions. `tox` is used to manage tests and testing environments.
```bash
pip install tox # this can be system-wide, no need to install it within an environment
```
:bulb: The test suite below can be very long, especially the benchmarks (up to 48 hours). All "bench-" tox commands can be appended with the `-quick` suffix ensuring they are ran on 1 gene (column) of the benchmark data, just to make sure everything runs correctly. Example:
```bash
tox -e bench-all-quick # This will run speed and memory benchmarks for illico, scanpy and pdex
# OR:  tox -e bench-illico-quick # This will run speed and memory benchmarks for illico only
# OR :tox -e bench-ref-quick # This will run speed and memory benchmarks for scanpy and pdex only
```
Appending the `-quick` suffix will not write any result file or json inside the `.benchmarks` or `.memray` folders (that are versioned). Instead, benchmark result files will be written to `/tmp`.
In this case, make sure to run `tox -e memray-stats /tmp`

### Unit testing
Those tests are simply used to ensure the p-values and fold-change returned by `illico` are correct, should be quick to run:
`tox -e unit-tests`
:warning: Those tests do not run `-quick` as they use synthetic data that results in much shorter runtime.

### Speed benchmarks
Speed benchmarks are ran against: `pdex` and `scanpy` as references. Those benchmarks take **a lot** of time (>10 hours on 8 CPUs) so they should not be re-ran for every new PR or release. However, if needed:
```bash
tox -e speed-bench-ref # Run speed benchmarks for scanpy and pdex, should not be re-ran, ideally.
```
Before issuing a new PR, in order to see if the updated code does not decrease speed performance, make sure to run:
```bash
tox -e speed-bench-illico #-quick
```
:bulb: Because benchmark performance depends on the testing environment (type of machine or OS), it is recommended to run this benchmark from `main` on your machine as well. This will give you a clear comparison point apple-to-apple.
Once the benchmarks have ran, you can cat the benchmark results in terminal with:
```bash
tox -e speed-bench-compare
```

### Memory benchmark
Similar remarks as for the speed benchmarks:
```bash
tox -e memory-bench-ref # Should not be re-ran, ideally
tox -e memory-bench-illico # Should be re-ran before every new PR
tox -e memray-stats
```

# Other tools available
1. `scanpy` also implements OVO and OVR asymptotic wilcoxon rank-sum tests.
2. `pdex` only implements OVO wilcoxon rank-sum tests.
3. As of December 2025, `rapids-singlecell` has a pending PR adding a `rank_genes_groups` feature. I could not benchmark this solution as I had no GPU available, but it is expected that it runs at least as fast as `illico`, because GPU-based.
