# Illico
## Overview
Illico is a python library performing blazing fast asymptotic wilcoxon rank-sum tests (same as `scanpy.tl.rank_genes_groups(… tie_correct=True)`), useful for single-cell RNASeq data analyses and processing. Illico's features are:
1. :rocket: Blazing fast: On K562 (essential) dataset (~300k cells, 8k genes, 2k perturbations), `illico` computes DE genes (with `reference="non-targeting"`) in a mere 30 seconds. That's more than 100 times faster than both `pdex` or `scanpy` with the same compute ressources (8 CPUs).
2. :diamond_shape_with_a_dot_inside: No compromise: On VCC's H1 dataset, `illico`'s p-values matched `scipy.stats.mannwhitneyu` up to a relative difference of 1.e-12, and an absolute tol of 0.
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

## Installation
`illico` can be installed via pip, compatible with Python 3.11 and onward:
```bash
pip install illico -U
```

## How to use
This library exposes one single function that returns a `pd.DataFrame` holding p-value, u-statistic and fold-change for each (group, gene). Except the few points below, the function and its arguments should be self-explanatory:
1. It is **required** to indicate if the data you run the tests on underwent log1p transform. This only impacts the fold-change calculation and not the test results (p-values, u-stats). The choice was made to not try to guess this information, as those often lead to error-prone and potentially harmful rules of thumb.
2. By default, `illico.asymptotic_wilcoxon` will use what lies in `adata.X` to compute DE genes. If you want a specific layer to be used to perform the tests, you must specify it.
3. As of December 2025, it only exposes two-sided and continuity-corrected Mann-Whitney tests. Adding non-corrected and/or one-sided tests could be done easily of need arise.

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

## How it works
The rank-sum tests performed by `illico` are classical, asymptotic, rank-sum tests. No approximation nor assumption is done. `Illico` relies on a few optimization tricks that are non-exhaustively listed below:
1. 🧀 Sparse first: if the input data is sparse, that can be a lot less values to sort. Instead of converting it to dense, `illico` will only sort and rank non-zero values, and adjust rank-sums and tie sums later on with missing zeros.
2. 🗑️ Memory-conscious: ranking and sorting values across groups often requires to slice and convert the data numerous times, especially for CSC or CSR data. Memory allocations are minimized and optimized so as to ensure better scalability and lower overall memory footprint.
3. :brain: Sort controls only once: for the one-versus-reference use case, `illico` takes care of not repeatdly sorting the control values. Controls are sorted only once, after what each "perturbation" chunk is sorted, and the two sorted sub-arrays are merged (linear cost). Because there are often much more control cells than perturbed cells, this is a huge economy of processing.
4. :loop: Vectorize everything: for the one-versus-ref use case, `illico` performs one single sorting of the whole batch (all groups combined) and sums ranks for each group in a vectorized manner. This allows to sort only once instead of repeatedly performing `scipy.stats.mannwhitneyu` on all-but-group-*g* and group-*g*, for all *g* - involving one sorting each.
4. :snake: Generally speaking, `illico` relies heavily on `numba`'s JIT kernels to ensure GIL-free operations and efficient vectorization.

## Benchmarks
### Benchmarking against other solutions
In order for benchmarks to run in a reasonable amount of time, the timings reported below were obtained by running each solution on **a subset of k562-essential** (20% of the genes). All solutions were find to scale linearly with the number of genes (columns in the adata). Extrapolating the value below will approximate runtime of those solutions on the whole dataset (8k genes). Benchmarks below depend on:
1. The cell line (K562 essential, RPE1, Hep-G2, Jurkat).
1. The data format (CSR, or dense)
2. The test performed: OVO (`reference="non-targeting"`) or OVR (`reference=None`).
```
----------------------------------------------- benchmark 'ovo-csr_matrix': 3 tests -----------------------------------------------
Name (time in s)                                                              Min                Mean                 Max
-----------------------------------------------------------------------------------------------------------------------------------
test_speed_benchmark[csr-small-illico-ovo-nthreads=8] (0010_illico-)       3.4252 (1.0)        3.4252 (1.0)        3.4252 (1.0)
test_speed_benchmark[csr-small-scanpy-ovo-nthreads=8] (0008_scanpy-)     300.7754 (87.81)    300.7754 (87.81)    300.7754 (87.81)
test_speed_benchmark[csr-small-pdex-ovo-nthreads=8] (0009_pdex-sm)       951.5820 (277.82)   951.5820 (277.82)   951.5820 (277.82)
-----------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------- benchmark 'ovo-ndarray': 3 tests -------------------------------------------------
Name (time in s)                                                                Min                Mean                 Max
-------------------------------------------------------------------------------------------------------------------------------------
test_speed_benchmark[dense-small-illico-ovo-nthreads=8] (0010_illico-)       2.6945 (1.0)        2.6945 (1.0)        2.6945 (1.0)
test_speed_benchmark[dense-small-scanpy-ovo-nthreads=8] (0008_scanpy-)     251.7694 (93.44)    251.7694 (93.44)    251.7694 (93.44)
test_speed_benchmark[dense-small-pdex-ovo-nthreads=8] (0009_pdex-sm)       730.6372 (271.16)   730.6372 (271.16)   730.6372 (271.16)
-------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------- benchmark 'ovr-csr_matrix': 2 tests --------------------------------------------------
Name (time in s)                                                                Min                  Mean                   Max
-----------------------------------------------------------------------------------------------------------------------------------------
test_speed_benchmark[csr-small-illico-ovr-nthreads=8] (0010_illico-)         2.2169 (1.0)          2.2169 (1.0)          2.2169 (1.0)
test_speed_benchmark[csr-small-scanpy-ovr-nthreads=8] (0008_scanpy-)     1,876.2998 (846.35)   1,876.2998 (846.35)   1,876.2998 (846.35)
-----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------- benchmark 'ovr-ndarray': 2 tests ----------------------------------------------------
Name (time in s)                                                                  Min                  Mean                   Max
-------------------------------------------------------------------------------------------------------------------------------------------
test_speed_benchmark[dense-small-illico-ovr-nthreads=8] (0010_illico-)         2.8279 (1.0)          2.8279 (1.0)          2.8279 (1.0)
test_speed_benchmark[dense-small-scanpy-ovr-nthreads=8] (0008_scanpy-)     2,276.9517 (805.16)   2,276.9517 (805.16)   2,276.9517 (805.16)
-------------------------------------------------------------------------------------------------------------------------------------------
```
1. Comparing frameworks on full H1 (OVO) with 8 threads: Scanpy (>2.5h), pdex (45mins), illico (<1min) so a total of
Below are shown benchmarks obtained on the first 904 genes of VCC's H1 dataset. `904` genes is `1/20` (5%) of the `18,080` gene sequenced in H1. All solutions are expected to scale linearly with number of genes. The column that matters is the *median* elapsed time.
1. For the OVO use case (one-versus-one, test each perturbation against the control cells): `illico` is between 50 to 60 times faster than `scanpy` and `pdex` when ran on sparse data, and around 20 times faster on dense data.
2. For the OVR use case (one-versus-the-rest, clustering analyses): `illico` is approx `150` times faster than scanpy when ran on sparse data, and 60 times faster when ran on dense data. `pdex` does not implement OVR test.

Ran on one thread: benchmark n°37
```
--------------------------- benchmark 'ovo-csc': 3 tests ---------------------------
Name (time in s)                                Median            Rounds  Iterations
------------------------------------------------------------------------------------
         test_benchmark[illico-ovo-csc-1]       5.4017 (1.0)           1           1
         test_benchmark[scanpy-ovo-csc-1]     300.2536 (55.58)         1           1
         test_benchmark[pdex-ovo-csc-1]       306.8924 (56.81)         1           1
------------------------------------------------------------------------------------

--------------------------- benchmark 'ovo-csr': 3 tests ---------------------------
Name (time in s)                                Median            Rounds  Iterations
------------------------------------------------------------------------------------
         test_benchmark[illico-ovo-csr-1]       4.7456 (1.0)           1           1
         test_benchmark[scanpy-ovo-csr-1]     248.6532 (52.40)         1           1
         test_benchmark[pdex-ovo-csr-1]       299.7366 (63.16)         1           1
------------------------------------------------------------------------------------

--------------------------- benchmark 'ovo-dense': 3 tests ---------------------------
Name (time in s)                                  Median            Rounds  Iterations
--------------------------------------------------------------------------------------
         test_benchmark[illico-ovo-dense-1]      13.3544 (1.0)           1           1
         test_benchmark[scanpy-ovo-dense-1]     238.2492 (17.84)         1           1
         test_benchmark[pdex-ovo-dense-1]       306.4146 (22.94)         1           1
--------------------------------------------------------------------------------------

--------------------------- benchmark 'ovr-csc': 2 tests ---------------------------
Name (time in s)                                Median            Rounds  Iterations
------------------------------------------------------------------------------------
         test_benchmark[illico-ovr-csc-1]       4.3178 (1.0)           1           1
         test_benchmark[scanpy-ovr-csc-1]     693.2938 (160.57)        1           1
------------------------------------------------------------------------------------

--------------------------- benchmark 'ovr-csr': 2 tests ---------------------------
Name (time in s)                                Median            Rounds  Iterations
------------------------------------------------------------------------------------
         test_benchmark[illico-ovr-csr-1]       4.7285 (1.0)           1           1
         test_benchmark[scanpy-ovr-csr-1]     673.5518 (142.45)        1           1
------------------------------------------------------------------------------------

--------------------------- benchmark 'ovr-dense': 2 tests ---------------------------
Name (time in s)                                  Median            Rounds  Iterations
--------------------------------------------------------------------------------------
         test_benchmark[illico-ovr-dense-1]      10.5175 (1.0)           1           1
         test_benchmark[scanpy-ovr-dense-1]     674.8101 (64.16)         1           1
--------------------------------------------------------------------------------------
Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
=========== 18 passed, 33 deselected, 13113 warnings in 11358.76s (3:09:18) ===========
```
Ran on two threads: benchmark n°38
```
----------------------- benchmark 'ovo-csc': 3 tests ----------------------
Name (time in s)                       Median            Rounds  Iterations
---------------------------------------------------------------------------
test_benchmark[illico-ovo-csc-2]       3.4325 (1.0)           1           1
test_benchmark[pdex-ovo-csc-2]       170.3768 (49.64)         1           1
test_benchmark[scanpy-ovo-csc-2]     198.0979 (57.71)         1           1
---------------------------------------------------------------------------

----------------------- benchmark 'ovo-csr': 3 tests ----------------------
Name (time in s)                       Median            Rounds  Iterations
---------------------------------------------------------------------------
test_benchmark[illico-ovo-csr-2]       3.0417 (1.0)           1           1
test_benchmark[pdex-ovo-csr-2]       171.4978 (56.38)         1           1
test_benchmark[scanpy-ovo-csr-2]     171.5105 (56.39)         1           1
---------------------------------------------------------------------------

----------------------- benchmark 'ovo-dense': 3 tests ----------------------
Name (time in s)                         Median            Rounds  Iterations
-----------------------------------------------------------------------------
test_benchmark[illico-ovo-dense-2]      11.9322 (1.0)           1           1
test_benchmark[scanpy-ovo-dense-2]     142.3267 (11.93)         1           1
test_benchmark[pdex-ovo-dense-2]       195.2699 (16.36)         1           1
-----------------------------------------------------------------------------

----------------------- benchmark 'ovr-csc': 2 tests ----------------------
Name (time in s)                       Median            Rounds  Iterations
---------------------------------------------------------------------------
test_benchmark[illico-ovr-csc-2]       2.8199 (1.0)           1           1
test_benchmark[scanpy-ovr-csc-2]     420.0390 (148.96)        1           1
---------------------------------------------------------------------------

----------------------- benchmark 'ovr-csr': 2 tests ----------------------
Name (time in s)                       Median            Rounds  Iterations
---------------------------------------------------------------------------
test_benchmark[illico-ovr-csr-2]       3.0468 (1.0)           1           1
test_benchmark[scanpy-ovr-csr-2]     418.4509 (137.34)        1           1
---------------------------------------------------------------------------

----------------------- benchmark 'ovr-dense': 2 tests ----------------------
Name (time in s)                         Median            Rounds  Iterations
-----------------------------------------------------------------------------
test_benchmark[illico-ovr-dense-2]       6.9618 (1.0)           1           1
test_benchmark[scanpy-ovr-dense-2]     409.9389 (58.88)         1           1
-----------------------------------------------------------------------------
```

### Scalability
Add scalability graphs for all solutions
```
------------------------------------------------------------------------------------------ benchmark 'ovo-csr': 3 tests ------------------------------------------------------------------------------------------
Name (time in s)                                        Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_speed_benchmark[illico-ovo-csr-nthreads=8]     24.0572 (1.0)      25.0281 (1.0)      24.4648 (1.0)      0.3640 (1.0)      24.4720 (1.0)      0.4360 (1.0)           2;0  0.0409 (1.0)           5           1
test_speed_benchmark[illico-ovo-csr-nthreads=4]     43.1721 (1.79)     44.6799 (1.79)     43.8410 (1.79)     0.5689 (1.56)     43.8020 (1.79)     0.7642 (1.75)          2;0  0.0228 (0.56)          5           1
test_speed_benchmark[illico-ovo-csr-nthreads=2]     84.0295 (3.49)     85.3700 (3.41)     84.5683 (3.46)     0.5363 (1.47)     84.5695 (3.46)     0.7881 (1.81)          2;0  0.0118 (0.29)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
### Memory
Add memit for all solutions, remind that memory footprint grows linearly with number of threads for illico.
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
