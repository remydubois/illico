# Benchmarks

## Benchmarking against other solutions

A *benchmark* is defined by:

1. The cell line (K562 essential, RPE1, Hep-G2, Jurkat) used as input.
2. The data format (CSR, or dense) used to contain the expression matrix.
3. The test performed: OVO (`reference="non-targeting"`) or OVR (`reference=None`).


<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/method-runtimes-comparison.png?raw=true" width="100%" />
  <figcaption>Runtime comparison for scanpy, pdex and illico on four cell lines.</figcaption>
</center>

## Impact of normalization
Applying total count normalization (`sc.pp.normalize`) to your count matrix will make `illico` run 2x-3x slower. This is essentially due to two things:
1. Non-round values forbid the use of fast sorting algorithm like radix or histogram sort, and fallback to the slower quicksort.
2. Total count normalization creates more tie blocks per column (more unique values per group), hence a longer processing time (as `illico` processes data tie block by tie block).
Version `0.7.0` helped this a lot for the OVO case, but a slowdown remains between raw (integer count) and TC-norm data. In all cases, `illico` remains >200x faster than `sc.tl.rank_genes_groups(method="wilcoxon")` (scanpy < v1.13).

<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/k562_ovo_throughput.png?raw=true" width="100%" />
  <figcaption>Illico and other solutions' throughput (reference="control") in different data format and normalization scenarii.</figcaption>
</center>

## Scalability

`illico` scales reasonably well with your compute budget, with a quasi-linear speedup up to 16 threads.

<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/illico-scaling-rust.jpg?raw=true" width="100%" />
  <figcaption>Throughput of illico with increasing compute budget, compared to a perfect scaling.</figcaption>
</center>

## Impact of data format (dense or sparse)
From v0.7.0, processing count matrices stored as CSR is not necessarily much faster than those stored dense, and can be slower in some cases. This is due to the "data wrangling" part (chunking columns to form batches, slicing groups for the OVO test) that is more expensive for CSR matrices than dense arrays. Depending on the sparsity (% of zeros) of your count matrices, and whether the data is total count normalized (see section above on normalization), one storage format is faster than the other.
> TL/DR: on TC-norm data, dense and CSR run almost equally fast regardless the sparsity. On raw data, CSR runs faster only when sparsity is above 50%.
Note: the chunking of dense arrays was made faster in v0.7.0, making the dense path faster. The (sparse and dense) mannwhitney test was also made faster, increasing the weight of data wrangling in the overall execution of `asymptotic_wilcoxon`. In all cases, v0.7.0 is 3 to 5 times faster for the OVO case than v0.6.0 in the same setup (data format, normalization).

<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/elapsed_time_vs_sparsity.png?raw=true" width="100%" />
  <figcaption>Impact of sparsity on runtime for dense and CSR raw or TC-norm count matrices.</figcaption>
</center>
