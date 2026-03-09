# Benchmarks

## Benchmarking against other solutions

In order for benchmarks to run in a reasonable amount of time, the timings reported below were obtained by running each solution on **a subset of each cell line** (20% of the genes). All solutions were find to scale linearly with the number of genes (columns in the adata). Extrapolating (x5) the elapsed times below will approximate runtime of those solutions on the whole datasets. Numbers in parenthesis report the multiplicative factor versus the fastest solution of each benchmark. A "benchmark" is defined by:

1. The cell line (K562 essential, RPE1, Hep-G2, Jurkat) used as input.
2. The data format (CSR, or dense) used to contain the expression matrix.
3. The test performed: OVO (`reference="non-targeting"`) or OVR (`reference=None`).

💡 Keep in mind that `pdex` does not implement *OVR* test.

<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/method-runtimes-comparison.png?raw=true" width="100%" />
  <figcaption>Runtime comparison for scanpy, pdex and illico on four cell lines.</figcaption>
</center>

## Scalability

`illico` scales reasonably well with your compute budget. On the K562-essential dataset spanning 8 threads instead of 1 brings a 7-folds speedup.

```bash
---------------------- benchmark 'k562-dense-ovo': 4 tests -----------------------
Name (time in s)                                                    Mean
----------------------------------------------------------------------------------
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=8]      29.6962 (1.0)
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=4]      53.4369 (1.80)
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=2]     100.3919 (3.38)
test_speed_benchmark[k562-dense-100%-illico-ovo-nthreads=1]     208.2443 (7.01)
----------------------------------------------------------------------------------
```
