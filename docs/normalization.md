# Normalization and log1p

1. `illico` does not care about your data being normalized or not, it is up to you to apply the preprocessing of your choice before running the tests. It is expected that `illico` is slower if ran on total-count normalized data by a factor ~2.5. See [the benchmark section](https://remydubois.github.io/illico/benchmarks.html#impact-of-normalization) for details.

2. In order to avoid any unintended conversion, or rely on failure-prone rules of thumb, **`illico` requires the user to indicate if the input data is log1p or not**. This is only used to compute appropriate fold-change, and does not impact test (p-value and statistic) results. Note also that unlike scanpy, if data is log1p, fold-change is computed by exponentiating before aggregation by default. This behavior can be changed by passing `exp_post_agg=False` to `asymptotic_wilcoxon`. See the [results](#results) section for more details on that.
