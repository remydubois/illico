# Normalization and log1p

1. `illico` does not care about your data being normalized or not, it is up to you to apply the preprocessing of your choice before running the tests. It is expected that `illico` is slower if ran on total-count normalized data by a factor ~2. This is because if applied on non total-count normalized data, sorting relies on radix sort which is faster than the usual quicksort (that is used if testing total-count normalized data).
2. In order to avoid any unintended conversion, or relie on failure-prone rules of thumb, **`illico` requires the user to indicate if the input data is log1p or not**. This is only used to compute appropriate fold-change, and does not impact test (p-value and statistic) results.
