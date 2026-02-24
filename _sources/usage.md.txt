# How to use

This library exposes one single function that returns a `pd.DataFrame` holding p-value, u-statistic and fold-change for each (group, gene). Except the few points below, the function and its arguments should be self-explanatory:

1. It is **required** to indicate if the data you run the tests on underwent log1p transform. This only impacts the fold-change calculation and not the test results (p-values, u-stats). The choice was made to not try to guess this information, as those often lead to error-prone and potentially harmful rules of thumb.
2. By default, `illico.asymptotic_wilcoxon` will use what lies in `adata.X` to compute DE genes. If you want a specific layer to be used to perform the tests, you must specify it.
3. By default again, `illico.asymptotic_wilcoxon` will apply continuity correction and tie correction factors. This is controllable with the `use_continuity` and `tie_correct` arguments.

## DE genes compared to control cells

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

## DE genes for clustering analyses

Let's say your `.obs` contains a clustering variable, assigning a label to each cell.

```python
from illico import asymptotic_wilcoxon

adata = ad.read_h5ad('dataset.h5ad') # (n_cells, n_genes)
adata.obs["cluster"] = ...
de_genes = asymptotic_wilcoxon(adata, group_keys="cluster", reference=None, is_log1p=[False|True])
```

In this case, the resulting dataframe contains `n_clusters * n_genes` rows and the same three columns: `(p_value, statistic, fold_change)`. In this case, the wilcoxon rank-sum test is performed between cells belonging to cluster *c_i* and all the other cells (one-versus-the-rest), for all *c_i*.
