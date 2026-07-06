# How to use

This library exposes one single function that returns either a `pd.DataFrame` or a dictionaries of record arrays holding p-value, u-statistic and fold-change for each (group, gene). Except the few points below, the function and its arguments should be self-explanatory:

1. It is **required** to indicate if the data you run the tests on underwent log1p transform. This only impacts the fold-change calculation and not the test results (p-values, u-stats). The choice was made to not try to guess this information, as those often lead to error-prone and potentially harmful rules of thumb.
2. By default, `illico.asymptotic_wilcoxon` will use what lies in `adata.X` to compute DE genes. If you want a specific layer to be used to perform the tests, you must specify it.
3. By default again, `illico.asymptotic_wilcoxon` will apply continuity correction and tie correction factors. This is controllable with the `use_continuity` and `tie_correct` arguments.
4. If you are coming from the `scanpy` ecosystem and want a drop-in replacement of `sc.tl.rank_genes_groups(…, method="wilcoxon")`, you can set `return_as_scanpy=True` when calling `illico.asymptotic_wilcoxon`. This will return a dictionary formatted for Scanpy's `rank_genes_groups` results, which you can then attach to `adata.uns["rank_genes_groups"]` and use with the rest of your Scanpy workflow as usual. See last section.
5. The `groups` argument serves the same purpose as `scanpy`'s `rank_genes_groups` `groups` argument, with an identical behavior.


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
       return_as_scanpy=[False|True], # <-- Whether to return a dict compatible with Scanpy's `rank_genes_groups` function, or a pd.DataFrame
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

## Integrating with Scanpy
Users coming from the `scanpy` ecosystem looking for a drop-in replacement of `sc.tl.rank_genes_groups(…, method="wilcoxon")` can set `return_as_scanpy=True` when calling `illico.asymptotic_wilcoxon`. This will return a dictionary formatted for Scanpy's `rank_genes_groups` results. Example:

```python
from illico import asymptotic_wilcoxon
adata = ad.read_h5ad('dataset.h5ad') # (n_cells, n_genes)

# ... Your preprocessing steps here ...

de_genes = asymptotic_wilcoxon(
       adata,
       group_keys="perturbation",
       reference="non-targeting",
       is_log1p=[False|True], # <-- Specify if your data underwent log1p or not
       return_as_scanpy=True, # <-- /!\
       )
adata.uns["rank_genes_groups"] = de_genes # Attach results to adata.uns
# Then the rest of your scanpy workflow can remain unchanged, for example:
sc.pl.rank_genes_groups(adata, sharey=False)
```

## Subsetting the groups to test
In some scenarii, one might want to perform the tests on a subset of groups. `illico` implements this with the `groups` argument which expects a list of group labels for which p-values should be computed:
1. If a reference group is provided, only the groups provided in `groups` will be tested against the reference group. For example, if `group_keys="perturbation"`, `reference="non-targeting"` and `groups=["p1", "p2"]`, only the tests "p1 vs non-targeting" and "p2 vs non-targeting" will be performed. In this case, the speedup is linear in the number of groups listed (the less tests to perform, the faster `asymptotic_wilcoxon` will run).
2. If no reference group is provided, only the groups provided in `groups` will be tested against the rest. For example, if `group_keys="cluster"`, `reference=None` and `groups=["c1", "c2"]`, only the tests "c1 vs the rest" and "c2 vs the rest" will be performed. Note that in this case, "rest" still means the entirety of the cell population, not only the cells belonging to the groups listed in `groups`. In this case, the speedup is neglectible as the expensive part of the algorithm is to rank the entirety of the dataset, which is still required even if only a subset of the groups is tested.

```python
de_genes = asymptotic_wilcoxon(
       adata,
       group_keys="perturbation",
       ...,
       groups=["p1", "p2"], # <-- Only compute & return p-values and fold-changes for these groups.
       )
```

Note: if your data is loaded in RAM, you can achieve similar results by subsetting the `adata` before running `asymptotic_wilcoxon`. However, if your data is backed on disk through h5, subsetting the `adata` will load the entirety of the subset in RAM, defeating the initial purpose of lazy loading. In this case, using the `groups` argument is the only way to achieve a speedup by testing only a subset of the groups. If your data is backed on disk through dask, the execution speed or memory footprint resulting in subsetting the `adata` before running `asymptotic_wilcoxon` is hard to predict and might be better or worse than using the `groups` argument, depending on the size of the subset and the chunk size used by dask to load the data. In this case, we recommend trying both approaches and keeping the one that best suits your needs.

## Excluding some cells in the OVO scenario
If a reference group is provided but some cells are to be excluded from the overall analysis, one can circumvent that by creating a new group label columns in `adata.obs` where the cells to be excluded are labeled with a specific label (e.g. "excluded"), and then providing all group labels but this "excluded" label in the `groups` argument. See example below:

```python
# Filter cells with at least 10 counts
adata.obs["synthetic_group_label"] = [label for label, count in zip(adata.obs["group_label"], adata.X.sum(1)) if count >= 10 else "excluded"]
de_genes = asymptotic_wilcoxon(
       adata,
       group_keys="synthetic_group_label",
       reference="non-targeting",
       is_log1p=[False|True], # <-- Specify if your data underwent log1p or not
       groups=[label for label in adata.obs["synthetic_group_label"].unique() if label != "excluded"],
       return_as_scanpy=[False|True], # <-- Whether to return a dict compatible with Scanpy's `rank_genes_groups` function, or a pd.DataFrame
       )
```


## Excluding some cells from the "rest" in the OVR scenario
In some cases, one might want to exclude some cells from the dataset. For example, in a perturbation dataset, one might want to exclude cells that are not confidently perturbed (e.g. cells with low expression of the perturbation marker). In this case, the `exclude_from_ovr` argument expects a list of group labels that will be excluded from the "rest". Naturally, this means no p-value nor fold-change will be computed for those groups.

```python
de_genes = asymptotic_wilcoxon(
       adata,
       group_keys="cluster",
       reference=None,
       exclude_from_ovr=["c3", "c4"], # <-- Exclude these groups from the "rest" in the OVR test
       ...,
       )
```

Note: similarly as the comment above, if your data is loaded in RAM, you can achieve similar results by subsetting the `adata` before running `asymptotic_wilcoxon`. However, if your data is backed on disk through h5, subsetting the `adata` will load the entirety of the subset in RAM, defeating the initial purpose of lazy loading. In this case, using the `exclude_from_ovr` argument is the only way to avoid excessive RAM consumption. If your data is backed on disk through dask, the execution speed or memory footprint resulting in subsetting the `adata` before running `asymptotic_wilcoxon` is hard to predict and might be better or worse than using the `exclude_from_ovr` argument, depending on the size of the subset and the chunk size used by dask to load the data. In this case, we recommend trying both approaches and keeping the one that best suits your needs.
