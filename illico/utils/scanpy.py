"""Utilities for formatting illico results to be compatible with Scanpy's output format."""

from typing import Literal

import anndata as ad
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm import trange


def adjust_pvalues(
    pvals: np.ndarray, method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg"
) -> np.ndarray:
    """Adjust p-values row-wise (pert-wise) for multiple testing."""
    assert pvals.ndim == 2
    adj_pvals = np.empty_like(pvals)
    n_tests = pvals.shape[1]
    for i in trange(pvals.shape[0], desc="Adjusting p-values…", leave=False):
        if method == "benjamini-hochberg":
            _, adj_pvals[i], _, _ = multipletests(pvals[i], method="fdr_bh", alpha=0.05)
        elif method == "bonferroni":
            adjusted = pvals[i] * n_tests
            adjusted = np.minimum(adjusted, 1.0)
            adj_pvals[i] = adjusted
        else:
            raise ValueError(f"Unknown adjustment method: {method}")

    return adj_pvals


def format_illico_results_for_scanpy(
    adata: ad.AnnData,
    reference: str | None,
    group_keys: str,
    layer: str | None,
    values: np.ndarray,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
) -> dict:
    """Format illico results to be compatible with Scanpy's output format."""
    # Evict the reference group from the results if provided
    sorted_pert_names = sorted(adata.obs[group_keys].unique())
    if reference is not None:
        mask = np.array([name != reference for name in sorted_pert_names])
        values = values[mask, :, :]
        sorted_pert_names = [name for name in sorted_pert_names if name != reference]

    # Sort by signed z-score
    indices = np.argsort(values[:, :, 2], axis=1)[:, ::-1]
    # Adjust p-values
    pvals_adj = adjust_pvalues(values[:, :, 0], method=corr_method)

    # Format output
    output = dict(
        params=dict(
            groupby=group_keys,
            reference=reference,
            method="wilcoxon",
            use_raw=False,
            layer=layer,
            corr_method=corr_method,
        ),
        names=np.rec.fromarrays(adata.var_names.values[indices], names=sorted_pert_names),
        scores=np.rec.fromarrays(np.take_along_axis(values[:, :, 2], indices, axis=1), names=sorted_pert_names),
        pvals=np.rec.fromarrays(np.take_along_axis(values[:, :, 0], indices, axis=1), names=sorted_pert_names),
        pvals_adj=np.rec.fromarrays(np.take_along_axis(pvals_adj, indices, axis=1), names=sorted_pert_names),
        logfoldchanges=np.rec.fromarrays(
            np.take_along_axis(np.log2(values[:, :, 3]), indices, axis=1), names=sorted_pert_names
        ),
    )
    return output
