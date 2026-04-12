"""Utilities for formatting illico results to be compatible with Scanpy's output format."""

from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
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


def select_top_n(scores: np.ndarray, n_top: int) -> np.ndarray:
    """Select indices of top-n scores, per row.

    This function is the vectorized version of scanpy.tl._rank_genes_groups._select_top_n.
    This sorting routine does not treat equal values the same way as a simple `np.argsort`.
    See https://github.com/scverse/scanpy/pull/4038 for details.

    Arguments:
        scores (np.ndarray): 2D array of scores, shape (n_groups, n_genes)
        n_top (int): Number of top scores to select per group
    Raises:
        ValueError: If n_top is not in the valid range [1, n_genes]
    Returns:
        np.ndarray: Indices of top-n scores per group, shape (n_groups, n_top)

    """
    n_cols = scores.shape[1]
    if not (1 <= n_top <= n_cols):
        raise ValueError(f"n_top must be in [1, {n_cols}], got {n_top}")

    # Pick top-n candidates per row (unordered)
    part = np.argpartition(scores, kth=n_cols - n_top, axis=1)[:, -n_top:]

    # Sort those candidates by score descending, per row
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(part_scores, axis=1)[:, ::-1]
    top_indices = np.take_along_axis(part, order, axis=1)

    return top_indices


def format_illico_results_for_scanpy(
    adata: ad.AnnData,
    unique_groups: np.ndarray | pd.Categorical,
    reference: str | None,
    group_keys: str,
    layer: str | None,
    values: np.ndarray,
    n_genes: int | None = None,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
) -> dict:
    """Format illico results to be compatible with Scanpy's output format."""
    # Evict the reference group from the results if provided
    if reference is not None:
        mask = np.array([name != reference for name in unique_groups])
        values = values[mask, :, :]
        unique_groups = [name for name in unique_groups if name != reference]

    # Sort by signed z-score
    if n_genes is None:
        n_genes = values.shape[1]
    indices = select_top_n(values[:, :, 2], n_genes)

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
        names=np.rec.fromarrays(adata.var_names.values[indices], dtype=[(g, "O") for g in unique_groups]),
        scores=np.rec.fromarrays(
            np.take_along_axis(values[:, :, 2], indices, axis=1), dtype=[(g, "float32") for g in unique_groups]
        ),
        pvals=np.rec.fromarrays(
            np.take_along_axis(values[:, :, 0], indices, axis=1), dtype=[(g, "float64") for g in unique_groups]
        ),
        pvals_adj=np.rec.fromarrays(
            np.take_along_axis(pvals_adj, indices, axis=1), dtype=[(g, "float64") for g in unique_groups]
        ),
        logfoldchanges=np.rec.fromarrays(
            np.take_along_axis(np.log2(values[:, :, 3]), indices, axis=1), dtype=[(g, "float32") for g in unique_groups]
        ),
    )
    return output
