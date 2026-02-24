# Results and expectations

## `illico` is not faster than `scanpy` or `pdex`, is there a bug?

`illico` relies on a few optimization tricks to be faster than other existing tools. It is very possible that for some reason, the specific layout of your dataset (very small control population, very low sparsity, very small amount of distinct values) result in those tricks being effect-less, or less effective than observed on the datasets used to develop & benchmark `illico`. It is also very possible that because of those, other solutions end up faster than `illico` ! If this is your case, please open a issue describing your situation.

## `illico`'s results (p-values or fold-change) do not match `pdex` or `scanpy`.

### Test results (p-values)

Please open an issue, but before that: make sure that you are running **asymptotic** wilcoxon rank-sum tests as this is the only test exposed by `illico`.

- `pdex` relies on `scipy.stats.mannwhitneyu` that runs exact (non asymptotic) when there are 8 values (or less) in both groups combined, and no ties.
- `scanpy` offers the possibility to run non-tie-corrected wilcoxon rank-sum tests (default behavior). If you come from scanpy, make sure arguments match.
- Also, `illico` uses continuity correction by default which is the best practice (can be disabled).

The test suite implemented in the CI and used to develop `illico` targets a precision of 1.e-12 compared to `scipy`, not `scanpy`. Consequently, there **will be** slight disagreement between `scanpy`'s p-values and `illico`'s p-values, as `scanpy` itself disagrees with `scipy`.

### Fold-change

The fold-change computed by illico is the most naive form of the fold-change:

$$\text{fold-change} = \frac{E[X_{\text{perturbed}}]}{E[X_{\text{control}}]}$$

If your data underwent log1p transform, `np.expm1` is applied **before** computing the expectations (means), in which case the fold change expression becomes:

$$\text{fold-change} = \frac{E[e^{X_{\text{perturbed}}} - 1]}{E[e^{X_{\text{control}}} - 1]}$$

I know several definitions exist, and adding more control over this should not be complicated. If this is your case, please open an issue.
