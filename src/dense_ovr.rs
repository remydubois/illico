use crate::groups::GroupContainer;
use crate::groups::GroupContainerNamedTuple;
use crate::math::{chunk_and_fortranize, dense_fold_change};
use crate::ranking::{accumulate_rank_and_tie_sums_from_argsort, argsort};
use crate::sparse::types::SparseFloat;
use crate::stats::compute_pvalue;
use ndarray::{Array1, Array2, ArrayView2, s};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python, pyfunction};

pub fn dense_ovr_kernel<D: SparseFloat>(
    x: ArrayView2<D>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f32>), String> {
    let chunk = chunk_and_fortranize(&x, chunk_lb, chunk_ub, None)?;
    // Now compute stats and pvalues
    let n_groups = grpc.counts.len();
    let mut p_values = Array2::zeros((n_groups, chunk_ub - chunk_lb));
    let mut u_stats = Array2::zeros((n_groups, chunk_ub - chunk_lb));
    let mut zscores = Array2::zeros((n_groups, chunk_ub - chunk_lb));

    // Compute ranksum and tie sum
    let mut ranksums: Array2<f64> = Array2::zeros((n_groups, chunk_ub - chunk_lb));
    let mut tie_sums = Array1::zeros(chunk_ub - chunk_lb);
    for j in 0..chunk.dim().1 {
        let sorted_indices = argsort(chunk.column(j));
        accumulate_rank_and_tie_sums_from_argsort(
            chunk.column(j),
            sorted_indices,
            grpc.encoded_groups,
            ranksums.column_mut(j),
            tie_sums.slice_mut(s![j]),
        )?;
    }

    let n = chunk.dim().0 as f64;
    let n_ref = grpc.counts.mapv(|v| n - v as f64);
    let n_tgt = grpc.counts.mapv(|x| x as f64);
    let mu = &n_ref * &n_tgt / 2.;
    let remainder = &n_tgt * (&n_tgt.map(|x| x + 1.)) / 2.;
    for i in 0..n_groups {
        for j in 0..chunk.dim().1 {
            u_stats[[i, j]] = ranksums[[i, j]] - remainder[i];
            let (pv, z) = compute_pvalue(
                n_ref[i],
                n_tgt[i],
                n,
                if tie_correct { tie_sums[j] } else { 0. },
                u_stats[[i, j]],
                mu[i],
                if use_continuity { 0.5 } else { 0. },
                &alternative,
            )?;
            p_values[[i, j]] = pv;
            zscores[[i, j]] = z
        }
    }

    // TODO: dense_fold_change could actually take an normal array, not a view, as we build and own it with chunk_and_fortranize
    let fold_change = dense_fold_change(chunk.view(), &grpc, is_log1p, exp_post_agg)?;

    Ok((p_values, u_stats, zscores, fold_change))
}

macro_rules! run_ovr_branch {
    ($py:expr, $x:expr, $chunk_lb:expr, $chunk_ub:expr, $grpc:expr, $is_log1p:expr, $use_continuity:expr, $tie_correct:expr, $exp_post_agg:expr, $alternative:expr, $dt:ty) => {{
        let x_pyarray = $x.extract::<PyReadonlyArray2<'py, $dt>>()?;
        let x = x_pyarray.as_array();
        $py.detach(|| {
            dense_ovr_kernel(
                x,
                $chunk_lb,
                $chunk_ub,
                $grpc,
                $is_log1p,
                $use_continuity,
                $tie_correct,
                $exp_post_agg,
                $alternative,
            )
        })
        .map_err(PyValueError::new_err)
    }};
}

type PyArr2<'py> = Bound<'py, PyArray2<f64>>;

#[rustfmt::skip]
#[pyfunction]
pub fn dense_ovr_over_contiguous_col_chunk_rust<'py>(
    py: Python<'py>,
    // x: PyReadonlyArray2<'py, f32>,
    x: Bound<'py, PyAny>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainerNamedTuple,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> PyResult<(
    PyArr2<'py>,
    PyArr2<'py>,
    PyArr2<'py>,
    Bound<'py, PyArray2<f32>>,
)> {
    let grpc = grpc.as_group_container();
    let data_dtype: String = x.getattr("dtype")?.getattr("str")?.extract()?;

    let (p, u, z, fc) = match data_dtype.as_str() {
        "f32" | "<f4" => run_ovr_branch!(
            py, x, chunk_lb, chunk_ub, grpc, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f32
        ),
        "f64" | "<f8" => run_ovr_branch!(
            py, x, chunk_lb, chunk_ub, grpc, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f64
        ),
        _ => Err(PyValueError::new_err(format!(
            "Input data should be f32 or f64, received {}",
            data_dtype
        ))),
    }?;
    Ok((
        PyArray2::from_array(py, &p),
        PyArray2::from_array(py, &u),
        PyArray2::from_array(py, &z),
        PyArray2::from_array(py, &fc),
    ))
}
