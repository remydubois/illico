use crate::groups::GroupContainer;
use crate::groups::GroupContainerNamedTuple;
use crate::math::{chunk_and_fortranize, dense_fold_change};
use crate::ranking::{rank_sum_and_ties, sort_along_axis_0_inplace};
use crate::sparse::types::SparseFloat;
use crate::stats::compute_pvalue;
use ndarray::ArrayViewMut1;
use ndarray::{Array1, Array2, ArrayView2, s};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::PyAny;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python, pyfunction};

pub fn dense_ovo_kernel<D: SparseFloat>(
    sorted_controls: ArrayView2<D>,
    sorted_tgt: ArrayView2<D>,
    use_continuity: bool,
    tie_correct: bool,
    alternative: &String,
    mut p_values: ArrayViewMut1<f64>,
    mut u_stats: ArrayViewMut1<f64>,
    mut zscores: ArrayViewMut1<f64>,
) -> Result<(), String> {
    let n_ctrl = sorted_controls.dim().0 as f64;
    let n_cols = sorted_controls.dim().1 as f64;
    let n_tgt = sorted_tgt.dim().0 as f64;

    let n = n_ctrl + n_tgt;
    let mu = n_ctrl * n_tgt / 2.;

    let remainder = &n_tgt * (&n_tgt + 1.) / 2.;
    for j in 0..n_cols as usize {
        let (rs, ts, _) = rank_sum_and_ties(sorted_controls.column(j), sorted_tgt.column(j), 0);

        let u = rs - remainder;

        let contin_corr = if use_continuity { 0.5 } else { 0. };
        let (pv, zscore) = compute_pvalue(
            n_ctrl,
            n_tgt,
            n,
            if tie_correct { ts } else { 0. },
            u,
            mu,
            contin_corr,
            alternative,
        )?;
        p_values[j] = pv;
        u_stats[j] = u;
        zscores[j] = zscore;
    }

    return Ok(());
}

#[pyfunction]
pub fn dense_ovo_kernel_rust<'py>(
    py: Python<'py>,
    sorted_controls: PyReadonlyArray2<'py, f32>,
    sorted_tgt: PyReadonlyArray2<'py, f32>,
    use_continuity: bool,
    tie_correct: bool,
    alternative: String,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let sorted_controls = sorted_controls.as_array();
    let mut p_values = Array1::zeros(sorted_controls.dim().1);
    let mut u_stats = Array1::zeros(sorted_controls.dim().1);
    let mut zscores = Array1::zeros(sorted_controls.dim().1);
    _ = dense_ovo_kernel(
        sorted_controls,
        sorted_tgt.as_array(),
        use_continuity,
        tie_correct,
        &alternative,
        p_values.view_mut(),
        u_stats.view_mut(),
        zscores.view_mut(),
    )
    .map_err(PyValueError::new_err)?;
    return Ok((
        PyArray1::from_array(py, &p_values),
        PyArray1::from_array(py, &u_stats),
    ));
}

pub fn dense_ovo_over_contiguous_col_chunk<D: SparseFloat>(
    x: ArrayView2<D>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: &String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
    if chunk_lb >= chunk_ub {
        return Err(format!(
            "Chunking error: lower bound ({}) is not smaller than upper bound ({}).",
            { chunk_lb },
            { chunk_ub }
        ));
    }

    if grpc.encoded_ref_group < 0 {
        return Err(format!(
            "Encoded ref group can not be negative. Received {}.",
            grpc.encoded_ref_group
        ));
    }
    let encoded_ref_group = grpc.encoded_ref_group as usize;

    // Chunk control cells out and sort in-place
    let ctrl_indices = grpc.indices.slice(s![
        grpc.indptr[encoded_ref_group]..grpc.indptr[encoded_ref_group + 1]
    ]);
    let mut ctrl_chunk = chunk_and_fortranize(&x, chunk_lb, chunk_ub, Some(ctrl_indices))?;
    sort_along_axis_0_inplace(ctrl_chunk.view_mut())?;

    // Initialize result placeholders
    let n_groups = grpc.counts.len();
    let n_cols = chunk_ub - chunk_lb;
    let mut p_values = Array2::<f64>::zeros((n_groups, n_cols));
    let mut u_stats = Array2::<f64>::zeros((n_groups, n_cols));
    let mut zscores = Array2::<f64>::zeros((n_groups, n_cols));

    for i in 0..n_groups {
        if i as isize == grpc.encoded_ref_group {
            p_values.row_mut(i).fill(1.);
            u_stats.row_mut(i).fill(-1.);
            zscores.row_mut(i).fill(0.);
        } else {
            // Grab indices of the target group's cells
            let tgt_indices = grpc.indices.slice(s![grpc.indptr[i]..grpc.indptr[i + 1]]);
            // Chunk the target group's cells
            let mut tgt_chunk = chunk_and_fortranize(&x, chunk_lb, chunk_ub, Some(tgt_indices))?;
            // Sort them
            sort_along_axis_0_inplace(tgt_chunk.view_mut())?;

            // Compute p values and u-stats
            _ = dense_ovo_kernel(
                ctrl_chunk.view(),
                tgt_chunk.view(),
                use_continuity,
                tie_correct,
                alternative,
                p_values.row_mut(i),
                u_stats.row_mut(i),
                zscores.row_mut(i),
            )?;

            // Fill in placeholders
            // p_values.row_mut(i).assign(&p);
            // u_stats.row_mut(i).assign(&u);
        }
    }
    let fc = dense_fold_change(
        x.slice(s![.., chunk_lb..chunk_ub]),
        &grpc,
        is_log1p,
        exp_post_agg,
    )?;
    return Ok((p_values, u_stats, zscores, fc));
}

macro_rules! run_ovo_branch {
    ($py:expr, $x:expr, $chunk_lb:expr, $chunk_ub:expr, $grpc:expr, $is_log1p:expr, $use_continuity:expr, $tie_correct:expr, $exp_post_agg:expr, $alternative:expr, $dt:ty) => {{
        let x_pyarray = $x.extract::<PyReadonlyArray2<'py, $dt>>()?;
        let x = x_pyarray.as_array();
        $py.detach(|| {
            dense_ovo_over_contiguous_col_chunk(
                x.view(),
                $chunk_lb,
                $chunk_ub,
                $grpc,
                $is_log1p,
                $use_continuity,
                $tie_correct,
                $exp_post_agg,
                &$alternative,
            )
        })
        .map_err(PyValueError::new_err)
    }};
}

type PyArr2<'py> = Bound<'py, PyArray2<f64>>;

#[rustfmt::skip]
#[pyfunction]
pub fn dense_ovo_over_contiguous_col_chunk_rust<'py>(
    py: Python<'py>,
    // x: PyReadonlyArray2<'py, f32>,
    x: Bound<'py, PyAny>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainerNamedTuple<'py>,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> PyResult<(
    PyArr2<'py>,
    PyArr2<'py>,
    PyArr2<'py>,
    PyArr2<'py>,
)> {
    let grpc = grpc.as_group_container();
    // let x = x.as_array();
    let data_dtype: String = x.getattr("dtype")?.getattr("str")?.extract()?;
    let (p_values, u_stats, zscores, fc) = match data_dtype.as_str() {
        "f32" | "<f4" => run_ovo_branch!(
            py, x, chunk_lb, chunk_ub, grpc, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f32
        ),
        "f64" | "<f8" => run_ovo_branch!(
            py, x, chunk_lb, chunk_ub, grpc, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f64
        ),
        _ => Err(PyValueError::new_err(format!(
            "Input data should be f32 or f64, received {}",
            data_dtype
        ))),
    }?;
    return Ok((
        PyArray2::from_array(py, &p_values),
        PyArray2::from_array(py, &u_stats),
        PyArray2::from_array(py, &zscores),
        PyArray2::from_array(py, &fc),
    ));
}
