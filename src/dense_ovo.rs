use crate::groups::GroupContainer;
use crate::groups::GroupContainerNamedTuple;
use crate::math::{chunk_and_fortranize, dense_fold_change};
use crate::ranking::{
    rank_sum_and_ties_from_binsearch, sort_along_axis_0_inplace, unique_from_sorted,
};
use crate::sparse::types::SparseFloat;
use crate::stats::compute_pvalue;
use ndarray::ArrayViewMut1;
use ndarray::ArrayViewMut2;
use ndarray::{Array1, Array2, ArrayView2, s};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::PyAny;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python, pyfunction};

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

pub fn single_group_dense_ovo_mwu_kernel<D: SparseFloat>(
    sorted_ref_data: &Array2<D>,
    sorted_tgt_data: &Array2<D>,
    n_uniques_ref: &Array1<usize>,
    ref_block_offsets: &Array2<usize>,
    tie_sums_ref: &Array1<usize>,
    use_continuity: bool,
    tie_correct: bool,
    alternative: &String,
    mut _pvalues: ArrayViewMut1<f64>,
    mut _zscores: ArrayViewMut1<f64>,
    mut _statistics: ArrayViewMut1<f64>,
) -> Result<(), String> {
    // Compute p values and u-stats
    let n_ctrl = sorted_ref_data.dim().0 as f64;
    let n_cols = sorted_ref_data.dim().1;
    let n_tgt = sorted_tgt_data.dim().0 as f64;
    let n = n_ctrl + n_tgt;
    for j in 0..n_cols {
        let n_ctrl_blocks = n_uniques_ref[j];
        // Compute ranksum and tiesum
        let (rs, ts, _) = rank_sum_and_ties_from_binsearch(
            sorted_ref_data.column(j),
            n_ctrl_blocks,
            ref_block_offsets.row(j),
            sorted_tgt_data.column(j),
            0,
        )?;

        let mu = n_ctrl * n_tgt / 2.;
        let u1 = rs - n_tgt * (n_tgt + 1.0) / 2.0;
        let (pv, zs) = compute_pvalue(
            n_ctrl,
            n_tgt,
            n,
            if tie_correct {
                (ts + tie_sums_ref[j]) as f64
            } else {
                0.0
            },
            u1,
            mu,
            if use_continuity { 0.5 } else { 0. },
            alternative,
        )?;
        _pvalues[j] = pv;
        _zscores[j] = zs;
        _statistics[j] = u1;
    }
    Ok(())
}

pub fn compute_unique_values_and_offset<D: SparseFloat>(
    mut x: ArrayViewMut2<D>,
) -> Result<(Array2<usize>, Array1<usize>, Array1<usize>), String> {
    let (nrows, ncols) = x.dim();
    let mut offsets: Array2<usize> = Array2::zeros((ncols, nrows + 1));
    let mut n_uniques: Array1<usize> = Array1::zeros(ncols);
    let mut tie_sums: Array1<usize> = Array1::zeros(ncols);

    for j in 0..ncols {
        // Store unique values in x itself (F-order)
        n_uniques[j] = unique_from_sorted(x.column_mut(j), offsets.row_mut(j).slice_mut(s![1..]))?;
        // Update tie sums before overwriting x with prefix sum, then prefix sum it
        for i in 1..(n_uniques[j] + 1) {
            let count = offsets[(j, i)];
            tie_sums[j] += count.pow(3) - count;
            offsets[(j, i)] += offsets[(j, i - 1)];
        }
    }

    Ok((offsets, n_uniques, tie_sums))
}

// Performance notes: would be faster to use vec of vecs insteads of Array for offsets, because it would help .slice()
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

    // Pre-compute unique values
    let (ctrl_offsets, ctrl_n_uniques, ctrl_tie_sums) =
        compute_unique_values_and_offset(ctrl_chunk.view_mut())?;

    // Initialize result placeholders
    let n_groups = grpc.counts.len();
    let n_cols = chunk_ub - chunk_lb;
    let mut p_values = Array2::<f64>::zeros((n_groups, n_cols));
    let mut u_stats = Array2::<f64>::zeros((n_groups, n_cols));
    let mut zscores = Array2::<f64>::zeros((n_groups, n_cols));

    for group_idx in 0..grpc.n_selected_groups {
        if group_idx == encoded_ref_group {
            p_values.row_mut(group_idx).fill(1.);
            u_stats.row_mut(group_idx).fill(-1.);
            zscores.row_mut(group_idx).fill(0.);
        } else {
            // Grab indices of the target group's cells
            let tgt_indices = grpc
                .indices
                .slice(s![grpc.indptr[group_idx]..grpc.indptr[group_idx + 1]]);
            // Chunk the target group's cells
            let mut tgt_chunk = chunk_and_fortranize(&x, chunk_lb, chunk_ub, Some(tgt_indices))?;
            // Sort them
            sort_along_axis_0_inplace(tgt_chunk.view_mut())?;

            // Now compute p-values and u-stats
            single_group_dense_ovo_mwu_kernel(
                &ctrl_chunk,
                &tgt_chunk,
                &ctrl_n_uniques,
                &ctrl_offsets,
                &ctrl_tie_sums,
                use_continuity,
                tie_correct,
                alternative,
                p_values.row_mut(group_idx),
                zscores.row_mut(group_idx),
                u_stats.row_mut(group_idx),
            )?;
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
