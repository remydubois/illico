use crate::csr::csr_fold_change;
use crate::groups::{GroupContainer, GroupContainerNamedTuple};
use crate::ranking::rank_sum_and_ties;
use crate::sparse::types::{
    CSCMatrix, CSRMatrix, OwnedCSCMatrix, OwnedCSRMatrix, PyCSCMatrix, PyCSRMatrix, SparseFloat,
    SparseIndex,
};
use crate::stats::compute_pvalue;
use ndarray::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn single_group_sparse_ovo_mwu_kernel<D: SparseFloat, I: SparseIndex>(
    ctrl: &OwnedCSCMatrix<D, I>,
    tgt: OwnedCSCMatrix<D, I>,
    use_continuity: bool,
    tie_correct: bool,
    alternative: &String,
    mut p_values: ArrayViewMut1<f64>,
    mut u_stats: ArrayViewMut1<f64>,
    mut zscores: ArrayViewMut1<f64>,
) -> Result<(), String> {
    let n_cols_ctrl = ctrl.shape.1;
    let n_ctrl = ctrl.shape.0 as f64;
    let n_cols_tgt = tgt.shape.1;
    let n_tgt = tgt.shape.0 as f64;
    if n_cols_tgt != n_cols_ctrl {
        return Err(format!(
            "Uneven number of columns in controls ({}) and targets ({}).",
            n_cols_ctrl as usize, n_cols_tgt as usize
        ));
    }
    let n_total = n_ctrl + n_tgt;

    // Count number of zeros in controls
    let mut n_zeros_ctrl = Array1::zeros(n_cols_ctrl);
    for j in 0..n_cols_ctrl {
        n_zeros_ctrl[j] = n_ctrl - (ctrl.indptr[j + 1] - ctrl.indptr[j]).to_usize() as f64; // TODO; fix more elegantly
    }
    // Count number of zeros in targets
    let mut n_zeros_tgt = Array1::zeros(n_cols_tgt);
    for j in 0..n_cols_tgt {
        n_zeros_tgt[j] = n_tgt - (tgt.indptr[j + 1] - tgt.indptr[j]).to_usize() as f64;
    }

    let mu = n_ctrl * n_tgt / 2.;
    let remainder = n_tgt * (n_tgt + 1.) / 2.;
    for j in 0..n_cols_ctrl {
        let n_zeros_total = n_zeros_ctrl[j] + n_zeros_tgt[j];
        let (lbc, ubc) = (ctrl.indptr[j].to_usize(), ctrl.indptr[j + 1].to_usize());
        let (lbt, ubt) = (tgt.indptr[j].to_usize(), tgt.indptr[j + 1].to_usize());

        // Compute ranksum and tiesum on non zeros only first
        let (mut ranksum, mut tiesum) =
            rank_sum_and_ties(ctrl.data.slice(s![lbc..ubc]), tgt.data.slice(s![lbt..ubt]));

        // Offset ranksum of nonzeros elements: all the ranks must be increase by n_zeros_total, and all the target elements (ubt - lbt) contribute to the ranksum
        ranksum += n_zeros_total * (ubt - lbt) as f64;
        // Now compute the ranksum of zero elements, those contribute as well
        let rank_of_zeros = (n_zeros_ctrl[j] + n_zeros_tgt[j] + 1.) * 0.5;
        ranksum += rank_of_zeros * n_zeros_tgt[j];

        // Now, icnrement the tiesum with the zeros
        tiesum += n_zeros_total.powi(3) - n_zeros_total;

        // Compute U-stats
        let u = ranksum - remainder;

        let (pv, z) = compute_pvalue(
            n_ctrl,
            n_tgt,
            n_total,
            if tie_correct { tiesum } else { 0. },
            u,
            mu,
            if use_continuity { 0.5 } else { 0. },
            alternative,
        )?;
        p_values[j] = pv;
        u_stats[j] = u;
        zscores[j] = z;
    }

    Ok(())
}

pub fn multigroup_sparse_ovo_mwu_kernel<D: SparseFloat, I: SparseIndex>(
    x: &OwnedCSRMatrix<D, I>,
    grpc: &GroupContainer,
    use_continuity: bool,
    tie_correct: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
    if grpc.encoded_ref_group < 0 {
        return Err(format!(
            "Encoded ref group can not be negative. Received {}.",
            grpc.encoded_ref_group
        ));
    }
    let encoded_ref_group = grpc.encoded_ref_group as usize;
    // chunk control cells
    let start = grpc.indptr[encoded_ref_group];
    let end = grpc.indptr[encoded_ref_group + 1];
    let control_indices = grpc.indices.slice(s![start..end]);
    let mut control_chunk = x.index_rows_into_csc(control_indices)?;
    control_chunk.sort_columns_inplace()?;

    let n_groups = grpc.counts.len();
    let mut pvalues = Array2::zeros((n_groups, x.shape.1));
    let mut u_stats = Array2::zeros((n_groups, x.shape.1));
    let mut zscores = Array2::zeros((n_groups, x.shape.1));
    for group_idx in 0..n_groups {
        if group_idx == encoded_ref_group {
            pvalues.row_mut(group_idx).fill(1.);
            u_stats.row_mut(group_idx).fill(-1.);
            zscores.row_mut(group_idx).fill(0.);
        } else {
            // Chunk the target
            let start = grpc.indptr[group_idx as usize];
            let end = grpc.indptr[group_idx as usize + 1];
            let tgt_indices = grpc.indices.slice(s![start..end]);
            let mut tgt_chunk = x.index_rows_into_csc(tgt_indices)?;
            tgt_chunk.sort_columns_inplace()?;

            // Now compute p-values and u-stats
            _ = single_group_sparse_ovo_mwu_kernel(
                &control_chunk,
                tgt_chunk,
                use_continuity,
                tie_correct,
                &alternative,
                pvalues.row_mut(group_idx),
                u_stats.row_mut(group_idx),
                zscores.row_mut(group_idx),
            )?;
        }
    }

    Ok((pvalues, u_stats, zscores))
}

pub fn csc_ovo_mwu_kernel_over_contiguous_col_chunk<'py, D: SparseFloat, I: SparseIndex>(
    x: &'py CSCMatrix<'py, D, I>,
    grpc: GroupContainer,
    chunk_lb: usize,
    chunk_ub: usize,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f32>), String> {
    let chunk = x.contig_cols_into_csr(chunk_lb, chunk_ub)?;

    let (pvalues, u_stats, zscores) =
        multigroup_sparse_ovo_mwu_kernel(&chunk, &grpc, use_continuity, tie_correct, alternative)?;

    let fc = csr_fold_change(&chunk, &grpc, is_log1p, exp_post_agg)?;

    Ok((pvalues, u_stats, zscores, fc))
}

pub fn csr_ovo_mwu_kernel_over_contiguous_col_chunk<'py, D: SparseFloat, I: SparseIndex>(
    x: &'py CSRMatrix<'py, D, I>,
    grpc: GroupContainer,
    chunk_lb: usize,
    chunk_ub: usize,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f32>), String> {
    let chunk = x.contig_cols_into_csr(chunk_lb, chunk_ub)?;

    let (pvalues, u_stats, zscores) =
        multigroup_sparse_ovo_mwu_kernel(&chunk, &grpc, use_continuity, tie_correct, alternative)?;

    let fc = csr_fold_change(&chunk, &grpc, is_log1p, exp_post_agg)?;

    Ok((pvalues, u_stats, zscores, fc))
}

type PyArr2f32<'py> = Bound<'py, PyArray2<f32>>;
type PyArr2f64<'py> = Bound<'py, PyArray2<f64>>;

// The extraction into PyArray + conversion to Array + compute has to be done in one single function, because dtypes are not known at compile time and pyfunctions dont accept generic traits.
// Hence, it is not possible to have let's say a function returning a dtyped object: even PyAny.extract -> PyArray because PyArray has to be typed.
// Previous implementation was 1/ FromPyObject's .extract returning a PyArray, 2/ then .as_csr returning an Array. None of those can be compiled in the dtype-agnostic setup.
// Hence, conversion into pyarray, then conversion into arrays must happen in the same scope when dtype is known.
#[rustfmt::skip]
macro_rules! run_branch {
    ($format:expr, $x:expr, $py:expr, $grpc:expr, $chunk_lb:expr, $chunk_ub:expr, $is_log1p:expr, $use_continuity:expr, $tie_correct:expr, $exp_post_agg:expr, $alternative:expr, $dt:ty, $it:ty) => {{
        let data = $x.data.extract::<PyReadonlyArray1<'py, $dt>>()?;
        let indices = $x.indices.extract::<PyReadonlyArray1<'py, $it>>()?;
        let indptr = $x.indptr.extract::<PyReadonlyArray1<'py, $it>>()?;

        let format = $format;

        match format {
            "CSR" => {
                let csr = CSRMatrix {
                    data: data.as_array(),
                    indices: indices.as_array(),
                    indptr: indptr.as_array(),
                    shape: $x.shape,
                };

                $py.detach(|| {
                    csr_ovo_mwu_kernel_over_contiguous_col_chunk(
                        &csr, $grpc, $chunk_lb, $chunk_ub, $is_log1p, $use_continuity, $tie_correct, $exp_post_agg, $alternative,
                    )
                })
                .map_err(PyValueError::new_err)
            }
            "CSC" => {
                let csc = CSCMatrix {
                    data: data.as_array(),
                    indices: indices.as_array(),
                    indptr: indptr.as_array(),
                    shape: $x.shape,
                };

                $py.detach(|| {
                    csc_ovo_mwu_kernel_over_contiguous_col_chunk(
                        &csc, $grpc, $chunk_lb, $chunk_ub, $is_log1p, $use_continuity, $tie_correct, $exp_post_agg, $alternative,
                    )
                })
                .map_err(PyValueError::new_err)
            }
            _ => panic!("Unkown format"),
        }
    }};
}

#[rustfmt::skip]
#[pyfunction]
pub fn csr_ovo_mwu_kernel_over_contiguous_col_chunk_rust<'py>(
    py: Python<'py>,
    x: PyCSRMatrix<'py>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainerNamedTuple,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> PyResult<(
    PyArr2f64<'py>,
    PyArr2f64<'py>,
    PyArr2f64<'py>,
    PyArr2f32<'py>,
)> {
    let grpc = grpc.as_group_container();

    let data_dtype: String = x.data.getattr("dtype")?.getattr("str")?.extract()?;
    let idx_dtype: String = x.indices.getattr("dtype")?.getattr("str")?.extract()?;
    let (pv, u, z, fc) = match (data_dtype.as_str(), idx_dtype.as_str()) {
        ("f32" | "<f4", "i32" | "<i4") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f32, i32
        ),
        ("f64" | "<f8", "i32" | "<i4") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f64, i32
        ),
        ("f32" | "<f4", "i64" | "<i8") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f32, i64
        ),
        ("f64" | "<f8", "i64" | "<i8") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f64, i64
        ),
        _ => Err(PyValueError::new_err(format!(
            "Error casting data (only f32 and f64 supported, received {}) and indices (only int32 and int64 supported, received {}).",
            data_dtype, idx_dtype
        ))),
    }?;

    return Ok((
        PyArray2::from_array(py, &pv),
        PyArray2::from_array(py, &u),
        PyArray2::from_array(py, &z),
        PyArray2::from_array(py, &fc),
    ));
}

#[rustfmt::skip]
#[pyfunction]
pub fn csc_ovo_mwu_kernel_over_contiguous_col_chunk_rust<'py>(
    py: Python<'py>,
    x: PyCSCMatrix<'py>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainerNamedTuple,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> PyResult<(
    PyArr2f64<'py>,
    PyArr2f64<'py>,
    PyArr2f64<'py>,
    PyArr2f32<'py>,
)> {
    let grpc = grpc.as_group_container();

    let data_dtype: String = x.data.getattr("dtype")?.getattr("str")?.extract()?;
    let idx_dtype: String = x.indices.getattr("dtype")?.getattr("str")?.extract()?;
    let (pv, u, z, fc) = match (data_dtype.as_str(), idx_dtype.as_str()) {
        ("f32" | "<f4", "i32" | "<i4") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f32, i32
        ),
        ("f64" | "<f8", "i32" | "<i4") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f64, i32
        ),
        ("f32" | "<f4", "i64" | "<i8") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f32, i64
        ),
        ("f64" | "<f8", "i64" | "<i8") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, alternative, f64, i64
        ),
        _ => Err(PyValueError::new_err(format!(
            "Error casting data (only f32 and f64 supported, received {}) and indices (only int32 and int64 supported, received {}).",
            data_dtype, idx_dtype
        ))),
    }?;

    return Ok((
        PyArray2::from_array(py, &pv),
        PyArray2::from_array(py, &u),
        PyArray2::from_array(py, &z),
        PyArray2::from_array(py, &fc),
    ));
}
