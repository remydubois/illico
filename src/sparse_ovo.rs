use crate::groups::{GroupContainer, GroupContainerNamedTuple};
use crate::math::fold_change_from_summed_expr;
use crate::ranking::{rank_sum_and_ties_from_binsearch, unique_from_sorted};
use crate::sparse::types::{
    CSCMatrix, CSRMatrix, OwnedCSCMatrix, PyCSCMatrix, PyCSRMatrix, SparseFloat, SparseIndex,
};
use crate::stats::compute_pvalue;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, s};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// pub fn single_group_sparse_ovo_mwu_kernel<D: SparseFloat, I: SparseIndex>(
//     ctrl: &OwnedCSCMatrix<D, I>,
//     tgt: OwnedCSCMatrix<D, I>,
//     use_continuity: bool,
//     tie_correct: bool,
//     alternative: &String,
//     mut p_values: ArrayViewMut1<f64>,
//     mut u_stats: ArrayViewMut1<f64>,
//     mut zscores: ArrayViewMut1<f64>,
// ) -> Result<(), String> {
//     let n_cols_ctrl = ctrl.shape.1;
//     let n_ctrl = ctrl.shape.0 as f64;
//     let n_cols_tgt = tgt.shape.1;
//     let n_tgt = tgt.shape.0 as f64;
//     if n_cols_tgt != n_cols_ctrl {
//         return Err(format!(
//             "Uneven number of columns in controls ({}) and targets ({}).",
//             n_cols_ctrl as usize, n_cols_tgt as usize
//         ));
//     }
//     let n_total = n_ctrl + n_tgt;

//     // Count number of zeros in controls
//     let mut n_zeros_ctrl = Array1::zeros(n_cols_ctrl);
//     for j in 0..n_cols_ctrl {
//         n_zeros_ctrl[j] = n_ctrl - (ctrl.indptr[j + 1] - ctrl.indptr[j]).to_usize() as f64; // TODO; fix more elegantly
//     }
//     // Count number of zeros in targets
//     let mut n_zeros_tgt = Array1::zeros(n_cols_tgt);
//     for j in 0..n_cols_tgt {
//         n_zeros_tgt[j] = n_tgt - (tgt.indptr[j + 1] - tgt.indptr[j]).to_usize() as f64;
//     }

//     let mu = n_ctrl * n_tgt / 2.;
//     let remainder = n_tgt * (n_tgt + 1.) / 2.;
//     for j in 0..n_cols_ctrl {
//         let n_zeros_total = n_zeros_ctrl[j] + n_zeros_tgt[j];
//         let (lbc, ubc) = (ctrl.indptr[j].to_usize(), ctrl.indptr[j + 1].to_usize());
//         let (lbt, ubt) = (tgt.indptr[j].to_usize(), tgt.indptr[j + 1].to_usize());

//         // Compute ranksum and tiesum on non zeros only first
//         let (mut ranksum, mut tiesum, zero_pos) = rank_sum_and_ties(
//             ctrl.data.slice(s![lbc..ubc]),
//             tgt.data.slice(s![lbt..ubt]),
//             n_zeros_total as usize,
//         );

//         // Now compute the ranksum of zero elements, those contribute as well
//         let rank_of_zeros = (n_zeros_ctrl[j] + n_zeros_tgt[j] + 1.) * 0.5 + zero_pos as f64;
//         ranksum += rank_of_zeros * n_zeros_tgt[j];

//         // Now, icnrement the tiesum with the zeros
//         tiesum += n_zeros_total.powi(3) - n_zeros_total;

//         // Compute U-stats
//         let u = ranksum - remainder;

//         let (pv, z) = compute_pvalue(
//             n_ctrl,
//             n_tgt,
//             n_total,
//             if tie_correct { tiesum } else { 0. },
//             u,
//             mu,
//             if use_continuity { 0.5 } else { 0. },
//             alternative,
//         )?;
//         p_values[j] = pv;
//         u_stats[j] = u;
//         zscores[j] = z;
//     }

//     Ok(())
// }

pub fn compute_sparse_unique_values_and_offsets<D: SparseFloat, I: SparseIndex>(
    csc_mat: &mut OwnedCSCMatrix<D, I>,
) -> Result<(Array1<usize>, Array1<usize>, Array1<usize>), String> {
    let ncols = csc_mat.shape.1;
    let mut offsets: Array1<usize> = Array1::zeros(csc_mat.data.dim() + ncols);
    let mut tie_sums: Array1<usize> = Array1::zeros(ncols);
    let mut n_uniques: Array1<usize> = Array1::zeros(ncols);
    for j in 0..ncols {
        let org_lb = csc_mat.indptr[j].to_usize();
        let org_ub = csc_mat.indptr[j + 1].to_usize();
        let offsets_lb = org_lb + j + 1;
        let offsets_ub = org_ub + j + 1;

        n_uniques[j] = unique_from_sorted(
            csc_mat.data.slice_mut(s![org_lb..org_ub]),
            offsets.slice_mut(s![offsets_lb..offsets_ub]),
        )?;
        let nz_offset_ub = offsets_lb + n_uniques[j];
        // this_col_counts =
        for k in offsets_lb..nz_offset_ub {
            // First increment tie sums
            if offsets[k] > 1 {
                tie_sums[j] += offsets[k].pow(3) - offsets[k];
            };
            // Then replace with prefix sum
            offsets[k] += offsets[k - 1];
        }
    }
    Ok((offsets, n_uniques, tie_sums))
}

pub fn single_group_sparse_ovo_mwu_kernel<D: SparseFloat, I: SparseIndex>(
    sorted_ref_data: &OwnedCSCMatrix<D, I>,
    sorted_tgt_data: &OwnedCSCMatrix<D, I>,
    n_uniques_ref: &Array1<usize>,
    ref_block_offsets: &Array1<usize>,
    tie_sums_ref: &Array1<usize>,
    use_continuity: bool,
    tie_correct: bool,
    alternative: &String,
    mut _pvalues: ArrayViewMut1<f64>,
    mut _zscores: ArrayViewMut1<f64>,
    mut _statistics: ArrayViewMut1<f64>,
) -> Result<(), String> {
    let (n_ref, n_cols) = sorted_ref_data.shape;
    let n_tgt = sorted_tgt_data.shape.0;
    let n = n_ref + n_tgt;
    let mu = n_ref as f64 * n_tgt as f64 / 2.0;

    for j in 0..n_cols {
        // Compute bounds
        let lbr = sorted_ref_data.indptr[j].to_usize();
        let ubr = lbr + n_uniques_ref[j];
        let lbt = sorted_tgt_data.indptr[j].to_usize();
        let ubt = sorted_tgt_data.indptr[j + 1].to_usize();
        let lboffsets = lbr + j;
        let uboffsets = lbr + j + 1 + n_uniques_ref[j];

        let n_zeros_ref = n_ref - (sorted_ref_data.indptr[j + 1].to_usize() - lbr);
        let n_zeros_tgt = n_tgt - (ubt - lbt);
        let n_zeros_combined = n_zeros_ref + n_zeros_tgt;

        let (nz_rs, nz_ts, zpos) = rank_sum_and_ties_from_binsearch(
            sorted_ref_data.data.slice(s![lbr..ubr]),
            n_uniques_ref[j],
            ref_block_offsets.slice(s![lboffsets..uboffsets]),
            sorted_tgt_data.data.slice(s![lbt..ubt]),
            n_zeros_combined,
        )?;

        let tie_sum = nz_ts + (n_zeros_combined.pow(3) - n_zeros_combined) + tie_sums_ref[j];
        // This is the ranksum of zero elements
        let mut ranksum = (zpos as f64 + (n_zeros_ref as f64 + n_zeros_tgt as f64 + 1.0) / 2.0)
            * (n_zeros_tgt as f64);
        // Add the ranksum of nonzero elements;
        ranksum += nz_rs;
        let u1 = ranksum - (n_tgt as f64) * (n_tgt as f64 + 1.0) / 2.0;

        (_pvalues[j], _zscores[j]) = compute_pvalue(
            n_ref as f64,
            n_tgt as f64,
            n as f64,
            if tie_correct { tie_sum as f64 } else { 0.0 },
            u1,
            mu,
            if use_continuity { 0.5 } else { 0.0 },
            alternative,
        )?;
        _statistics[j] = u1;
    }
    Ok(())
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
    alternative: &String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
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
    let mut control_chunk =
        x.index_rows_contig_cols_into_csc(chunk_lb, chunk_ub, control_indices)?;
    control_chunk.sort_columns_inplace()?;

    // Initialize aggregated counts
    let n_groups = grpc.counts.len();
    let mut agg_count = Array2::<D>::zeros((n_groups, control_chunk.shape.1));
    // Populate it for the reference group because we will overwrite its content
    agg_count
        .row_mut(encoded_ref_group)
        .assign(&control_chunk.sum_axis0(is_log1p & !exp_post_agg)?);

    // Pre-compute unique values, counts and offsets in the control
    let (ctrl_offsets, ctrl_n_uniques, ctrl_tie_sums) =
        compute_sparse_unique_values_and_offsets(&mut control_chunk)?;

    // Allocate result arrays
    let mut pvalues: Array2<f64> = Array2::zeros((n_groups, control_chunk.shape.1));
    let mut u_stats: Array2<f64> = Array2::zeros((n_groups, control_chunk.shape.1));
    let mut zscores: Array2<f64> = Array2::zeros((n_groups, control_chunk.shape.1));
    for group_idx in 0..grpc.n_selected_groups {
        if group_idx == encoded_ref_group {
            pvalues.row_mut(group_idx).fill(1.);
            u_stats.row_mut(group_idx).fill(-1.);
            zscores.row_mut(group_idx).fill(0.);
        } else {
            // Chunk the target
            let start = grpc.indptr[group_idx as usize];
            let end = grpc.indptr[group_idx as usize + 1];
            let tgt_indices = grpc.indices.slice(s![start..end]);
            let mut tgt_chunk =
                x.index_rows_contig_cols_into_csc(chunk_lb, chunk_ub, tgt_indices)?;
            tgt_chunk.sort_columns_inplace()?;

            // Aggregate counts
            agg_count
                .row_mut(group_idx)
                .assign(&tgt_chunk.sum_axis0(is_log1p & !exp_post_agg)?);

            // Now compute p-values and u-stats
            single_group_sparse_ovo_mwu_kernel(
                &control_chunk,
                &tgt_chunk,
                &ctrl_n_uniques,
                &ctrl_offsets,
                &ctrl_tie_sums,
                use_continuity,
                tie_correct,
                alternative,
                pvalues.row_mut(group_idx),
                zscores.row_mut(group_idx),
                u_stats.row_mut(group_idx),
            )?;
        }
    }
    // Kind of ugly to force conversion here, but for now fold_change_from_summed_expr only accepts f32.
    // TODO: this could take D
    let fc = fold_change_from_summed_expr(
        agg_count.map(|&x| x.to_f64()),
        &grpc,
        is_log1p & exp_post_agg,
    )?;

    Ok((pvalues, u_stats, zscores, fc))
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
    alternative: &String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
    let chunk = x.contig_cols_into_csr(chunk_lb, chunk_ub)?;

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
    let mut control_chunk = chunk.index_rows_into_csc(control_indices)?;
    control_chunk.sort_columns_inplace()?;
    // Initiliaze aggregated count placeholder
    let n_groups = grpc.counts.len();
    let mut agg_count = Array2::<D>::zeros((n_groups, control_chunk.shape.1));
    agg_count
        .row_mut(encoded_ref_group)
        .assign(&control_chunk.sum_axis0(is_log1p & !exp_post_agg)?);

    // Pre-compute unique values and tie block offsets
    let (ctrl_offsets, ctrl_n_uniques, ctrl_tie_sums) =
        compute_sparse_unique_values_and_offsets(&mut control_chunk)?;

    // Allocate result arrays
    let mut pvalues = Array2::zeros((n_groups, control_chunk.shape.1));
    let mut u_stats = Array2::zeros((n_groups, control_chunk.shape.1));
    let mut zscores = Array2::zeros((n_groups, control_chunk.shape.1));
    for group_idx in 0..grpc.n_selected_groups {
        if group_idx == encoded_ref_group {
            pvalues.row_mut(group_idx).fill(1.);
            u_stats.row_mut(group_idx).fill(-1.);
            zscores.row_mut(group_idx).fill(0.);
        } else {
            // Chunk the target
            let start = grpc.indptr[group_idx as usize];
            let end = grpc.indptr[group_idx as usize + 1];
            let tgt_indices = grpc.indices.slice(s![start..end]);
            let mut tgt_chunk = chunk.index_rows_into_csc(tgt_indices)?;
            tgt_chunk.sort_columns_inplace()?;

            // Aggregate counts
            agg_count
                .row_mut(group_idx)
                .assign(&tgt_chunk.sum_axis0(is_log1p & !exp_post_agg)?);

            // Now compute p-values and u-stats
            single_group_sparse_ovo_mwu_kernel(
                &control_chunk,
                &tgt_chunk,
                &ctrl_n_uniques,
                &ctrl_offsets,
                &ctrl_tie_sums,
                use_continuity,
                tie_correct,
                alternative,
                pvalues.row_mut(group_idx),
                zscores.row_mut(group_idx),
                u_stats.row_mut(group_idx),
            )?;
        }
    }
    // Kind of ugly to force conversion here, but for now fold_change_from_summed_expr only accepts f32.
    // TODO: this could take D
    let fc = fold_change_from_summed_expr(
        agg_count.map(|&x| x.to_f64()),
        &grpc,
        is_log1p & exp_post_agg,
    )?;

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
    PyArr2f64<'py>,
)> {
    let grpc = grpc.as_group_container();

    let data_dtype: String = x.data.getattr("dtype")?.getattr("str")?.extract()?;
    let idx_dtype: String = x.indices.getattr("dtype")?.getattr("str")?.extract()?;
    let (pv, u, z, fc) = match (data_dtype.as_str(), idx_dtype.as_str()) {
        ("f32" | "<f4", "i32" | "<i4") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f32, i32
        ),
        ("f64" | "<f8", "i32" | "<i4") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f64, i32
        ),
        ("f32" | "<f4", "i64" | "<i8") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f32, i64
        ),
        ("f64" | "<f8", "i64" | "<i8") => run_branch!(
            "CSR", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f64, i64
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
    PyArr2f64<'py>,
)> {
    let grpc = grpc.as_group_container();

    let data_dtype: String = x.data.getattr("dtype")?.getattr("str")?.extract()?;
    let idx_dtype: String = x.indices.getattr("dtype")?.getattr("str")?.extract()?;
    let (pv, u, z, fc) = match (data_dtype.as_str(), idx_dtype.as_str()) {
        ("f32" | "<f4", "i32" | "<i4") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f32, i32
        ),
        ("f64" | "<f8", "i32" | "<i4") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f64, i32
        ),
        ("f32" | "<f4", "i64" | "<i8") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f32, i64
        ),
        ("f64" | "<f8", "i64" | "<i8") => run_branch!(
            "CSC", x, py, grpc, chunk_lb, chunk_ub, is_log1p, use_continuity, tie_correct, exp_post_agg, &alternative, f64, i64
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
