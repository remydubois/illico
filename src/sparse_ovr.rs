use crate::csc::csc_fold_change;
use crate::groups::{GroupContainer, GroupContainerNamedTuple};
use crate::ranking::{accumulate_rank_and_tie_sums_from_argsort, argsort};
use crate::sparse::types::{
    CSCMatrix, CSRMatrix, OwnedCSCMatrix, PyCSCMatrix, PyCSRMatrix, SparseFloat, SparseIndex,
};
use crate::stats::compute_pvalue;
use ndarray::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn sparse_ovr_mwu_kernel<D: SparseFloat, I: SparseIndex>(
    x: &OwnedCSCMatrix<D, I>,
    grpc: &GroupContainer,
    use_continuity: bool,
    tie_correct: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
    let n_cols = x.shape.1;
    let n_zeros = x.shape.0 - x.indptr.diff(1, Axis(0)).mapv(|x| x.to_usize());

    // Allocate placeholders for results
    let mut p_values = Array2::zeros((grpc.counts.len(), x.shape.1));
    let mut u_stats = Array2::zeros((grpc.counts.len(), x.shape.1));
    let mut zscores = Array2::zeros((grpc.counts.len(), x.shape.1));

    let n_total = x.shape.0 as f64;
    let n_ref = grpc.counts.mapv(|x| n_total - x as f64);
    let n_tgt = grpc.counts.mapv(|x| x as f64);
    let mu = &n_ref * &n_tgt / 2.;

    // In theory I could allocate just once and reset for each column but then the typing for accumulate_rank_and_tie_sums_from_argsort
    // is non-natural (because I use the same func for the dense use case)
    let mut ranksum = Array2::zeros((grpc.counts.len(), x.shape.1));
    let mut tiesum = Array1::<f64>::zeros(x.shape.1);
    let mut nnz_per_group = Array2::<usize>::zeros((grpc.counts.len(), x.shape.1));
    let remainder = &n_tgt * (&n_tgt + 1.) / 2.;
    // Note: ideally I would benchmark between the two scenarios: keep all of n, nz, nnz as usize and convert the end result to f64,
    // or convert them right away to f64. Summation on f64 is slower than on integers but casting and memory alloc of vectors takes time.
    for j in 0..n_cols {
        let (start, end) = (x.indptr[j].to_usize(), x.indptr[j + 1].to_usize());
        let nz_row_idx = x.indices.slice(s![start..end]);

        // Ranksum and tie sum for non zero values
        let col_nz_values = x.data.slice(s![start..end]);
        let argsorted_idxs = argsort(col_nz_values);
        // let nz_row_indices = x.indices.slice(s![start..end]).mapv(|x| x as usize);
        let nz_group_labels = x
            .indices
            .slice(s![start..end])
            .mapv(|x| grpc.encoded_groups[x.to_usize()]);
        accumulate_rank_and_tie_sums_from_argsort(
            col_nz_values,
            argsorted_idxs,
            nz_group_labels.view(),
            ranksum.column_mut(j),
            tiesum.slice_mut(s![j]),
        )?;

        // Need to offset the ranks by the number of zeros per group
        for i in nz_row_idx {
            // &nnz_per_group[[grpc.encoded_groups[*i], j]] += 1
            nnz_per_group.column_mut(j)[grpc.encoded_groups[(*i).to_usize()]] += 1
        }
        // This is messy: why nz_per_group is column-unique while nnz_per_group is all columns
        let nz_per_group = &grpc.counts - &nnz_per_group.column(j);
        let mut rs = ranksum.column_mut(j);
        rs += &(n_zeros[j] * &nnz_per_group.column(j)).mapv(|x| x as f64); // nnz and not nz !

        // Now need to add the contributions of zero to the ranksum of each group
        rs += &((nz_per_group * (n_zeros[j] + 1)).mapv(|x| x as f64) / 2.);
        let mut ts = tiesum.slice_mut(s![j]);
        ts += (n_zeros[j].pow(3) - n_zeros[j]) as f64;

        // Now compute u-stat: one value per group
        let u_stat = &rs - &remainder;
        u_stats.column_mut(j).assign(&u_stat); // Assign

        // Now compute p-values
        for k in 0..grpc.counts.len() {
            let (p, z) = compute_pvalue(
                n_ref[k],
                n_tgt[k],
                n_total,
                if tie_correct { tiesum[j] } else { 0. },
                u_stat[k],
                mu[k],
                if use_continuity { 0.5 } else { 0. },
                &alternative,
            )?;
            p_values[[k, j]] = p;
            zscores[[k, j]] = z;
        }
    }
    Ok((p_values, u_stats, zscores))
}

pub fn csr_ovr_mwu_kernel_over_contiguous_col_chunk<'py, D: SparseFloat, I: SparseIndex>(
    x: &'py CSRMatrix<'py, D, I>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
    let csc_chunk = x.contig_col_chunk_into_csc(chunk_lb, chunk_ub)?;

    let (p_values, u_stats, zscores) =
        sparse_ovr_mwu_kernel(&csc_chunk, &grpc, use_continuity, tie_correct, alternative)?;

    let fc = csc_fold_change(&csc_chunk, &grpc, is_log1p, exp_post_agg)?;
    Ok((p_values, u_stats, zscores, fc))
}

pub fn csc_ovr_mwu_kernel_over_contiguous_col_chunk<'py, D: SparseFloat, I: SparseIndex>(
    x: &'py CSCMatrix<'py, D, I>,
    chunk_lb: usize,
    chunk_ub: usize,
    grpc: GroupContainer,
    is_log1p: bool,
    use_continuity: bool,
    tie_correct: bool,
    exp_post_agg: bool,
    alternative: String,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
    let csc_chunk = x.contig_col_chunk_into_csc(chunk_lb, chunk_ub)?;

    let (p_values, u_stats, zscores) =
        sparse_ovr_mwu_kernel(&csc_chunk, &grpc, use_continuity, tie_correct, alternative)?;

    let fc = csc_fold_change(&csc_chunk, &grpc, is_log1p, exp_post_agg)?;
    Ok((p_values, u_stats, zscores, fc))
}

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
                    // ovo_mwu_kernel_over_contiguous_col_chunk!(
                    csr_ovr_mwu_kernel_over_contiguous_col_chunk(
                        &csr, $chunk_lb, $chunk_ub, $grpc, $is_log1p, $use_continuity, $tie_correct, $exp_post_agg, $alternative,
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
                    csc_ovr_mwu_kernel_over_contiguous_col_chunk(
                        &csc, $chunk_lb, $chunk_ub, $grpc, $is_log1p, $use_continuity, $tie_correct, $exp_post_agg, $alternative,
                    )
                })
                .map_err(PyValueError::new_err)
            }
            _ => panic!("Unkown format"),
        }
    }};
}

type PyArr2f32<'py> = Bound<'py, PyArray2<f32>>;
type PyArr2f64<'py> = Bound<'py, PyArray2<f64>>;

#[rustfmt::skip]
#[pyfunction]
pub fn csr_ovr_mwu_kernel_over_contiguous_col_chunk_rust<'py>(
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
    let indices_dtype: String = x.indices.getattr("dtype")?.getattr("str")?.extract()?;

    let (pvalues, u_stats, zscores, fc) = match (data_dtype.as_str(), indices_dtype.as_str()) {
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
            data_dtype, indices_dtype
        ))),
    }?;

    Ok((
        PyArray2::from_array(py, &pvalues),
        PyArray2::from_array(py, &u_stats),
        PyArray2::from_array(py, &zscores),
        PyArray2::from_array(py, &fc),
    ))
}

#[rustfmt::skip]
#[pyfunction]
pub fn csc_ovr_mwu_kernel_over_contiguous_col_chunk_rust<'py>(
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
    let indices_dtype: String = x.indices.getattr("dtype")?.getattr("str")?.extract()?;

    let (pvalues, u_stats, zscores, fc) = match (data_dtype.as_str(), indices_dtype.as_str()) {
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
            data_dtype, indices_dtype
        ))),
    }?;

    Ok((
        PyArray2::from_array(py, &pvalues),
        PyArray2::from_array(py, &u_stats),
        PyArray2::from_array(py, &zscores),
        PyArray2::from_array(py, &fc),
    ))
}
