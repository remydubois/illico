use pyo3::prelude::*;
mod dense_ovo;
mod dense_ovr;
mod groups;
mod math;
mod ranking;
mod sparse;
mod stats;
use sparse::csc;
mod sparse_ovo;
mod sparse_ovr;

#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        ranking::sort_along_axis_0_inplace_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ranking::rank_sum_and_ties_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ranking::argsort_rust, m)?)?;
    m.add_function(wrap_pyfunction!(stats::compute_pvalue_rust, m)?)?;
    m.add_function(wrap_pyfunction!(math::chunk_and_fortranize_rust, m)?)?;
    m.add_function(wrap_pyfunction!(math::add_at_vec_rust, m)?)?;
    m.add_function(wrap_pyfunction!(math::dense_fold_change_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        math::fold_change_from_summed_expr_rust,
        m
    )?)?;
    // m.add_function(wrap_pyfunction!(sparse::csc::count_non_zeros_on_csc, m)?)?;
    // m.add_function(wrap_pyfunction!(sparse::csc::csc_contig_cols_into_csr, m)?)?;
    // m.add_function(wrap_pyfunction!(sparse::csr::csr_contig_cols_into_csr, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr::searchsorted_left_rust, m)?)?;
    // m.add_function(wrap_pyfunction!(sparse::csr::index_rows_into_csc_rust, m)?)?;
    // m.add_function(wrap_pyfunction!(
    //     sparse::csr::csr_contig_col_into_csc_rust,
    //     m
    // )?)?;
    m.add_function(wrap_pyfunction!(
        dense_ovo::dense_ovo_over_contiguous_col_chunk_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dense_ovo::dense_ovo_kernel_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        dense_ovr::dense_ovr_over_contiguous_col_chunk_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sparse_ovo::csc_ovo_mwu_kernel_over_contiguous_col_chunk_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sparse_ovo::csr_ovo_mwu_kernel_over_contiguous_col_chunk_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sparse_ovr::csc_ovr_mwu_kernel_over_contiguous_col_chunk_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sparse_ovr::csr_ovr_mwu_kernel_over_contiguous_col_chunk_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stats::erfc_rust, m)?)?;
    Ok(())
}
