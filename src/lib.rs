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
    m.add_function(wrap_pyfunction!(
        dense_ovo::dense_ovo_over_contiguous_col_chunk_rust,
        m
    )?)?;
    // m.add_function(wrap_pyfunction!(dense_ovo::dense_ovo_kernel_rust, m)?)?;
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
    Ok(())
}
