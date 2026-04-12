use crate::groups::GroupContainer;
use crate::math::fold_change_from_summed_expr;
use crate::sparse::types::{CSRMatrix, OwnedCSCMatrix, OwnedCSRMatrix};
use crate::sparse::types::{SparseFloat, SparseIndex};
use ndarray::{Array1, Array2, ArrayView1, s};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// impl OwnedCSRMatrix {
impl<D: SparseFloat, I: SparseIndex> OwnedCSRMatrix<D, I> {
    pub fn index_rows_into_csc(
        &self,
        row_indices: ArrayView1<usize>,
    ) -> Result<OwnedCSCMatrix<D, I>, String> {
        let mut chunk_nnz = Array1::<i64>::zeros(self.shape.1 + 1);
        for i in row_indices {
            for j in self.indices.slice(s![
                self.indptr[*i].to_usize()..self.indptr[*i + 1].to_usize()
            ]) {
                chunk_nnz[(*j).to_usize() + 1] += 1
            }
        }
        for i in 1..chunk_nnz.len() {
            chunk_nnz[i] += chunk_nnz[i - 1]
        } // Compute cumsum

        let nnz_total = chunk_nnz[chunk_nnz.len() - 1] as usize;
        let mut csc_data = Array1::<D>::zeros(nnz_total);
        let mut csc_indices = Array1::<I>::zeros(nnz_total);
        let mut counter = chunk_nnz.mapv(|x| x.to_usize());
        let mut row_idx = 0;
        for j in row_indices {
            for i in self.indptr[*j].to_usize()..self.indptr[j + 1].to_usize() {
                let i = i as usize;
                let col_idx = self.indices[i].to_usize();
                csc_data[counter[col_idx]] = self.data[i];
                csc_indices[counter[col_idx]] = I::from(row_idx).ok_or_else(|| {
                    format!("Can't format row index into generic integer: {row_idx}")
                })?;
                counter[col_idx] += 1;
            }
            row_idx += 1;
        }

        Ok(OwnedCSCMatrix {
            data: csc_data,
            indices: csc_indices,
            indptr: chunk_nnz.mapv(|x| I::from(x).unwrap()),
            shape: (row_indices.len(), self.shape.1),
        })
    }
}

// #[pyfunction]
// pub fn index_rows_into_csc_rust<'py>(
//     py: Python<'py>,
//     csr: PyCSRMatrix<'py>,
//     row_indices: PyReadonlyArray1<usize>,
// ) -> PyResult<Bound<'py, PyTuple>> {
//     let csr = csr.as_owned_csr_matrix();
//     let row_indices = row_indices.as_array();
//     let csc = csr
//         .index_rows_into_csc(row_indices)
//         .map_err(PyValueError::new_err)?;

//     PyTuple::new(
//         py,
//         &[
//             PyArray1::from_array(py, &csc.data).into_any(),
//             PyArray1::from_array(py, &csc.indices).into_any(),
//             PyArray1::from_array(py, &csc.indptr).into_any(),
//             PyTuple::new(py, &[csc.shape.0, csc.shape.1])?.into_any(),
//         ],
//     )
// }

// TODO: this could be a method
pub fn csr_fold_change<D: SparseFloat, I: SparseIndex>(
    x: &OwnedCSRMatrix<D, I>,
    grpc: &GroupContainer,
    is_log1p: bool,
    exp_post_agg: bool,
) -> Result<Array2<f64>, String> {
    // Compute summed expression
    let mut summed_expr = Array2::<f64>::zeros((grpc.counts.len(), x.shape.1));
    for i in 0..x.shape.0 {
        let start = x.indptr[i].to_usize();
        let end = x.indptr[i + 1].to_usize();
        for pointer in start..end {
            let pointer = pointer;
            let col_idx = x.indices[pointer].to_usize();
            let group_idx = grpc.encoded_groups[i];
            let val = x.data[pointer].to_f32();
            summed_expr[[group_idx, col_idx]] += if is_log1p && !exp_post_agg {
                val.exp_m1() as f64
            } else {
                val as f64
            };
        }
    }
    let fc = fold_change_from_summed_expr(summed_expr, &grpc, exp_post_agg && is_log1p)?;
    Ok(fc)
}

// fn searchsorted_right(sorted_array: & [usize], value: usize) -> usize {
//     sorted_array.binary_search(&value)
//         .map(|idx| idx + 1)
//         .unwrap_or_else(|idx| idx)
// }

fn searchsorted_left<I: SparseIndex>(sorted_array: &[I], value: usize) -> usize {
    let value = I::from(value).unwrap();
    sorted_array.partition_point(|&x| x < value)
}

#[pyfunction]
pub fn searchsorted_left_rust(
    sorted_array: PyReadonlyArray1<i32>,
    value: usize,
) -> PyResult<usize> {
    Ok(searchsorted_left(
        sorted_array
            .as_array()
            .as_slice()
            .ok_or_else(|| format!("CSR indices should a C-contiguous array."))
            .map_err(PyValueError::new_err)?,
        value,
    ))
}

// Does this have to actually return owned data ? CSC should because of inplace sorting but this CSR is converted to CSC later on.
impl<'py, D: SparseFloat, I: SparseIndex> CSRMatrix<'py, D, I> {
    pub fn contig_cols_into_csr(
        &'py self,
        chunk_lb: usize,
        chunk_ub: usize,
    ) -> Result<OwnedCSRMatrix<D, I>, String> {
        let mut bounds = Array2::zeros((self.shape.0, 2));
        let mut n_nzeros = Array1::zeros(self.shape.0 + 1);
        let indices = self.indices.as_slice().ok_or_else(|| format!("Error"))?;
        for i in 0..self.shape.0 {
            let col_indices = &indices[self.indptr[i].to_usize()..self.indptr[i + 1].to_usize()];

            let cb = searchsorted_left(col_indices, chunk_lb);
            let rb = searchsorted_left(col_indices, chunk_ub);
            bounds[[i, 0]] = cb;
            bounds[[i, 1]] = rb;
            n_nzeros[i + 1] = rb - cb;
        }

        // Compute indptr: cumsum nonzeros
        // let mut indptr = n_nzeros.to_owned();
        let mut indptr = n_nzeros.mapv(|x| x as i32);
        for i in 1..indptr.len() {
            indptr[i] = indptr[i] + indptr[i - 1]
        }

        // Now retrieve data and indices for the chunk, across all rows
        let nnz_total = indptr[indptr.len() - 1] as usize;
        let mut new_data = vec![D::zero(); nnz_total];
        let mut new_indices = vec![I::zero(); nnz_total];
        let chunk_lb_gen = I::from(chunk_lb).unwrap();
        for i in 0..self.shape.0 {
            let org_start = self.indptr[i].to_usize();
            let (chunk_start, chunk_end) =
                (org_start + bounds[[i, 0]], (org_start + bounds[[i, 1]]));
            if chunk_start == chunk_end {
                continue;
            }

            let data_chunk = self.data.slice(s![chunk_start..chunk_end]).to_vec();
            new_data[indptr[i] as usize..indptr[i + 1] as usize].copy_from_slice(&data_chunk);
            let indices_chunk = self.indices.slice(s![chunk_start..chunk_end]).to_vec();
            // new_indices[indptr[i] as usize ..indptr[i + 1] as usize].copy_from_slice(&indices_chunk);
            for (k, i) in (indptr[i] as usize..indptr[i + 1] as usize).enumerate() {
                new_indices[i] = indices_chunk[k] - chunk_lb_gen
            }
        }

        Ok(OwnedCSRMatrix {
            data: Array1::from_vec(new_data),
            indices: Array1::from_vec(new_indices),
            indptr: indptr.mapv(|x| I::from(x).unwrap()),
            shape: (self.shape.0, chunk_ub - chunk_lb),
        })
    }
}

// #[pyfunction]
// pub fn csr_contig_cols_into_csr<'py>(
//     py: Python<'py>,
//     csr: PyCSRMatrix<'py>,
//     chunk_lb: usize,
//     chunk_ub: usize,
// ) -> PyResult<Bound<'py, PyTuple>> {
//     let csr = csr.as_csr_matrix();
//     let csr = csr
//         .contig_cols_into_csr(chunk_lb, chunk_ub)
//         .map_err(PyValueError::new_err)?;

//     PyTuple::new(
//         py,
//         &[
//             PyArray1::from_array(py, &csr.data).into_any(),
//             PyArray1::from_array(py, &csr.indices).into_any(),
//             PyArray1::from_array(py, &csr.indptr).into_any(),
//             PyTuple::new(py, &[csr.shape.0, csr.shape.1])?.into_any(),
//         ],
//     )
// }

impl<'py, D: SparseFloat, I: SparseIndex> CSRMatrix<'py, D, I> {
    pub fn contig_col_chunk_into_csc(
        &'py self,
        chunk_lb: usize,
        chunk_ub: usize,
    ) -> Result<OwnedCSCMatrix<D, I>, String> {
        let mut bounds = Array2::<usize>::zeros((self.shape.0, 2));
        let mut n_nzeros = Array1::<usize>::zeros(chunk_ub - chunk_lb + 1);
        for i in 0..self.shape.0 {
            let col_indices = self
                .indices
                .slice(s![self.indptr[i].to_usize()..self.indptr[i + 1].to_usize()]);
            let col_indices = col_indices
                .as_slice()
                .ok_or_else(|| format!("CSR indices should a C-contiguous array."))?;

            let cb = searchsorted_left(col_indices, chunk_lb);
            let rb = searchsorted_left(col_indices, chunk_ub);
            bounds[[i, 0]] = cb;
            bounds[[i, 1]] = rb;
            for j in cb..rb {
                let col_idx = col_indices[j].to_usize() - chunk_lb;
                n_nzeros[col_idx + 1] += 1
            }
        }
        // Compute cumsum
        let mut indptr = n_nzeros;
        for j in 1..indptr.len() {
            indptr[j] += indptr[j - 1]
        }

        let mut new_data = Array1::<D>::zeros(indptr[indptr.len() - 1]);
        let mut new_indices = Array1::zeros(indptr[indptr.len() - 1]);
        let mut counter = Array1::<usize>::zeros(chunk_ub - chunk_lb);
        for i in 0..self.shape.0 {
            let chunk_start = self.indptr[i].to_usize() + bounds[[i, 0]];
            let chunk_end = self.indptr[i].to_usize() + bounds[[i, 1]];
            for pointer in chunk_start..chunk_end {
                let pointer = pointer as usize;
                let col_idx = self.indices[pointer].to_usize() - chunk_lb;
                new_data[indptr[col_idx] + counter[col_idx]] = self.data[pointer];
                new_indices[indptr[col_idx] + counter[col_idx]] = I::from(i).unwrap();
                counter[col_idx] += 1;
            }
        }
        Ok(OwnedCSCMatrix {
            data: new_data,
            indices: new_indices,
            indptr: indptr.mapv(|x| I::from(x).unwrap()),
            shape: (self.shape.0, chunk_ub - chunk_lb),
        })
    }

    pub fn index_rows_contig_cols_into_csc(
        &'py self,
        chunk_lb: usize,
        chunk_ub: usize,
        row_indices: ArrayView1<usize>,
    ) -> Result<OwnedCSCMatrix<D, I>, String> {
        let mut bounds = Array2::<usize>::zeros((row_indices.len(), 2));
        let mut chunk_nnz = Array1::<i64>::zeros(chunk_ub - chunk_lb + 1);
        let indices = self
            .indices
            .as_slice()
            .ok_or_else(|| format!("CSR indices should a C-contiguous array."))?;

        for (row_idx, i) in row_indices.iter().enumerate() {
            let col_indices = &indices[self.indptr[*i].to_usize()..self.indptr[*i + 1].to_usize()];

            let cb = searchsorted_left(col_indices, chunk_lb);
            let rb = searchsorted_left(col_indices, chunk_ub);
            bounds[[row_idx, 0]] = cb;
            bounds[[row_idx, 1]] = rb;
            for j in cb..rb {
                let col_idx = col_indices[j].to_usize() - chunk_lb;
                chunk_nnz[col_idx + 1] += 1
            }
        }
        for i in 1..chunk_nnz.len() {
            chunk_nnz[i] += chunk_nnz[i - 1]
        }

        let nnz_total = chunk_nnz[chunk_nnz.len() - 1] as usize;
        let mut csc_data = Array1::<D>::zeros(nnz_total);
        let mut csc_indices = Array1::<I>::zeros(nnz_total);
        let mut counter = chunk_nnz.mapv(|x| x.to_usize());
        for (row_idx, j) in row_indices.iter().enumerate() {
            let org_start = self.indptr[*j].to_usize();
            let chunk_start = org_start + bounds[[row_idx, 0]];
            let chunk_end = org_start + bounds[[row_idx, 1]];
            for i in chunk_start..chunk_end {
                let i = i as usize;
                let col_idx = self.indices[i].to_usize() - chunk_lb;
                csc_data[counter[col_idx]] = self.data[i];
                csc_indices[counter[col_idx]] = I::from(row_idx).ok_or_else(|| {
                    format!("Can't format row index into generic integer: {row_idx}")
                })?;
                counter[col_idx] += 1;
            }
        }

        Ok(OwnedCSCMatrix {
            data: csc_data,
            indices: csc_indices,
            indptr: chunk_nnz.mapv(|x| I::from(x).unwrap()),
            shape: (row_indices.len(), chunk_ub - chunk_lb),
        })
    }
}

// #[pyfunction]
// pub fn csr_contig_col_into_csc_rust<'py>(
//     py: Python<'py>,
//     csr: PyCSRMatrix<'py>,
//     chunk_lb: usize,
//     chunk_ub: usize,
// ) -> PyResult<Bound<'py, PyTuple>> {
//     let csr = csr.as_csr_matrix();
//     let csc = csr
//         .contig_col_chunk_into_csc(chunk_lb, chunk_ub)
//         .map_err(PyValueError::new_err)?;

//     PyTuple::new(
//         py,
//         &[
//             PyArray1::from_array(py, &csc.data).into_any(),
//             PyArray1::from_array(py, &csc.indices).into_any(),
//             PyArray1::from_array(py, &csc.indptr).into_any(),
//             PyTuple::new(py, &[csc.shape.0, csc.shape.1])?.into_any(),
//         ],
//     )
// }
