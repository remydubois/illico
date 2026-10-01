use crate::groups::GroupContainer;
use crate::math::fold_change_from_summed_expr;
use crate::sparse::types::{CSCMatrix, OwnedCSCMatrix, OwnedCSRMatrix, SparseFloat, SparseIndex};
use ndarray::{Array1, Array2, s};

// impl<'a> CSCMatrix<'a> {
//     pub fn count_nonzeros(&'a self, axis: usize) -> Result<Array1<usize>, String> {
//         match axis {
//             0 => {
//                 let mut nnz = Array1::zeros(self.shape.1);
//                 for j in 0..(self.indptr.len() - 1) {
//                     nnz[j] = self.indptr[j + 1] - self.indptr[j];
//                 }
//                 return Ok(nnz);
//             }
//             1 => {
//                 let mut nnz = Array1::zeros(self.shape.0);
//                 for i in self.indices {
//                     nnz[*i as usize] += 1;
//                 }
//                 return Ok(nnz);
//             }
//             _ => return Err(format!("Axis must be 0 or 1. Received {axis}.")),
//         }
//     }
// }

impl<'a, D: SparseFloat, I: SparseIndex> CSCMatrix<'a, D, I> {
    pub fn contig_cols_into_csr(
        &'a self,
        chunk_lb: usize,
        chunk_ub: usize,
    ) -> Result<OwnedCSRMatrix<D, I>, String> {
        let mut chunk_nnz = Array1::<i32>::zeros(self.shape.0 + 1);
        for i in self.indices.slice(s![
            self.indptr[chunk_lb].to_usize()..self.indptr[chunk_ub].to_usize()
        ]) {
            chunk_nnz[(*i).to_usize() + 1] += 1
        }
        for i in 1..chunk_nnz.len() {
            chunk_nnz[i] += chunk_nnz[i - 1]
        } // Compute cumsum

        let nnz_total = chunk_nnz[chunk_nnz.len() - 1] as usize;
        let mut csc_data = Array1::<D>::zeros(nnz_total);
        let mut csc_indices = Array1::<I>::zeros(nnz_total);
        // let mut csc_indptr = Array1::zeros(nnz_total);
        let mut counter = chunk_nnz.mapv(|x| x as usize);
        for j in chunk_lb..chunk_ub {
            let col_idx = j - chunk_lb;
            for i in self.indptr[j].to_usize()..self.indptr[j + 1].to_usize() {
                let row_idx = self.indices[i].to_usize();
                csc_data[counter[row_idx]] = self.data[i];
                csc_indices[counter[row_idx]] = I::from(col_idx).unwrap();
                counter[row_idx] += 1;
            }
        }

        Ok(OwnedCSRMatrix {
            data: csc_data,
            indices: csc_indices,
            indptr: chunk_nnz.mapv(|i| I::from(i).unwrap()),
            shape: (self.shape.0, chunk_ub - chunk_lb),
        })
    }
}

// #[pyfunction]
// pub fn count_non_zeros_on_csc<'py>(
//     py: Python<'py>,
//     csc: PyCSCMatrix,
//     axis: usize,
// ) -> PyResult<Bound<'py, PyArray1<usize>>> {
//     let csc = csc.as_csc_matrix();
//     let nnz = csc.count_nonzeros(axis).map_err(PyValueError::new_err)?;
//     return Ok(PyArray1::from_array(py, &nnz));
// }

// #[pyfunction]
// pub fn csc_contig_cols_into_csr<'py>(
//     py: Python<'py>,
//     csc: PyCSCMatrix,
//     chunk_lb: usize,
//     chunk_ub: usize,
// ) -> PyResult<Bound<'py, PyTuple>> {
//     let csc = csc.as_csc_matrix();
//     let csr = csc
//         .contig_cols_into_csr(chunk_lb, chunk_ub)
//         .map_err(PyValueError::new_err)?;
//     // An alternative would be to implement a IntoPyobject for all CSC, CSR, OwnedCSC and OwnedCSR structs,
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

impl<D: SparseFloat, I: SparseIndex> OwnedCSCMatrix<D, I> {
    // This func will break indices and values matching, but we dont really care
    pub fn sort_columns_inplace(&mut self) -> Result<(), String> {
        for j in 0..self.shape.1 {
            let start = self.indptr[j].to_usize();
            let end = self.indptr[j + 1].to_usize();
            let mut col_view = self.data.slice_mut(s![start..end]);
            let col = col_view
                .as_slice_mut()
                .ok_or_else(|| format!("CSC matrix data should be col-contig"))?;
            // col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(())
    }
}

impl<I: SparseIndex, D: SparseFloat> OwnedCSCMatrix<D, I> {
    pub fn sum_axis0(&self, expm1: bool) -> Result<Array1<D>, String> {
        let mut out = Array1::<D>::zeros(self.shape.1);
        for j in 0..self.shape.1 {
            let start = self.indptr[j].to_usize();
            let end = self.indptr[j + 1].to_usize();
            if !expm1 {
                out[j] = self.data.slice(s![start..end]).sum();
            } else {
                out[j] = self.data.slice(s![start..end]).exp_m1().sum();
            }
        }
        Ok(out)
    }
}

pub fn csc_fold_change<D: SparseFloat, I: SparseIndex>(
    x: &OwnedCSCMatrix<D, I>,
    grpc: &GroupContainer,
    is_log1p: bool,
    exp_post_agg: bool,
) -> Result<Array2<f64>, String> {
    let mut summed_expr = Array2::zeros((grpc.counts.len(), x.shape.1));

    for j in 0..x.shape.1 {
        let (start, end) = (x.indptr[j].to_usize(), x.indptr[j + 1].to_usize());
        for i in start..end {
            let i = i as usize;
            let row_idx = x.indices[i].to_usize();
            let group_idx = grpc.encoded_groups[row_idx];
            let val = x.data[i].to_f32();
            summed_expr[[group_idx, j]] += if is_log1p && !exp_post_agg {
                val.exp_m1() as f64
            } else {
                val as f64
            };
        }
    }

    let fc = fold_change_from_summed_expr(summed_expr, &grpc, exp_post_agg && is_log1p)?;

    Ok(fc)
}
