use ndarray::{ArrayView1, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2};
use numpy::PyArray1;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadwriteArray2};
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::sparse::types::SparseFloat;

// pub fn sort_along_axis_0(x: &ArrayView2<f64>) -> Array2<f64> {
//     let (nrows, ncols) = x.dim();
//     let mut output = Array2::zeros((nrows, ncols));

//     for (col_idx, col) in x.columns().into_iter().enumerate() {
//         let mut vec = col.to_vec();
//         vec.sort_unstable_by(|a, b| a.total_cmp(b));
//         // output.column_mut(col_idx) = vec
//         for (row_idx, val) in vec.into_iter().enumerate() {
//             output[[row_idx, col_idx]] = val;
//         }
//     }

//     output
// }

// #[pyfunction]
// pub fn sort_along_axis_0_rust<'py>(
//     py: Python<'py>,
//     x: Bound<'py, PyArray2<f64>>,
// ) -> Bound<'py, PyArray2<f64>> {
//     let x = unsafe { x.as_array() };
//     let result = sort_along_axis_0(&x);
//     PyArray2::from_array(py, &result).into()
// }

pub fn sort_along_axis_0_inplace<D: SparseFloat>(mut x: ArrayViewMut2<D>) -> Result<(), String> {
    for mut col in x.columns_mut() {
        let col = col
            .as_slice_mut()
            .ok_or_else(|| format!("Columns must be contiguous data"))?;
        col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    return Ok(());
}

#[pyfunction]
pub fn sort_along_axis_0_inplace_rust(mut x: PyReadwriteArray2<f32>) -> PyResult<()> {
    let x = x.as_array_mut();
    sort_along_axis_0_inplace(x).map_err(PyValueError::new_err)?;
    Ok(())
}

pub fn rank_sum_and_ties<D: SparseFloat>(ctrl: ArrayView1<D>, tgt: ArrayView1<D>) -> (f64, f64) {
    let n_ctrl = ctrl.len();
    let n_tgt = tgt.len();

    // Initialize pointers
    let mut i = 0;
    let mut j = 0;
    let mut k: usize = 0;

    // Initialize accumulators
    let mut rank_sum_tgt: f64 = 0.;
    let mut tie_sum: f64 = 0.;

    // Go through both sorted arrays first
    while i < n_ctrl && j < n_tgt {
        let v = ctrl[i].min(tgt[j]);

        // Count unique values in control values
        let mut count_ctrl: usize = 0;
        let mut offset_i = i;
        while offset_i < n_ctrl && ctrl[offset_i] == v {
            count_ctrl += 1;
            offset_i += 1
        }
        // let ctrl_tie_block_size: &usize = &ctrl.slice(s![i..]).iter().take_while(|&x| x == &v).count();
        // let tgt_tie_block_size: &usize = &tgt.slice(s![j..]).iter().take_while(|&x| x == &v).count();
        // for l in i..n_ctrl {
        //     if ctrl[l] == v {
        //         count_ctrl += 1.;
        //     } else {
        //         break;
        //     }
        // }

        // Count unique values in target values
        let mut count_tgt: usize = 0;
        let mut offset_j = j;
        while offset_j < n_tgt && tgt[offset_j] == v {
            count_tgt += 1;
            offset_j += 1
        }
        // let mut count_tgt: f64 = 0.;
        // for l in j..n_tgt {
        //     if tgt[l] == v {
        //         count_tgt += 1.;
        //     } else {
        //         break;
        //     }
        // }

        // Compute ties and average rank
        let total_count = count_ctrl + count_tgt;
        let avg_rank = k as f64 + (total_count as f64 + 1.) * 0.5;
        // Update rank sum and tie sum
        rank_sum_tgt += count_tgt as f64 * avg_rank;
        tie_sum += (total_count.pow(3) - total_count) as f64;

        // Update counters
        // i += count_ctrl as usize;
        // j += count_tgt as usize;
        // k += total_count;
        i += count_ctrl;
        j += count_tgt;
        k += total_count;
    }
    // Drain remaining control elements
    while i < n_ctrl {
        let v = ctrl[i];

        // Count unique values in control values
        let mut count_ctrl: usize = 0;
        for l in i..n_ctrl {
            if ctrl[l] == v {
                count_ctrl += 1;
            } else {
                break;
            }
        }
        // Update tie sum, we don't update ranksum for controls
        tie_sum += (count_ctrl.pow(3) - count_ctrl) as f64;

        // Update counters
        i += count_ctrl;
        k += count_ctrl;
    }

    // Drain remaining target (perturbed) elements
    while j < n_tgt {
        let v = tgt[j];

        // Count unique values in target values
        let mut count_tgt: usize = 0;
        for l in j..n_tgt {
            if tgt[l] == v {
                count_tgt += 1;
            } else {
                break;
            }
        }

        // Compute ties and average rank
        let avg_rank = k as f64 + (count_tgt as f64 + 1.) * 0.5;
        // Update rank sum and tie sum
        rank_sum_tgt += count_tgt as f64 * avg_rank;
        tie_sum += (count_tgt.pow(3) - count_tgt) as f64;

        // Update counters
        j += count_tgt;
        k += count_tgt;
    }

    return (rank_sum_tgt, tie_sum);
}

#[pyfunction]
pub fn rank_sum_and_ties_rust(
    controls: PyReadonlyArray1<f32>,
    target: PyReadonlyArray1<f32>,
) -> (f64, f64) {
    let controls = controls.as_array();
    let target = target.as_array();
    let (ranksum, tiesum) = rank_sum_and_ties(controls, target);
    return (ranksum, tiesum);
}

pub fn accumulate_rank_and_tie_sums_from_argsort<D: SparseFloat>(
    x: ArrayView1<D>,
    sorted_indices: Vec<usize>,
    group_labels: ArrayView1<usize>,
    mut ranksums: ArrayViewMut1<f64>,
    mut tie_sum: ArrayViewMut0<f64>,
) -> Result<(), String> {
    let n_sorted_idx = sorted_indices.len();
    let n_values = x.len();
    let n_grp_indices = group_labels.len();
    let n_groups = ranksums.len();
    if n_values != n_sorted_idx {
        return Err(format!(
            "Argsort indices not compatible with values to index"
        ));
    }
    if n_values != n_grp_indices {
        return Err(format!("Group indices not compatible with values to index"));
    }
    if n_sorted_idx != n_grp_indices {
        return Err(format!("Argsort indices not compatible with group indices"));
    }
    if let Some(value) = sorted_indices.iter().max() {
        if value >= &n_values {
            return Err(format!(
                "Out of bounds error: {} values but indices up to {}",
                { n_values },
                { value }
            ));
        }
    }
    if let Some(value) = group_labels.iter().max() {
        if value >= &n_groups {
            return Err(format!(
                "Out of bounds error: {} groups but indices up to {}",
                { n_groups },
                { value }
            ));
        }
    }

    let mut i: usize = 0;
    while i < n_values {
        // Find tie block
        let mut j = i + 1;
        while j < n_values && x[sorted_indices[j]] == x[sorted_indices[i]] {
            j += 1
        }
        // Now by def: j - i gives the size of the tie block
        // Compute the average rank for this block
        let avg_rank = 0.5 * (i + j + 1) as f64;

        // Accumulate this avg rank in each group's palceholder
        for k in i..j {
            // This gives the index of the value
            let idx = sorted_indices[k];
            // This gives the index of the row where to accumulate rank sums
            let row_idx = group_labels[idx];
            ranksums[row_idx] += avg_rank
        }

        // Now take care of the tie sum
        let tie_block_size: f64 = (j - i) as f64;
        tie_sum += tie_block_size.powi(3) - tie_block_size;
        i = j;
    }

    Ok(())
}

// Because only the indices are mutated in argsort, input values do not need to be a vec or a contiguous memory alloc
pub fn argsort<D: SparseFloat>(x: ArrayView1<D>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    indices
}

#[pyfunction]
pub fn argsort_rust<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let x = x.as_array(); // unwrap the Result of as_slice
    let indices = argsort(x);
    Ok(PyArray1::from_vec(py, indices))
}
