use crate::groups::{GroupContainer, GroupContainerNamedTuple};
use crate::sparse::types::SparseFloat;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, ShapeBuilder, Zip};
use numpy::PyReadonlyArray2;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::Python;
use pyo3::{exceptions::PyValueError, prelude::*};

pub fn chunk_and_fortranize<D: SparseFloat>(
    x: &ArrayView2<D>,
    chunk_lb: usize,
    chunk_ub: usize,
    indices: Option<ArrayView1<usize>>,
) -> Result<Array2<D>, String> {
    let ncols = chunk_ub - chunk_lb;

    if chunk_ub > x.dim().1 {
        return Err(format!(
            "Chunk upper bound {} is larger than the number of columns {}",
            chunk_ub,
            x.dim().1
        ));
    }

    match indices {
        Some(indices) => {
            let nrows = x.dim().0;

            if let Some(idxmax) = indices.iter().max() {
                if idxmax >= &nrows {
                    return Err(format!(
                        "Indices are out of bounds: {} is bigger than {}",
                        { idxmax },
                        { nrows }
                    ));
                }
            }

            let nrows = indices.dim();
            let mut output = Array2::zeros((nrows, ncols).f());
            let indices = indices
                .as_slice()
                .ok_or_else(|| format!("Group indices must be mem-contiguous"))?;
            for (j, col_idx) in (chunk_lb..chunk_ub).enumerate() {
                let mut col = output.column_mut(j);
                // TODO: avoid as_slice which is not super clean. Wont crash tho as group indices are mem contiguous.
                col.assign(&x.column(col_idx).select(Axis(0), indices));
                // output[[i, j]] = x[[*row_idx, col_idx]]
            }
            return Ok(output);
        }
        None => {
            let nrows = x.dim().0;
            let mut output = Array2::zeros((nrows, ncols).f());
            for (j, col_idx) in (chunk_lb..chunk_ub).enumerate() {
                let mut col = output.column_mut(j);
                col.assign(&x.column(col_idx))
            }
            return Ok(output);
        }
    }
}

#[pyfunction]
pub fn chunk_and_fortranize_rust<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArray2<f32>>,
    chunk_lb: usize,
    chunk_ub: usize,
    indices: Option<PyReadonlyArray1<'py, usize>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = unsafe { x.as_array() };
    let option_indices = indices.as_ref().map(|indices| indices.as_array());
    // let option_indices = indices.map(|idx| idx.as_array());

    let chunk = chunk_and_fortranize(&x, chunk_lb, chunk_ub, option_indices)
        .map_err(PyValueError::new_err)?;
    Ok(PyArray2::from_array(py, &chunk))
}

pub fn add_at_vec(
    mut x: ArrayViewMut1<f32>,
    y: ArrayView1<f32>,
    indices: ArrayView1<usize>,
) -> Result<(), String> {
    let n_indices = indices.len();
    let n_values = y.len();
    let max_idx = indices.iter().max();
    // This is purely educational, there is a more concise syntax for that
    match max_idx {
        Some(value) => {
            if *value >= x.len() {
                return Err(format!(
                    "Out-of-bound error: {} is not smaller than {}",
                    { value },
                    { x.len() }
                ));
            }
        }
        None => {
            if n_values > 0 {
                return Err(format!("Indices is empty but not values."));
            }
        }
    }

    /*
    An alternative and shorter syntax is:
    if let Some(value) = indices.iter().max() {
        if *value >= x.len() {return Err(format!("Out-of-bound error: {} is not smaller than {}", {value}, {x.len()}));}
    } else {
        if n_values > 0 {return Err(format!("Indices is empty but not values."));}
    }
     */

    if n_indices != n_values {
        return Err(format!(
            "Values and indices have different sizes: {} and {}",
            { n_values },
            { n_indices }
        ));
    }
    Zip::from(&y).and(&indices).for_each(|&v, &i| x[[i]] += v);
    Ok(())
}

#[pyfunction]
pub fn add_at_vec_rust(
    mut x: PyReadwriteArray1<f32>,
    y: PyReadonlyArray1<f32>,
    indices: PyReadonlyArray1<usize>,
) -> PyResult<()> {
    let x = x.as_array_mut();
    let y = y.as_array();
    let indices = indices.as_array();
    add_at_vec(x, y, indices).map_err(PyValueError::new_err)?;
    Ok(())
}

pub fn fold_change_from_summed_expr(
    summed_x: Array2<f64>,
    grpc: &GroupContainer,
    exp_post_agg: bool,
) -> Result<Array2<f64>, String> {
    // Convert and unsqueeze the counts
    let counts = grpc.counts.map(|x| *x as f64).insert_axis(Axis(1)); // (#groups, 1)
    let mu_tgt = &summed_x / &counts;
    // Idk how to avoid repeating the if exp_post_agg else syntax
    if grpc.encoded_ref_group == -1 {
        let total_count = counts.sum();
        // this one contains, for each group, the count of all cell minus itself
        let other_count = total_count - counts;
        let ctrl_sum = summed_x.sum_axis(Axis(0)).insert_axis(Axis(0)) - summed_x;
        let mu_ctrl = ctrl_sum / other_count;
        // println!("Mu ctrl: {:?}", mu_ctrl);
        if exp_post_agg {
            return Ok((mu_tgt.exp_m1() + 1e-9) / (mu_ctrl.exp_m1() + 1e-9));
        } else {
            return Ok((mu_tgt + 1e-9) / (mu_ctrl + 1e-9));
        }
    } else {
        if grpc.encoded_ref_group < 0 {
            return Err(format!(
                "Encoded ref group can not be negative. Received {}.",
                grpc.encoded_ref_group
            ));
        }
        let mu_ctrl = mu_tgt
            .row(grpc.encoded_ref_group as usize)
            .insert_axis(Axis(0));
        if exp_post_agg {
            return Ok((mu_tgt.exp_m1() + 1e-9) / (mu_ctrl.exp_m1() + 1e-9));
        } else {
            return Ok((&mu_tgt + 1e-9) / (&mu_ctrl + 1e-9));
        }
    }
}

#[pyfunction]
pub fn fold_change_from_summed_expr_rust<'py>(
    py: Python<'py>,
    summed_x: PyReadonlyArray2<'py, f64>,
    grpc: GroupContainerNamedTuple<'py>,
    exp_post_agg: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = summed_x.as_array().to_owned();
    let grpc = grpc.as_group_container();
    let fc = fold_change_from_summed_expr(x, &grpc, exp_post_agg).map_err(PyValueError::new_err)?;
    return Ok(PyArray2::from_array(py, &fc));
}

pub fn dense_fold_change<D: SparseFloat>(
    x: ArrayView2<D>,
    grpc: &GroupContainer,
    is_log1p: bool,
    exp_post_agg: bool,
) -> Result<Array2<f64>, String> {
    let n_groups = grpc.counts.len();
    let mut group_agg_counts = Array2::<f64>::zeros((n_groups, x.dim().1));

    let row_indexer = grpc.encoded_groups;
    // Check on the row indices to catch out of bounds
    if let Some(max_idx) = row_indexer.iter().max() {
        if max_idx >= &n_groups {
            return Err(format!(
                "Encoded groups max to {} but only {} distinct groups recorded.",
                { max_idx },
                { n_groups }
            ));
        }
    } else {
        // Enter this branch if max() returns None, which happens if row_indexer is empty
        if x.dim().0 > 0 {
            return Err(format!("Non null number of cells but no groups indicated."));
        }
    }

    if is_log1p && !exp_post_agg {
        for i in 0..x.dim().0 {
            let mut row = group_agg_counts.row_mut(grpc.encoded_groups[i]);
            // row += &x.row(i).exp_m1();
            for j in 0..x.dim().1 {
                row[j] += x[[i, j]].to_f64().exp_m1()
            }
        }
    } else {
        for i in 0..x.dim().0 {
            let mut row = group_agg_counts.row_mut(grpc.encoded_groups[i]);
            // row += &x.row(i);
            for j in 0..x.dim().1 {
                row[j] += x[[i, j]].to_f64()
            }
        }
    }
    // return Ok(group_agg_counts);
    let fold_change =
        fold_change_from_summed_expr(group_agg_counts, grpc, exp_post_agg && is_log1p)?;

    return Ok(fold_change);
}

#[pyfunction]
pub fn dense_fold_change_rust<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    grpc: GroupContainerNamedTuple<'py>,
    is_log1p: bool,
    exp_post_agg: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = x.as_array();
    let grpc = grpc.as_group_container();
    let fc = dense_fold_change(x, &grpc, is_log1p, exp_post_agg).map_err(PyValueError::new_err)?;

    Ok(PyArray2::from_array(py, &fc))
}
