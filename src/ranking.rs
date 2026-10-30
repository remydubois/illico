use crate::sparse::types::{SparseFloat, SparseIndex};
use ndarray::{ArrayView1, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2};

pub fn sort_along_axis_0_inplace<D: SparseFloat>(mut x: ArrayViewMut2<D>) -> Result<(), String> {
    for mut col in x.columns_mut() {
        let col = col
            .as_slice_mut()
            .ok_or_else(|| format!("Columns must be contiguous data"))?;
        col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    return Ok(());
}

pub fn rank_sum_and_ties<D: SparseFloat>(
    ctrl: ArrayView1<D>,
    tgt: ArrayView1<D>,
    mut zero_values_offset: usize,
) -> (f64, f64, usize) {
    let n_ctrl = ctrl.len();
    let n_tgt = tgt.len();

    // Initialize pointers
    let mut i = 0;
    let mut j = 0;
    let mut k: usize = 0;
    let mut zero_pos: isize = -1;

    // Initialize accumulators
    let mut rank_sum_tgt: f64 = 0.;
    let mut tie_sum: f64 = 0.;

    // Go through both sorted arrays first
    while i < n_ctrl && j < n_tgt {
        let v = ctrl[i].min(tgt[j]);

        if (v.to_f64() > 0.) & (zero_values_offset > 0) {
            zero_pos = k as isize;
            k += zero_values_offset;
            zero_values_offset = 0;
        }

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
        if (v.to_f64() > 0.) & (zero_values_offset > 0) {
            zero_pos = k as isize;
            k += zero_values_offset;
            zero_values_offset = 0;
        }

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

        if (v.to_f64() > 0.) && (zero_values_offset > 0) {
            zero_pos = k as isize;
            k += zero_values_offset;
            zero_values_offset = 0;
        }

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

    if (zero_pos == -1) && (zero_values_offset > 0) {
        zero_pos = k as isize;
    }

    return (rank_sum_tgt, tie_sum, zero_pos as usize);
}

pub fn accumulate_rank_and_tie_sums_from_argsort<D: SparseFloat>(
    x: ArrayView1<D>,
    sorted_indices: Vec<usize>,
    group_labels: ArrayView1<usize>,
    mut ranksums: ArrayViewMut1<f64>,
    mut tie_sum: ArrayViewMut0<f64>,
    mut zero_values_offset: usize,
) -> Result<usize, String> {
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
    let mut rank: usize = 0;
    let mut zero_pos: isize = -1;
    while i < n_values {
        // Find tie block
        let mut j = i + 1;
        if (zero_values_offset > 0) && (x[sorted_indices[i]].to_f64() > 0.) {
            zero_pos = i as isize;
            rank += zero_values_offset;
            zero_values_offset = 0;
        }
        while j < n_values && x[sorted_indices[j]] == x[sorted_indices[i]] {
            j += 1
        }
        // Now by def: j - i gives the size of the tie block
        // Compute the average rank for this block
        let avg_rank = rank as f64 + 0.5 * (j - i + 1) as f64;

        // Accumulate this avg rank in each group's palceholder
        for k in i..j {
            // This gives the index of the value
            let idx = sorted_indices[k];
            // This gives the index of the row where to accumulate rank sums
            let row_idx = group_labels[idx];
            ranksums[row_idx] += avg_rank
        }

        // Now take care of the tie sum
        let tie_block_size = j - i;
        tie_sum += tie_block_size.pow(3) as f64 - tie_block_size as f64;
        rank += tie_block_size;
        i = j;
    }

    if (zero_pos == -1) && (zero_values_offset > 0) {
        zero_pos = n_values as isize;
    }

    Ok(zero_pos as usize)
}

// Because only the indices are mutated in argsort, input values do not need to be a vec or a contiguous memory alloc
pub fn argsort<D: SparseFloat>(x: ArrayView1<D>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    indices
}

pub fn unique_from_sorted<D: SparseFloat>(
    mut x: ArrayViewMut1<D>,
    mut counts: ArrayViewMut1<usize>,
) -> Result<usize, String> {
    let n_vals = x.dim();
    if n_vals == 0 {
        return Ok(0);
    };
    let mut prev_val: D = x[0];
    let mut count: usize = 0;
    let mut k: usize = 0;
    for i in 0..n_vals {
        let val = x[i];
        if val == prev_val {
            count += 1;
        } else {
            x[k] = prev_val;
            counts[k] = count;
            k += 1;
            count = 1;
            prev_val = val;
        }
    }
    // Finalize block
    x[k] = prev_val;
    counts[k] = count;
    k += 1;

    Ok(k) // Return number of unique values
}

#[inline]
pub fn tie_sum_delta(a: usize, b: usize) -> Result<usize, String> {
    let increment: usize = b * (3 * (a.pow(2)) + 3 * a * b + b.pow(2) - 1);
    Ok(increment)
}

pub fn searchsorted_left<I: SparseIndex>(sorted_array: &[I], value: usize) -> usize {
    let value = I::from(value).unwrap();
    sorted_array.partition_point(|&x| x < value)
}

pub fn fsearchsorted_left<D: SparseFloat>(sorted_array: &[D], value: D) -> usize {
    sorted_array.partition_point(|&x| x < value)
}

pub fn partial_left_binsearch<D: SparseFloat>(
    arr: ArrayView1<D>,
    x: D,
    mut lo: usize,
    high: Option<usize>,
) -> Result<usize, String> {
    let mut high_bound: usize = high.unwrap_or(arr.dim());
    while lo < high_bound {
        let mid = (lo + high_bound) / 2;
        if arr[mid] < x {
            lo = mid + 1;
        } else {
            high_bound = mid;
        }
    }
    Ok(lo)
}

pub fn rank_sum_and_ties_from_binsearch<D: SparseFloat>(
    ctrl_values: ArrayView1<D>,
    n_uniques: usize,
    offsets: ArrayView1<usize>,
    pert_values: ArrayView1<D>,
    zero_values_offset: usize,
) -> Result<(f64, usize, usize), String> {
    let n_ctrl = n_uniques;
    let n_pert: usize = pert_values.dim();

    // Fallback to the linear merge algorithm, which executes the exhaust loop, not a proper merge
    if n_ctrl == 0 {
        let (rs, ts, zpos) = rank_sum_and_ties(ctrl_values, pert_values, zero_values_offset);
        return Ok((rs, ts as usize, zpos));
    }

    // Prepare accumulators
    let mut tie_sum: usize = 0;
    let mut rank_sum: f64 = 0.0;
    let mut zero_pos: Option<usize> = None;
    let zero_val = D::from(0.).unwrap();
    // Prepare var to track position of zeros in the sorted array
    let _ctrl_slice = ctrl_values
        .as_slice()
        .ok_or_else(|| "ctrl_values must be contiguous".to_string())?;

    // In this case: ranksum is 0 by def, tie_sum is just the contribution of control values, and zpos is tracked normally
    // Note: this algo was made simpler as all control values are unique by definition,
    // but it's fundamentally the same as the exhaust loop done in the linear merge
    if n_pert == 0 {
        if zero_values_offset > 0 {
            let zero_index = if ctrl_values[0] > zero_val {
                0
            } else {
                fsearchsorted_left(_ctrl_slice, zero_val)
            };
            zero_pos = Some(offsets[zero_index]);
        } else {
            zero_pos = Some(0);
        }
        return Ok((0., tie_sum, zero_pos.unwrap()));
    }
    // Prepare vars for block comparison
    let mut val = pert_values[0];
    let mut count: usize = 0;
    let mut lo: usize = partial_left_binsearch(ctrl_values, val, 0, Some(n_ctrl))?;
    // let mut lo: usize;
    let mut prev_lo: usize = lo;
    let mut block_size: usize;

    for i in 0..n_pert {
        let x = pert_values[i];

        // Get position of x in the sorted control values
        // lo = partial_left_binsearch(ctrl_values, x, lo, Some(n_ctrl))?;
        // lo = fsearchsorted_left(&ctrl_slice[lo..n_ctrl], x) + lo;

        // Record position of the zeros in the sorted array if needed
        if (x > zero_val) && (zero_pos.is_none()) && (zero_values_offset > 0) {
            if ctrl_values[0] < zero_val {
                let lo_zero = fsearchsorted_left(_ctrl_slice, zero_val);
                zero_pos = Some(offsets[lo_zero] + i);
            } else {
                zero_pos = Some(offsets[0] + i);
            }
        }

        // Now check block condition
        if x != val {
            lo = partial_left_binsearch(ctrl_values, x, lo, Some(n_ctrl))?;
            // lo = fsearchsorted_left(&ctrl_slice[lo..n_ctrl], x) + lo;
            if (prev_lo < n_ctrl) && (ctrl_values[prev_lo] == val) {
                let ctrl_cnt = offsets[prev_lo + 1] - offsets[prev_lo];
                block_size = count + ctrl_cnt;
                tie_sum += tie_sum_delta(ctrl_cnt, count)?;
            } else {
                if count > 1 {
                    tie_sum += count.pow(3) - count;
                };
                block_size = count;
            }

            let mut first_rank = offsets[prev_lo] + (i - count);
            // If zero has been crossed, the first rank is offset
            if (zero_pos.is_some()) && (val > zero_val) {
                first_rank += zero_values_offset;
            }
            rank_sum += count as f64 * (first_rank as f64 + (block_size as f64 + 1.) / 2.);

            // Reset the block vars
            count = 1;
            val = x;
        } else {
            count += 1;
        }

        prev_lo = lo;
    }

    // Finalize the last block
    if (prev_lo < n_ctrl) && (ctrl_values[prev_lo] == val) {
        let ctrl_cnt = offsets[prev_lo + 1] - offsets[prev_lo];
        block_size = count + ctrl_cnt;
        tie_sum += tie_sum_delta(ctrl_cnt, count)?;
    } else {
        if count > 1 {
            tie_sum += count.pow(3) - count;
        };
        block_size = count;
    }
    let mut first_rank = offsets[prev_lo] + (n_pert - count);
    // If zero has been crossed, the first rank is offset
    if (zero_pos.is_some()) && (val > zero_val) {
        first_rank += zero_values_offset;
    }
    rank_sum += count as f64 * (first_rank as f64 + (block_size as f64 + 1.) / 2.);

    // Take care that if zero_pos has not been set yet (no positive value seen in the loop)
    // then the zeros will sit at the end
    if zero_pos.is_none() && zero_values_offset > 0 {
        let zero_index = if ctrl_values[0] > zero_val {
            0
        } else {
            fsearchsorted_left(_ctrl_slice, zero_val)
        };
        zero_pos = Some(n_pert + offsets[zero_index]);
    }

    // If zero_pos has been set nowhere, it's just that we are in the dense case where no value is omitted
    Ok((rank_sum, tie_sum, zero_pos.unwrap_or(0)))
}
