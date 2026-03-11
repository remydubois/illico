use libm::erfc;
use pyo3::{exceptions::PyValueError, prelude::*}; // use libm to match numba and scipy

pub fn compute_pvalue(
    n_ref: f64,
    n_tgt: f64,
    n: f64,
    tie_sum: f64,
    U: f64,
    mu: f64,
    contin_corr: f64,
    alternative: &String,
) -> Result<f64, String> {
    let tie_corr: f64 = 1.0 - tie_sum / (n * (n - 1.) * (n + 1.));
    if tie_corr > 1e-9 {
        let sigma: f64 = (n_ref * n_tgt * (n_ref + n_tgt + 1.) / 12.0 * tie_corr).powf(0.5);

        match alternative.as_str() {
            "two-sided" => {
                let min_u = U.min(n_ref * n_tgt - U);
                let delta = min_u - mu;
                let z = (delta.abs() + delta.signum() * contin_corr) / sigma;
                return Ok(erfc(z / (2.0 as f64).sqrt()));
            }
            "greater" => {
                let delta = U - mu;
                let z = (delta - contin_corr) / sigma;
                return Ok(0.5 * erfc(z / (2.0 as f64).sqrt()));
            }
            "less" => {
                let delta = U - mu;
                let z = (delta + contin_corr) / sigma;
                return Ok(0.5 * erfc(-z / (2.0 as f64).sqrt()));
            }
            _ => Err(format!("Invalid alternative: received {alternative}.")),
        }
    } else {
        return Ok(1.0 as f64);
    }
}

#[pyfunction]
pub fn compute_pvalue_rust(
    n_ref: usize,
    n_tgt: usize,
    n: usize,
    tie_sum: f64,
    U: f64,
    mu: f64,
    contin_corr: f64,
    alternative: String,
) -> PyResult<f64> {
    // let p_value: f64 = compute_pvalue(n_ref, n_tgt, n, tie_sum, U, mu, contin_corr, &alternative)?;
    compute_pvalue(
        n_ref as f64,
        n_tgt as f64,
        n as f64,
        tie_sum,
        U,
        mu,
        contin_corr,
        &alternative,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn erfc_rust(x: f64) -> f64 {
    erfc(x / (2.0 as f64).sqrt())
}
