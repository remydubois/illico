use libm::erfc;

pub fn compute_pvalue(
    n_ref: f64,
    n_tgt: f64,
    n: f64,
    tie_sum: f64,
    u: f64,
    mu: f64,
    contin_corr: f64,
    alternative: &String,
) -> Result<(f64, f64), String> {
    let tie_corr: f64 = 1.0 - tie_sum / (n * (n - 1.) * (n + 1.));
    if tie_corr > 1e-9 {
        let sigma: f64 = (n_ref * n_tgt * (n_ref + n_tgt + 1.) / 12.0 * tie_corr).powf(0.5);

        match alternative.as_str() {
            "two-sided" => {
                let delta = u - mu;
                let z = (delta - delta.signum() * contin_corr) / sigma;
                return Ok((erfc(z.abs() / (2.0 as f64).sqrt()), z));
            }
            "greater" => {
                let delta = u - mu;
                let z = (delta - contin_corr) / sigma;
                return Ok((0.5 * erfc(z / (2.0 as f64).sqrt()), z));
            }
            "less" => {
                let delta = u - mu;
                let z = (delta + contin_corr) / sigma;
                return Ok((0.5 * erfc(-z / (2.0 as f64).sqrt()), z));
            }
            _ => Err(format!("Invalid alternative: received {alternative}.")),
        }
    } else {
        return Ok((1.0, 0.));
    }
}
