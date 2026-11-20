//! The preference field: the Gaussian process fitted to your ratings (ratings/fit.py), evaluated
//! natively so the genome walk can steer around voices you dislike. See GENOME_VOICE_DESIGN.md §3.
//!
//! The fitted model is compiled in from data/field_vN.json. Only `badness` matters to the walk:
//! it's 0 wherever the model isn't confident a voice is disliked.

use std::sync::OnceLock;

use serde::Deserialize;

use crate::model::Source;
use crate::space::AXES;

#[derive(Deserialize)]
struct FieldJson {
    field_version: u32,
    space_version: Vec<u32>,
    features: Vec<String>,
    length_scale: Vec<f64>,
    signal_variance: f64,
    noise: f64,
    y_mean: f64,
    y_std: f64,
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    #[serde(rename = "L")]
    l: Vec<Vec<f64>>,
    badness: BadnessJson,
}

#[derive(Deserialize)]
struct BadnessJson {
    theta: f64,
    kappa: f64,
    beta: f64,
}

pub struct Field {
    pub version: u32,
    pub space_versions: Vec<u32>,
    pub features: Vec<String>,
    inv_ls2: Vec<f64>,
    c: f64,
    noise: f64,
    y_mean: f64,
    y_std: f64,
    n: usize,
    d: usize,
    /// Training inputs, n x d, row-major.
    x: Vec<f64>,
    alpha: Vec<f64>,
    /// Cholesky factor of the training covariance, n x n, row-major lower triangle.
    l: Vec<f64>,
    pub theta: f64,
    pub kappa: f64,
    pub beta: f64,
    /// Feature index of each voice-space axis.
    axis_index: [usize; 10],
}

const BUILTIN: &str = include_str!("../data/field_v1.json");

/// The field compiled into this build.
pub fn builtin() -> &'static Field {
    static F: OnceLock<Field> = OnceLock::new();
    F.get_or_init(|| Field::from_json(BUILTIN).expect("built-in field"))
}

impl Field {
    pub fn from_json(json: &str) -> Result<Field, String> {
        let f: FieldJson = serde_json::from_str(json).map_err(|e| format!("field JSON: {e}"))?;
        let n = f.x.len();
        let d = f.features.len();
        if f.length_scale.len() != d || f.alpha.len() != n || f.l.len() != n || f.x.iter().any(|r| r.len() != d) {
            return Err("field JSON: inconsistent sizes".into());
        }
        let mut axis_index = [0; 10];
        for (k, name) in AXES.iter().enumerate() {
            axis_index[k] = f.features.iter().position(|s| s == name).ok_or_else(|| format!("field has no axis {name}"))?;
        }
        Ok(Field {
            version: f.field_version,
            space_versions: f.space_version,
            inv_ls2: f.length_scale.iter().map(|l| 1.0 / (l * l)).collect(),
            c: f.signal_variance,
            noise: f.noise,
            y_mean: f.y_mean,
            y_std: f.y_std,
            n,
            d,
            x: f.x.into_iter().flatten().collect(),
            alpha: f.alpha,
            l: f.l.into_iter().flatten().collect(),
            theta: f.badness.theta,
            kappa: f.badness.kappa,
            beta: f.badness.beta,
            axis_index,
            features: f.features,
        })
    }

    /// The model's input for a voice: axes, log10 mass, source and intent one-hots. An intent the
    /// field wasn't fitted on (or none, for the trunk) counts as an even mix of those it was.
    pub fn input(&self, axes: &[f64; 10], mass: f64, source: Source, intent: Option<&str>) -> Vec<f64> {
        let src = match source {
            Source::Larynx => "larynx",
            Source::Syrinx => "syrinx",
            Source::Pulse => "pulse",
            Source::Hiss => "hiss",
        };
        let intents: Vec<usize> = (0..self.d).filter(|&j| self.features[j].starts_with("intent=")).collect();
        let known = intent.and_then(|i| intents.iter().copied().find(|&j| self.features[j][7..] == *i));
        let mut x = vec![0.0; self.d];
        for (k, &j) in self.axis_index.iter().enumerate() {
            x[j] = axes[k];
        }
        for j in 0..self.d {
            let name = &self.features[j];
            if name == "log_mass" {
                x[j] = mass.log10();
            } else if let Some(s) = name.strip_prefix("source=") {
                x[j] = if s == src { 1.0 } else { 0.0 };
            } else if name.starts_with("intent=") {
                x[j] = match known {
                    Some(k) => if k == j { 1.0 } else { 0.0 },
                    None => 1.0 / intents.len() as f64,
                };
            }
        }
        x
    }

    /// Predicted score and its uncertainty at an input.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let k = self.kernel_row(x);
        let m: f64 = k.iter().zip(&self.alpha).map(|(a, b)| a * b).sum();
        (self.y_mean + self.y_std * m, self.sd(&k))
    }

    fn kernel_row(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let row = &self.x[i * self.d..(i + 1) * self.d];
                let q: f64 = row.iter().zip(x).zip(&self.inv_ls2).map(|((r, v), w)| (v - r) * (v - r) * w).sum();
                self.c * (-0.5 * q).exp()
            })
            .collect()
    }

    fn sd(&self, k: &[f64]) -> f64 {
        // v = L^-1 k; var = k(x,x) + noise - v.v (as sklearn's predict, noise included)
        let mut v = vec![0.0; self.n];
        let mut vv = 0.0;
        for i in 0..self.n {
            let row = &self.l[i * self.n..i * self.n + i];
            let s: f64 = row.iter().zip(&v[..i]).map(|(a, b)| a * b).sum();
            v[i] = (k[i] - s) / self.l[i * self.n + i];
            vv += v[i] * v[i];
        }
        self.y_std * (self.c + self.noise - vv).max(1e-12).sqrt()
    }

    /// badness(x) = softplus(β(−μ − κσ − θ))/β, and its gradient over the voice-space axes. The
    /// gradient treats σ as fixed: it follows the predicted score downhill.
    pub fn badness(&self, x: &[f64]) -> (f64, [f64; 10]) {
        let k = self.kernel_row(x);
        let m: f64 = k.iter().zip(&self.alpha).map(|(a, b)| a * b).sum();
        let mu = self.y_mean + self.y_std * m;
        // σ never drops below the noise floor, y_std·√noise: skip the costly part where even that
        // leaves badness under 1e-3, below where the walk deflects at all.
        let sd_floor = self.y_std * self.noise.max(0.0).sqrt();
        if self.beta * (-mu - self.kappa * sd_floor - self.theta) < -5.0 {
            return (0.0, [0.0; 10]);
        }
        let z = self.beta * (-mu - self.kappa * self.sd(&k) - self.theta);
        let bad = softplus(z) / self.beta;
        let sig = 1.0 / (1.0 + (-z).exp());
        let mut g = [0.0; 10];
        for (a, &j) in self.axis_index.iter().enumerate() {
            let mut dmu = 0.0;
            for i in 0..self.n {
                dmu += self.alpha[i] * k[i] * -(x[j] - self.x[i * self.d + j]) * self.inv_ls2[j];
            }
            g[a] = -sig * self.y_std * dmu;
        }
        (bad, g)
    }

    /// A probe for one body and intent: badness at any axis position, fast. The training points
    /// whose mass, source and intent put them too far away to matter (kernel below e^-15) are left
    /// out of every evaluation.
    pub fn probe(&self, mass: f64, source: Source, intent: Option<&str>) -> Probe<'_> {
        let x = self.input(&[0.0; 10], mass, source, intent);
        let is_axis: Vec<bool> = (0..self.d).map(|j| self.axis_index.contains(&j)).collect();
        let near = (0..self.n)
            .filter_map(|i| {
                let q0: f64 = (0..self.d)
                    .filter(|&j| !is_axis[j])
                    .map(|j| (x[j] - self.x[i * self.d + j]).powi(2) * self.inv_ls2[j])
                    .sum();
                (q0 < 30.0).then_some((i, q0))
            })
            .collect();
        Probe { f: self, x, near }
    }
}

/// Badness for one body and intent; see Field::probe.
pub struct Probe<'a> {
    f: &'a Field,
    x: Vec<f64>,
    near: Vec<(usize, f64)>,
}

impl Probe<'_> {
    /// As Field::badness, at this axis position.
    pub fn badness(&mut self, axes: &[f64; 10]) -> (f64, [f64; 10]) {
        let f = self.f;
        for (k, &j) in f.axis_index.iter().enumerate() {
            self.x[j] = axes[k];
        }
        let mut k = vec![0.0; f.n];
        let mut m = 0.0;
        for &(i, q0) in &self.near {
            let row = &f.x[i * f.d..(i + 1) * f.d];
            let q: f64 = f.axis_index.iter().map(|&j| (self.x[j] - row[j]).powi(2) * f.inv_ls2[j]).sum();
            k[i] = f.c * (-0.5 * (q0 + q)).exp();
            m += k[i] * f.alpha[i];
        }
        let mu = f.y_mean + f.y_std * m;
        let sd_floor = f.y_std * f.noise.max(0.0).sqrt();
        if f.beta * (-mu - f.kappa * sd_floor - f.theta) < -5.0 {
            return (0.0, [0.0; 10]);
        }
        let z = f.beta * (-mu - f.kappa * f.sd(&k) - f.theta);
        let bad = softplus(z) / f.beta;
        let sig = 1.0 / (1.0 + (-z).exp());
        let mut g = [0.0; 10];
        for (a, &j) in f.axis_index.iter().enumerate() {
            let dmu: f64 = self.near.iter().map(|&(i, _)| f.alpha[i] * k[i] * -(self.x[j] - f.x[i * f.d + j]) * f.inv_ls2[j]).sum();
            g[a] = -sig * f.y_std * dmu;
        }
        (bad, g)
    }
}

fn softplus(z: f64) -> f64 {
    if z > 30.0 { z } else { z.exp().ln_1p() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_loads_and_predicts() {
        let f = builtin();
        let x = f.input(&[0.0; 10], 5.0, Source::Larynx, Some("idle"));
        let (mu, sd) = f.predict(&x);
        // The same formula in numpy on data/field_v1.json gives 0.419916, 0.297339.
        assert!((mu - 0.419916).abs() < 1e-5 && (sd - 0.297339).abs() < 1e-5, "{mu} {sd}");
        let (b, g) = f.badness(&x);
        assert!(b >= 0.0 && g.iter().all(|v| v.is_finite()));
    }
}
