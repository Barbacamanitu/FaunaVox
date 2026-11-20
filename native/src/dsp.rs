//! Filters and helpers matching the scipy/numpy calls used by the Python model.

use std::f64::consts::PI;

use crate::synth::SR;

const SRF: f64 = SR as f64;

/// numpy.interp: linear interpolation, clamped to the end values.
pub fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // Curves have a handful of points; a linear scan beats a binary search here.
    let mut k = 1;
    while k < n - 1 && xs[k] < x {
        k += 1;
    }
    let (x0, x1) = (xs[k - 1], xs[k]);
    if x1 <= x0 {
        return ys[k];
    }
    ys[k - 1] + (ys[k] - ys[k - 1]) * (x - x0) / (x1 - x0)
}

/// Evaluate a (u, value) curve at u.
pub fn curve_at(points: &[(f64, f64)], u: f64) -> f64 {
    let n = points.len();
    if n == 0 {
        return 0.0;
    }
    if u <= points[0].0 {
        return points[0].1;
    }
    if u >= points[n - 1].0 {
        return points[n - 1].1;
    }
    let mut k = 1;
    while k < n - 1 && points[k].0 < u {
        k += 1;
    }
    let (x0, y0) = points[k - 1];
    let (x1, y1) = points[k];
    if x1 <= x0 {
        return y1;
    }
    y0 + (y1 - y0) * (u - x0) / (x1 - x0)
}

pub fn curve(points: &[(f64, f64)], u: &[f64]) -> Vec<f64> {
    u.iter().map(|&x| curve_at(points, x)).collect()
}

pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 1e-12;
    }
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt() + 1e-12
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Population standard deviation (numpy default).
pub fn std(x: &[f64]) -> f64 {
    let m = mean(x);
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

pub fn peak(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

// ------------------------------------------------------------------ biquads

/// One second-order section, a0 normalized to 1 (scipy convention: y = b.x - a1 y1 - a2 y2).
#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Biquad {
    /// Filter in place (transposed direct form II, zero initial state like scipy).
    pub fn process(&self, x: &mut [f64]) {
        let (mut z1, mut z2) = (0.0, 0.0);
        for v in x.iter_mut() {
            let input = *v;
            let out = self.b0 * input + z1;
            z1 = self.b1 * input - self.a1 * out + z2;
            z2 = self.b2 * input - self.a2 * out;
            *v = out;
        }
    }
}

pub fn sosfilt(sos: &[Biquad], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for s in sos {
        s.process(&mut y);
    }
    y
}

fn warp(fc: f64) -> f64 {
    // Bilinear transform with prewarping, as scipy.signal.butter does.
    (PI * (fc / SRF).min(0.499_999)).tan()
}

/// scipy butter(order, fc, "lowpass"), order 1 or 2.
pub fn butter_lowpass(order: u32, fc: f64) -> Biquad {
    let k = warp(fc);
    if order == 1 {
        let n = 1.0 / (1.0 + k);
        Biquad { b0: k * n, b1: k * n, b2: 0.0, a1: (k - 1.0) * n, a2: 0.0 }
    } else {
        let q = std::f64::consts::SQRT_2;
        let n = 1.0 / (1.0 + q * k + k * k);
        let b0 = k * k * n;
        Biquad { b0, b1: 2.0 * b0, b2: b0, a1: 2.0 * (k * k - 1.0) * n, a2: (1.0 - q * k + k * k) * n }
    }
}

/// scipy butter(order, fc, "highpass"), order 1 or 2.
pub fn butter_highpass(order: u32, fc: f64) -> Biquad {
    let k = warp(fc);
    if order == 1 {
        let n = 1.0 / (1.0 + k);
        Biquad { b0: n, b1: -n, b2: 0.0, a1: (k - 1.0) * n, a2: 0.0 }
    } else {
        let q = std::f64::consts::SQRT_2;
        let n = 1.0 / (1.0 + q * k + k * k);
        Biquad { b0: n, b1: -2.0 * n, b2: n, a1: 2.0 * (k * k - 1.0) * n, a2: (1.0 - q * k + k * k) * n }
    }
}

/// Four-pole bandpass (a 2nd-order highpass and lowpass in series). Used only to colour noise
/// that is renormalized afterwards, so it stands in for scipy's butter(2, [lo, hi], "bandpass").
pub fn bandpass(lo: f64, hi: f64) -> [Biquad; 2] {
    [butter_highpass(2, lo), butter_lowpass(2, hi)]
}

/// scipy iirnotch(f0, Q).
pub fn notch(f0: f64, q: f64) -> Biquad {
    let w0 = 2.0 * PI * f0 / SRF;
    let bw = w0 / q;
    let beta = (bw / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let c = w0.cos();
    Biquad { b0: gain, b1: -2.0 * gain * c, b2: gain, a1: -2.0 * gain * c, a2: 2.0 * gain - 1.0 }
}

/// Klatt-style two-pole resonator with unity gain at DC.
pub fn resonator(freq: f64, bw: f64) -> Biquad {
    let r = (-PI * bw / SRF).exp();
    let b1 = 2.0 * r * (2.0 * PI * freq / SRF).cos();
    let b2 = -r * r;
    Biquad { b0: 1.0 - b1 - b2, b1: 0.0, b2: 0.0, a1: -b1, a2: -b2 }
}

// ------------------------------------------------------------------ FFT convolution

fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..half {
                let (a, b) = (i + k, i + k + half);
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let s = 1.0 / n as f64;
        re.iter_mut().for_each(|v| *v *= s);
        im.iter_mut().for_each(|v| *v *= s);
    }
}

/// Full linear convolution (scipy fftconvolve), length a.len() + b.len() - 1.
pub fn fft_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let n = out_len.next_power_of_two();
    // Pack both real signals into one complex FFT: z = a + i b.
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    re[..a.len()].copy_from_slice(a);
    im[..b.len()].copy_from_slice(b);
    fft(&mut re, &mut im, false);
    // A(k) = (Z(k) + conj Z(n-k)) / 2, B(k) = (Z(k) - conj Z(n-k)) / 2i; product A*B.
    let mut pr = vec![0.0; n];
    let mut pi = vec![0.0; n];
    for k in 0..n {
        let m = (n - k) % n;
        let (zr, zi) = (re[k], im[k]);
        let (cr, ci) = (re[m], -im[m]);
        let (ar, ai) = (0.5 * (zr + cr), 0.5 * (zi + ci));
        let (br, bi) = (0.5 * (zi - ci), -0.5 * (zr - cr));
        pr[k] = ar * br - ai * bi;
        pi[k] = ar * bi + ai * br;
    }
    fft(&mut pr, &mut pi, true);
    pr.truncate(out_len);
    pr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolve_matches_direct() {
        let a = [1.0, 2.0, 3.0, -1.0];
        let b = [0.5, -1.0, 2.0];
        let c = fft_convolve(&a, &b);
        let mut d = vec![0.0; a.len() + b.len() - 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                d[i + j] += a[i] * b[j];
            }
        }
        for (x, y) in c.iter().zip(&d) {
            assert!((x - y).abs() < 1e-9, "{c:?} vs {d:?}");
        }
    }

    #[test]
    fn lowpass_unity_dc() {
        let mut x = vec![1.0; 20000];
        butter_lowpass(2, 500.0).process(&mut x);
        assert!((x[19999] - 1.0).abs() < 1e-6);
        let mut x = vec![1.0; 20000];
        butter_highpass(2, 500.0).process(&mut x);
        assert!(x[19999].abs() < 1e-6);
    }
}
