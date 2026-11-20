//! Small deterministic RNG (xoshiro256++ seeded through SplitMix64).
//!
//! The same seed always gives the same sound on every platform. It does not reproduce numpy's
//! PCG64 stream, so a seed renders a different (but statistically equivalent) take than Python.

#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Rng { s: [splitmix(&mut x), splitmix(&mut x), splitmix(&mut x), splitmix(&mut x)] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }

    /// Integer in [lo, hi] inclusive.
    pub fn integer_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }

    pub fn standard_normal(&mut self) -> f64 {
        // Box-Muller; 1 - random() is in (0, 1] so the log is finite.
        let u1 = 1.0 - self.random();
        let u2 = self.random();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.standard_normal()
    }

    /// Exponential with the given mean.
    pub fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.random()).ln()
    }

    /// Poisson count (Knuth; fine for the small means used here).
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        if !(lambda > 0.0) {
            return 0;
        }
        let limit = (-lambda.min(30.0)).exp();
        let (mut k, mut p) = (0u32, 1.0);
        loop {
            p *= self.random();
            if p <= limit {
                return k;
            }
            k += 1;
        }
    }

    pub fn normals(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.standard_normal()).collect()
    }
}
