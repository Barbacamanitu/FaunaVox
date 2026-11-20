//! Voice space: a position on 10 perceptual axes (plus a small offset per setting), relative to what
//! the body predicts, mapped to a complete creature for the synth. See GENOME_VOICE_DESIGN.md §2.
//!
//! The axes are what the genome walk moves and what ratings are fitted over. Position 0 everywhere
//! is an ordinary voice for this body, so the same coordinates mean the same thing on a mouse and a
//! bear. Every position maps to a valid creature: each setting goes through a soft-limited curve
//! around its body-predicted value.

use serde::{Deserialize, Serialize};

use crate::model::{Body, Creature, Source, Voice, VowelSpec};
use crate::rng::Rng;

/// Bump when the axes, settings or curves change: ratings keep the version they were made under.
pub const SPACE_VERSION: u32 = 1;

pub const AXES: [&str; 10] = [
    "resonance_scale",  // bigger or smaller than it is: tract length, larynx lowering
    "pitch_height",     // higher, apart from size
    "pitch_liveliness", // monotone -> expressive, yelpy
    "breathy",          // pressed and buzzy -> soft and airy
    "roughness",        // smooth -> gravelly
    "instability",      // clean -> subharmonics, chaos, two tones at once
    "brightness",       // dark and muffled -> bright and thin
    "ringiness",        // dull -> clear, ringing vowels
    "nasality",         // oral -> through the nose
    "body_resonance",   // no thump -> chest thump or air-sac boom
];

/// How a setting's value follows its raw coordinate: `s = tanh(raw / 2)` runs from the centre
/// toward `high` (s > 0) or `low` (s < 0), on a linear or log scale.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scale {
    Lin,
    Log,
}

/// The walked settings, in order. `residual` vectors use this order too.
pub const SETTINGS: [&str; 29] = [
    "tract_length", "larynx_drop", "fold_scale", "pitch", "falsetto", "pitch_range", "vibrato", "jumps",
    "breathiness", "breath_pulse", "open_quotient", "buzz", "roughness", "harshness", "subharmonics", "chaos",
    "biphonation", "biphonation_ratio", "hf_loss", "noise_color", "low_cut", "formant_gain", "formant_width",
    "tract_damping", "nasality", "chest", "chest_width", "resonator_freq", "resonator_harmonic",
];

fn idx(name: &str) -> usize {
    SETTINGS.iter().position(|s| *s == name).expect("setting")
}

/// Each axis as weights on the settings: (setting, weight). Positive moves toward the axis' name.
const AXIS_WEIGHTS: [&[(&str, f64)]; 10] = [
    &[("tract_length", 1.0), ("larynx_drop", 0.6)],
    &[("pitch", 1.0), ("fold_scale", -0.3), ("falsetto", 0.4)],
    &[("pitch_range", 1.0), ("vibrato", 0.6), ("jumps", 0.5)],
    &[("breathiness", 1.0), ("open_quotient", 0.8), ("breath_pulse", -0.3), ("buzz", -0.6)],
    &[("roughness", 1.0), ("harshness", 0.8)],
    &[("subharmonics", 1.0), ("chaos", 0.8), ("biphonation", 0.8)],
    &[("hf_loss", -1.0), ("noise_color", 0.6), ("low_cut", 0.5)],
    &[("formant_gain", 1.0), ("formant_width", -0.8), ("tract_damping", -0.6)],
    &[("nasality", 1.0)],
    &[("chest", 1.0), ("chest_width", -0.3), ("resonator_harmonic", 0.8), ("resonator_freq", -0.4)],
];

/// What Unity tells the DLL about a creature's body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BodyInput {
    pub mass: f64,
    /// The tract from the body's measurements; when absent, from head and neck, or else from mass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tract_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neck_length: Option<f64>,
    pub source: Source,
    /// For display and anatomy: "quadruped", "bird", "bug" or "serpent".
    pub shape: String,
    pub lips: bool,
    pub teeth: bool,
}

impl Default for BodyInput {
    fn default() -> Self {
        BodyInput {
            mass: 50.0, tract_length: None, head_length: None, neck_length: None, source: Source::Larynx,
            shape: "quadruped".into(), lips: true, teeth: true,
        }
    }
}

impl BodyInput {
    pub fn tract(&self) -> f64 {
        Body { mass: self.mass, tract_length: self.tract_length, head_length: self.head_length, neck_length: self.neck_length, ..Body::default() }
            .tract()
    }
}

/// (centre, low, high, scale) for each setting on this body: the body prior and how far the walk
/// can take it.
fn ranges(b: &BodyInput) -> [(f64, f64, f64, Scale); 29] {
    use Scale::*;
    let tract = b.tract();
    let syrinx = b.source == Source::Syrinx;
    let pulse = b.source == Source::Pulse;
    // A syrinx's natural pitch from mass alone runs very high (a crow's preset sits 30 semitones
    // under it), so its centre is lower and its range reaches further down.
    let pitch = if syrinx { (-12.0, -40.0, 8.0) } else { (0.0, -12.0, 12.0) };
    let damping = if syrinx { 4.0 } else { 1.0 };
    let carrier = 400.0 * b.mass.powf(-0.3);
    [
        (tract, tract / 1.6, tract * 1.6, Log),                // tract_length
        (0.15, 0.0, 0.6, Lin),                                 // larynx_drop
        (1.0, 0.5, 2.0, Log),                                  // fold_scale
        (pitch.0, pitch.1, pitch.2, Lin),                      // pitch (semitones)
        (0.05, 0.0, 1.0, Lin),                                 // falsetto
        (if pulse { 0.5 } else { 1.0 }, 0.3, 2.5, Log),        // pitch_range
        (0.0, -0.4, 1.2, Lin),                                 // vibrato
        (0.05, 0.0, 1.0, Lin),                                 // jumps
        (0.25, 0.0, 0.9, Lin),                                 // breathiness
        (crate::synth::BREATH_PULSE, 0.3, 1.0, Lin),           // breath_pulse
        (crate::synth::OPEN_QUOTIENT, 0.3, 0.75, Lin),         // open_quotient
        (if syrinx { 0.3 } else { 0.0 }, 0.0, 1.0, Lin),       // buzz
        (0.3, 0.0, 0.8, Lin),                                  // roughness
        (0.0, -0.2, 0.5, Lin),                                 // harshness
        (0.08, 0.0, 1.0, Lin),                                 // subharmonics
        (0.05, 0.0, 1.0, Lin),                                 // chaos
        (0.05, 0.0, 1.0, Lin),                                 // biphonation
        (1.45, 1.15, 2.3, Log),                                // biphonation_ratio
        (1.0, 0.3, 2.5, Log),                                  // hf_loss
        (1.0, 0.5, 2.5, Log),                                  // noise_color
        (20.0, 0.0, 150.0, Lin),                               // low_cut (Hz)
        (crate::synth::FORMANT_GAIN, 0.0, 16.0, Lin),          // formant_gain (dB)
        (1.0, 0.5, 2.0, Log),                                  // formant_width
        (damping, damping * 0.5, damping * 2.0, Log),          // tract_damping
        (0.1, 0.0, 1.0, Lin),                                  // nasality
        (1.0, 0.0, 8.0, Lin),                                  // chest: its resonance is weak below ~3
        (70.0, 35.0, 200.0, Log),                              // chest_width (Hz)
        (carrier, carrier * 0.5, carrier * 2.0, Log),          // resonator_freq (Hz)
        (0.2, 0.0, 1.5, Lin),                                  // resonator_harmonic
    ]
}

/// The raw coordinate of each setting at a position.
pub fn raw(axes: &[f64; 10], residual: &[f64; 29]) -> [f64; 29] {
    let mut r = *residual;
    for (a, weights) in axes.iter().zip(AXIS_WEIGHTS.iter()) {
        for &(name, w) in weights.iter() {
            r[idx(name)] += w * a;
        }
    }
    r
}

/// Every setting's value at a position, on this body.
pub fn settings(b: &BodyInput, axes: &[f64; 10], residual: &[f64; 29]) -> [f64; 29] {
    let raw = raw(axes, residual);
    let ranges = ranges(b);
    std::array::from_fn(|k| {
        let (c, lo, hi, scale) = ranges[k];
        let s = (raw[k] / 2.0).tanh();
        match scale {
            Scale::Lin => if s >= 0.0 { c + (hi - c) * s } else { c + (c - lo) * s },
            Scale::Log => if s >= 0.0 { c * (hi / c).powf(s) } else { c * (c / lo).powf(s) },
        }
    })
}

/// The creature at a position on this body, with the given vowels.
pub fn creature(b: &BodyInput, axes: &[f64; 10], residual: &[f64; 29], vowels: Vec<VowelSpec>) -> Creature {
    let v = settings(b, axes, residual);
    let g = |name: &str| v[idx(name)];
    let mut body = Body {
        mass: b.mass,
        tract_length: Some(g("tract_length")),
        source: b.source,
        fold_scale: g("fold_scale"),
        larynx_drop: g("larynx_drop"),
        roughness: g("roughness"),
        breathiness: g("breathiness"),
        nasality: g("nasality"),
        lung_capacity: 1.0,
        resonator_freq: Some(g("resonator_freq")),
        resonator_harmonic: g("resonator_harmonic"),
        buzz: g("buzz"),
        chest: g("chest"),
        chest_width: g("chest_width"),
        noise_color: g("noise_color"),
        tract_damping: Some(g("tract_damping")),
        hf_loss: g("hf_loss"),
        low_cut: g("low_cut"),
        formant_gain: g("formant_gain"),
        formant_width: g("formant_width"),
        breath_pulse: g("breath_pulse"),
        open_quotient: g("open_quotient"),
        head_length: None,
        neck_length: None,
    };
    if b.source == Source::Hiss {
        body.breathiness = 1.0; // all breath: a hiss has no voice to be breathy against
    }
    let voice = Voice {
        vowels,
        pitch: g("pitch"),
        pitch_range: g("pitch_range"),
        harshness: g("harshness"),
        vibrato: g("vibrato"),
        falsetto: g("falsetto"),
        subharmonics: g("subharmonics"),
        chaos: g("chaos"),
        biphonation: g("biphonation"),
        biphonation_ratio: g("biphonation_ratio"),
        jumps: g("jumps"),
        ..Voice::default()
    };
    Creature { name: None, body, voice }
}

// ------------------------------------------------------------------ sampling

/// The kinds of body the game spawns, by sound source: (source, shape, share of samples, mass
/// range in kg, has lips, has teeth). Unity picks the source from wings and legs (VocalProfile).
pub const BODY_KINDS: [(Source, &str, f64, f64, f64, bool, bool); 4] = [
    (Source::Larynx, "quadruped", 0.55, 0.05, 800.0, true, true),
    (Source::Syrinx, "bird", 0.2, 0.02, 20.0, false, false),
    (Source::Pulse, "bug", 0.15, 0.001, 10.0, false, false),
    (Source::Hiss, "serpent", 0.1, 0.3, 100.0, false, true),
];

/// A random body: a kind by its share, a mass even on a log scale over its range, and a tract that
/// varies around the mass-predicted one as head and neck shapes do (Unity clamps it to 0.4-3x).
pub fn sample_body(rng: &mut Rng) -> BodyInput {
    let total: f64 = BODY_KINDS.iter().map(|k| k.2).sum();
    let mut pick = rng.random() * total;
    let mut kind = BODY_KINDS[0];
    for k in BODY_KINDS {
        if pick < k.2 {
            kind = k;
            break;
        }
        pick -= k.2;
    }
    let mass = (kind.3.ln() + rng.random() * (kind.4.ln() - kind.3.ln())).exp();
    body_of(kind, mass, (0.25 * rng.standard_normal()).exp().clamp(0.5, 2.0))
}

/// A body of this kind and mass, its tract this multiple of the mass-predicted one.
pub fn body_of(kind: (Source, &str, f64, f64, f64, bool, bool), mass: f64, tract_factor: f64) -> BodyInput {
    BodyInput {
        mass,
        tract_length: Some(0.06 * mass.cbrt() * tract_factor),
        head_length: None,
        neck_length: None,
        source: kind.0,
        shape: kind.1.into(),
        lips: kind.5,
        teeth: kind.6,
    }
}

/// A random position from the sampling distribution: axes Student-t (4 degrees of freedom, scale 1),
/// mostly near the middle with some far out, as a genome walk spreads; residuals normal, scale 0.1.
pub fn sample_position(rng: &mut Rng) -> ([f64; 10], [f64; 29]) {
    let t = |rng: &mut Rng| {
        let z = rng.standard_normal();
        let chi2: f64 = (0..4).map(|_| rng.standard_normal().powi(2)).sum();
        z / (chi2 / 4.0).sqrt()
    };
    let axes = std::array::from_fn(|_| t(rng));
    let residual = std::array::from_fn(|_| 0.1 * rng.standard_normal());
    (axes, residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_the_body_prior() {
        let b = BodyInput::default();
        let v = settings(&b, &[0.0; 10], &[0.0; 29]);
        assert!((v[idx("tract_length")] - b.tract()).abs() < 1e-12);
        assert!((v[idx("roughness")] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn every_position_is_valid() {
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let b = sample_body(&mut rng);
            let (mut a, r) = sample_position(&mut rng);
            a.iter_mut().for_each(|x| *x *= 4.0); // far out too
            let c = creature(&b, &a, &r, vec![VowelSpec::Name("a".into())]);
            c.body.validate().unwrap();
            c.voice.validate().unwrap();
            let s = settings(&b, &a, &r);
            let rg = ranges(&b);
            for k in 0..29 {
                assert!(s[k] >= rg[k].1 - 1e-9 && s[k] <= rg[k].2 + 1e-9, "{} out of range", SETTINGS[k]);
            }
        }
    }
}
