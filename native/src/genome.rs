//! Genome → voice, decoder version 2: decodes the voice section of a creature's DNA into a creature
//! for the synth. See GENOME_VOICE_DESIGN.md §4.
//!
//! The voice section is a plain string of bases with no structure: no promoters, stops or genes.
//! It's read as overlapping 4-base windows. Each of the 256 possible windows has a fixed vector of
//! traits, and a trait is a weighted sum over the windows that read it, divided by the root of the
//! summed squared weights. So it's spread like a standard normal over random DNA, and any one
//! mutation changes only the few windows that overlap it:
//!
//! - **Base voice:** every window, in DNA order, as steps of a walk through voice space (steered
//!   around disliked regions by the preference field). It moves least per mutation.
//! - **Inventory:** the first quarter defines the creature's 2-6 syllable types: consonant, vowel,
//!   an optional glide to a second vowel, length, pitch step, pitch slope and level.
//! - **Phrases:** the rest is split into 8 stretches, for `speak` and the 7 calls. Each stretch
//!   gives its call a pattern over the inventory (one pick per syllable slot, read by position, so
//!   neighbours tend to repeat), a syllable count, a rhythm, and a small timbre offset walked from
//!   the base voice. A mutation in one stretch changes only that call.
//!
//! Decoding is deterministic: the same DNA, body and versions always give the same creature.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Serialize;

use crate::field::{self, Field, Probe};
use crate::model::{base_intent, Consonant, Creature, IntentOverride, VowelSpec, CONSONANTS, VOWELS};
use crate::rng::Rng;
use crate::space::{self, BodyInput, AXES, SETTINGS, SPACE_VERSION};

/// Bump when the tables, trait mappings or layout change: saves keep the version they decoded under.
pub const DECODER_VERSION: u32 = 2;

/// The voice section length the scales are tuned for. Any length decodes; longer means smaller
/// changes per mutation.
pub const SECTION_LENGTH: usize = 1600;

const WINDOW: usize = 4;
/// Spread of the base voice on each axis over random DNA, of its residuals, and of a call's timbre
/// offset from the base voice.
const BASE_SCALE: f64 = 0.9;
const RESIDUAL_SCALE: f64 = 0.1;
const CALL_SCALE: f64 = 0.3;
const AXIS_LIMIT: f64 = 6.0;

/// Layout, as shares of the section: the inventory, then 8 equal phrase stretches.
const INVENTORY_END: f64 = 0.25;
const MAX_TYPES: usize = 6;
/// Width of a syllable slot's bell, in slot widths: inventory types barely overlap, phrase slots
/// share a slot's width each side, so neighbours often pick the same type.
const TYPE_SD: f64 = 0.45;
const SLOT_SD: f64 = 0.6;
/// The most vowels one syllable glides through (the synth lengthens a syllable for each).
const MAX_GLIDE: usize = 3;

/// Deflection: the share of a step's downhill part removed, at most, and the badness at which it's
/// 63 % of that.
const LAMBDA_MAX: f64 = 0.9;
const B0: f64 = 0.3;

/// The intents a genome builds calls for.
pub const CALL_INTENTS: [&str; 7] = ["mating", "warn", "alarm", "idle", "flee", "pain", "beg"];
/// The phrases, in stretch order: (intent, fewest syllables, most, typical, spread). The count is
/// round(typical + spread x z), clamped. Mating and alarm reach their most only at z = 2.5 (about 1
/// creature in 200 at random), so long songs and long alarm series are there for selection to find.
const PHRASES: [(&str, usize, usize, f64, f64); 8] = [
    ("speak", 2, 8, 5.0, 2.0),
    ("mating", 2, 8, 4.0, 1.6),
    ("warn", 1, 3, 2.0, 2.0 / 3.0),
    ("alarm", 1, 5, 2.0, 1.2),
    ("idle", 1, 3, 2.0, 2.0 / 3.0),
    ("flee", 1, 4, 2.5, 1.0),
    ("pain", 1, 2, 1.5, 1.0 / 3.0),
    ("beg", 1, 4, 2.5, 1.0),
];

/// Which of each window's trait values each trait reads.
mod dim {
    pub const AXES: usize = 0; // 10
    pub const RESIDUAL: usize = 10; // 29
    pub const OFFSET: usize = 39; // 10
    pub const TYPE_COUNT: usize = 49;
    pub const F1: usize = 50;
    pub const F2: usize = 51;
    pub const GLIDE: usize = 52;
    pub const F1B: usize = 53;
    pub const F2B: usize = 54;
    pub const CONSTRICTION: usize = 55;
    pub const PLACE: usize = 56;
    pub const VOICING: usize = 57;
    pub const LENGTH: usize = 58;
    pub const PITCH: usize = 59;
    pub const SLOPE: usize = 60;
    pub const LEVEL: usize = 61;
    pub const COUNT: usize = 62;
    pub const PICK: usize = 63;
    pub const LINK: usize = 64;
    pub const GAP: usize = 65;
    pub const ACCEL: usize = 66;
    pub const REPEATS: usize = 67;
    pub const SWELL: usize = 68; // 5
    pub const N: usize = 73;
}

/// The window table: a fixed vector of trait values for each of the 256 windows, each trait
/// centred and scaled to unit variance over the windows.
fn table() -> &'static Vec<[f64; dim::N]> {
    static T: OnceLock<Vec<[f64; dim::N]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut rng = Rng::new(0x5EED_0F_70CE);
        let mut t: Vec<[f64; dim::N]> = (0..256).map(|_| std::array::from_fn(|_| rng.standard_normal())).collect();
        for d in 0..dim::N {
            let mean = t.iter().map(|r| r[d]).sum::<f64>() / 256.0;
            let sd = (t.iter().map(|r| (r[d] - mean).powi(2)).sum::<f64>() / 256.0).sqrt();
            t.iter_mut().for_each(|r| r[d] = (r[d] - mean) / sd);
        }
        t
    })
}

fn base_index(c: u8) -> Option<u16> {
    match c {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// The voice section as window codes, in order. Anything but A, C, G and T is skipped.
fn windows(dna: &str) -> Vec<u16> {
    let b: Vec<u16> = dna.bytes().filter_map(base_index).collect();
    if b.len() < WINDOW {
        return vec![];
    }
    (0..=b.len() - WINDOW).map(|i| b[i] * 64 + b[i + 1] * 16 + b[i + 2] * 4 + b[i + 3]).collect()
}

/// Weights over the windows: (window, weight).
type Weights = Vec<(usize, f64)>;

/// The windows whose position (as a share of the section) lies in [a, b), weighted evenly.
fn even(n: usize, a: f64, b: f64) -> Weights {
    (0..n).filter(|&i| (a..b).contains(&((i as f64 + 0.5) / n as f64))).map(|i| (i, 1.0)).collect()
}

/// The windows in [a, b), weighted by a bell around `centre` (shares of the section).
fn bell(n: usize, a: f64, b: f64, centre: f64, sd: f64) -> Weights {
    even(n, a, b)
        .into_iter()
        .map(|(i, _)| (i, (-0.5 * (((i as f64 + 0.5) / n as f64 - centre) / sd).powi(2)).exp()))
        .collect()
}

fn norm(w: &Weights) -> f64 {
    w.iter().map(|(_, x)| x * x).sum::<f64>().sqrt()
}

/// A trait: the weighted sum of its value over the windows, divided by the weights' norm. 0 when
/// there are no windows.
fn score(codes: &[u16], w: &Weights, d: usize) -> f64 {
    let n = norm(w);
    if n == 0.0 {
        return 0.0;
    }
    let t = table();
    w.iter().map(|&(i, x)| x * t[codes[i] as usize][d]).sum::<f64>() / n
}

/// The standard normal CDF (Abramowitz-Stegun 7.1.26, error < 1.5e-7).
fn phi(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    0.5 * (1.0 + if z < 0.0 { -y } else { y })
}

/// Which of `k` equally likely bands a standard-normal score falls in.
fn band(z: f64, k: usize) -> usize {
    ((phi(z) * k as f64) as usize).min(k - 1)
}

// ------------------------------------------------------------------ steering walk

struct Walker<'a> {
    body: &'a BodyInput,
    field: Option<&'a Field>,
    steps: usize,
    deflected: usize,
    max_badness: f64,
}

impl<'a> Walker<'a> {
    fn probe(&self, intent: Option<&str>) -> Option<Probe<'a>> {
        self.field.map(|f| f.probe(self.body.mass, self.body.source, intent))
    }

    fn badness(&self, a: &[f64; 10], intent: Option<&str>) -> (f64, [f64; 10]) {
        self.probe(intent).map_or((0.0, [0.0; 10]), |mut p| p.badness(a))
    }

    /// One step from `a`, deflected around disliked regions: the downhill part is cut by up to
    /// LAMBDA_MAX, and the lost length given back in proportion to the room to slide sideways, so a
    /// glancing step slides past at full length and a head-on one slows down.
    fn step(&mut self, a: &mut [f64; 10], mut d: [f64; 10], probe: &mut Option<Probe<'a>>) {
        let amount = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        if amount == 0.0 {
            return;
        }
        let (bad, g) = probe.as_mut().map_or((0.0, [0.0; 10]), |p| p.badness(a));
        self.max_badness = self.max_badness.max(bad);
        let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        let dg: f64 = d.iter().zip(&g).map(|(x, y)| x * y).sum();
        if bad > 1e-3 && gn > 0.0 && dg > 0.0 {
            let lambda = LAMBDA_MAX * (1.0 - (-bad / B0).exp());
            let along = dg / gn;
            for k in 0..10 {
                d[k] -= lambda * along * g[k] / gn;
            }
            let len = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let side = (1.0 - (along / amount).powi(2)).max(0.0).sqrt();
            let want = len + (amount - len) * side;
            if len > 0.0 {
                d.iter_mut().for_each(|v| *v *= want / len);
            }
            self.deflected += 1;
        }
        for k in 0..10 {
            a[k] = (a[k] + d[k]).clamp(-AXIS_LIMIT, AXIS_LIMIT);
        }
        self.steps += 1;
    }

    /// Walks from `start` through the weighted windows in DNA order, each window a step of its
    /// 10-axis vector (from `d0`) x scale / the weights' norm.
    fn walk(&mut self, start: [f64; 10], codes: &[u16], w: &Weights, d0: usize, scale: f64, intent: Option<&str>) -> [f64; 10] {
        let n = norm(w);
        let mut a = start;
        if n == 0.0 {
            return a;
        }
        let t = table();
        let mut probe = self.probe(intent);
        for &(i, x) in w {
            let v = &t[codes[i] as usize];
            let d: [f64; 10] = std::array::from_fn(|k| scale * x * v[d0 + k] / n);
            self.step(&mut a, d, &mut probe);
        }
        a
    }
}

// ------------------------------------------------------------------ syllables

/// One syllable type in a creature's inventory.
#[derive(Clone, Debug, PartialEq)]
struct SylType {
    cons: Option<Consonant>,
    vowel: [f64; 2],
    glide_to: Option<[f64; 2]>,
    /// [length x, pitch step, pitch slope, level x]
    prosody: [f64; 4],
}

fn vowel_at(f1: f64, f2: f64) -> [f64; 2] {
    [(1.0 + 0.3 * f1).clamp(0.5, 1.5), (1.05 + 0.3 * f2).clamp(0.55, 1.6)]
}

fn consonant(constriction: f64, place: f64, voiced: bool, lips: bool) -> Option<Consonant> {
    use Consonant::*;
    // Ordered by how closed the mouth gets, so a small change moves one band: none, breath,
    // fricative, stop.
    let manner = if constriction < -0.2 { 0 } else if constriction < 0.3 { 1 } else if constriction < 0.9 { 2 } else { 3 };
    let mut place = [-0.84, -0.25, 0.25, 0.84].iter().filter(|&&t| place >= t).count();
    if place == 0 && !lips {
        place = 1; // a beak: labials move to the teeth
    }
    match manner {
        1 => Some(H),
        2 => Some([F, Th, S, Sh, H][place]),
        3 => Some(match (place, voiced) {
            (0, false) => P,
            (0, true) => B,
            (1 | 2, false) => T,
            (1 | 2, true) => D,
            (3, false) => Ch,
            (3, true) => Jh,
            (_, false) => K,
            (_, true) => G,
        }),
        _ => None,
    }
}

fn inventory(codes: &[u16], body: &BodyInput) -> Vec<SylType> {
    let n = codes.len();
    let count = (4.0 + 1.2 * score(codes, &even(n, 0.0, INVENTORY_END), dim::TYPE_COUNT)).round().clamp(2.0, MAX_TYPES as f64) as usize;
    let width = INVENTORY_END / MAX_TYPES as f64;
    (0..count)
        .map(|t| {
            let w = bell(n, 0.0, INVENTORY_END, (t as f64 + 0.5) * width, TYPE_SD * width);
            let s = |d: usize| score(codes, &w, d);
            SylType {
                cons: consonant(s(dim::CONSTRICTION), s(dim::PLACE), s(dim::VOICING) > 0.0, body.lips),
                vowel: vowel_at(s(dim::F1), s(dim::F2)),
                glide_to: (s(dim::GLIDE) > 0.8).then(|| vowel_at(s(dim::F1B), s(dim::F2B))),
                prosody: [
                    (0.35 * s(dim::LENGTH)).exp().clamp(0.5, 2.0),
                    (3.0 * s(dim::PITCH)).clamp(-8.0, 8.0),
                    (4.0 * s(dim::SLOPE)).clamp(-10.0, 10.0),
                    (0.2 * s(dim::LEVEL)).exp().clamp(0.5, 1.2),
                ],
            }
        })
        .collect()
}

/// A decoded phrase: which inventory type each syllable is, and how it joins the next.
#[derive(Clone, Debug, PartialEq)]
struct Phrase {
    picks: Vec<usize>,
    /// Per syllable: 1 = glides into the next, -1 = pause after it, 0 = plain.
    links: Vec<i8>,
    gap: f64,
    accel: f64,
    extra_repeats: f64,
    swell: [f64; 5],
}

fn stretch(p: usize) -> (f64, f64) {
    let w = (1.0 - INVENTORY_END) / PHRASES.len() as f64;
    (INVENTORY_END + p as f64 * w, INVENTORY_END + (p + 1) as f64 * w)
}

fn phrase(codes: &[u16], p: usize, types: usize) -> Phrase {
    let n = codes.len();
    let (a, b) = stretch(p);
    let (_, lo, hi, typical, spread) = PHRASES[p];
    let all = even(n, a, b);
    let s = |d: usize| score(codes, &all, d);
    let count = (typical + spread * s(dim::COUNT)).round().clamp(lo as f64, hi as f64) as usize;
    let width = (b - a) / hi as f64;
    let mut picks = Vec::new();
    let mut links = Vec::new();
    for j in 0..count {
        let w = bell(n, a, b, a + (j as f64 + 0.5) * width, SLOT_SD * width);
        picks.push(band(score(codes, &w, dim::PICK), types));
        let link = score(codes, &w, dim::LINK);
        links.push(if j + 1 == count { 0 } else if link > 1.0 { 1 } else if link < -1.2 { -1 } else { 0 });
    }
    let accel = s(dim::ACCEL);
    Phrase {
        picks,
        links,
        gap: (0.4 * s(dim::GAP)).exp().clamp(0.4, 2.5),
        accel: if accel < -0.8 { 0.85 } else if accel > 0.8 { 1.15 } else { 1.0 },
        extra_repeats: (0.9 * s(dim::REPEATS)).round().clamp(0.0, 2.0),
        swell: std::array::from_fn(|k| (0.25 * s(dim::SWELL + k)).exp()),
    }
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn vowel_spec(v: [f64; 2]) -> VowelSpec {
    let (f1, f2) = (v[0], v[1]);
    let f3 = 1.0 + 0.15 * (f2 - 1.0);
    let rounding = ((1.05 - f2) * 1.6).clamp(0.0, 1.0);
    VowelSpec::Ratios(vec![r3(f1), r3(f2), r3(f3), r3(rounding)])
}

fn cons_name(c: Consonant) -> &'static str {
    CONSONANTS.iter().find(|e| e.2 == c).map(|e| e.0).unwrap_or("h")
}

/// The nearest named vowel, for reports.
fn vowel_name(v: [f64; 2]) -> &'static str {
    VOWELS
        .iter()
        .min_by(|x, y| {
            let d = |e: &[f64; 3]| (e[0] - v[0]).powi(2) + (e[1] - v[1]).powi(2);
            d(&x.2).partial_cmp(&d(&y.2)).unwrap()
        })
        .map(|e| if e.1.is_empty() { e.0 } else { e.1 })
        .unwrap()
}

fn type_text(t: &SylType) -> String {
    let mut s = String::new();
    if let Some(c) = t.cons {
        s.push_str(cons_name(c));
    }
    s.push_str(vowel_name(t.vowel));
    if let Some(g) = t.glide_to {
        s.push('~');
        s.push_str(vowel_name(g));
    }
    s
}

/// A phrase as the synth's vowel list and per-syllable prosody. Glide-joined slots become one
/// syllable (the later one's consonant is dropped), which keeps the first's prosody.
fn render_phrase(ph: &Phrase, inv: &[SylType]) -> (Vec<VowelSpec>, Vec<[f64; 4]>, String) {
    let mut list = Vec::new();
    let mut prosody = Vec::new();
    let mut text = Vec::new();
    let mut joined = false;
    let mut vowels_in_syllable = 0;
    for (j, &k) in ph.picks.iter().enumerate() {
        let t = &inv[k];
        let own = 1 + t.glide_to.is_some() as usize;
        // A syllable glides through at most MAX_GLIDE vowels; past that the next starts afresh.
        if joined && vowels_in_syllable + own > MAX_GLIDE {
            joined = false;
        }
        vowels_in_syllable = if joined { vowels_in_syllable + own } else { own };
        if joined {
            list.push(VowelSpec::Name("~".into()));
            text.push("~".to_string());
        } else {
            if let Some(c) = t.cons {
                list.push(VowelSpec::Name(cons_name(c).into()));
            }
            prosody.push(t.prosody.map(r3));
            if !text.is_empty() {
                text.push(" ".into());
            }
        }
        list.push(vowel_spec(t.vowel));
        if let Some(g) = t.glide_to {
            list.push(VowelSpec::Name("~".into()));
            list.push(vowel_spec(g));
        }
        text.push(if joined { type_text(&SylType { cons: None, ..t.clone() }) } else { type_text(t) });
        joined = ph.links[j] == 1;
        if ph.links[j] == -1 {
            list.push(VowelSpec::Name("|".into()));
            text.push(" |".into());
        }
    }
    (list, prosody, text.concat())
}

// ------------------------------------------------------------------ output

/// What the decoder did, for debugging and tuning (`cv_genome_report`).
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub decoder_version: u32,
    pub space_version: u32,
    /// None when decoded without the preference field.
    pub field_version: Option<u32>,
    pub bases: usize,
    pub steps: usize,
    pub deflected: usize,
    pub max_badness: f64,
    pub base: BTreeMap<String, f64>,
    pub base_badness: f64,
    /// The syllable types, as text: consonant, nearest named vowel, "~" and a second vowel.
    pub inventory: Vec<String>,
    pub speak: String,
    pub calls: BTreeMap<String, CallReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CallReport {
    /// Where the call is on the axes, and its offset from the base voice.
    pub axes: BTreeMap<String, f64>,
    pub offset: BTreeMap<String, f64>,
    pub badness: f64,
    pub phrase: String,
    /// The inventory type of each syllable slot.
    pub picks: Vec<usize>,
}

/// Which versions to decode under. `field: None` decodes without steering.
#[derive(Clone, Debug)]
pub struct Versions {
    pub field: Option<u32>,
}

impl Default for Versions {
    fn default() -> Self {
        Versions { field: Some(field::builtin().version) }
    }
}

impl Versions {
    /// From JSON such as {"decoder": 2, "space": 1, "field": 1}; "field": 0 or false turns
    /// steering off. Missing versions mean the current ones.
    pub fn from_json(json: &str) -> Result<Versions, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("versions JSON: {e}"))?;
        let check = |key: &str, have: u32| -> Result<(), String> {
            match v.get(key).and_then(|x| x.as_u64()) {
                Some(n) if n as u32 != have => Err(format!("{key} version {n} isn't in this build (it has {have})")),
                _ => Ok(()),
            }
        };
        check("decoder", DECODER_VERSION)?;
        check("space", SPACE_VERSION)?;
        let field = match v.get("field") {
            Some(serde_json::Value::Bool(false)) => None,
            Some(x) if x.as_u64() == Some(0) => None,
            Some(x) => {
                let have = field::builtin().version;
                match x.as_u64() {
                    Some(n) if n as u32 == have => Some(have),
                    Some(n) => return Err(format!("field version {n} isn't in this build (it has {have})")),
                    None if x.as_bool() == Some(true) => Some(have),
                    None => return Err("versions.field must be a number or false".into()),
                }
            }
            None => Some(field::builtin().version),
        };
        Ok(Versions { field })
    }
}

fn named(a: &[f64; 10]) -> BTreeMap<String, f64> {
    AXES.iter().zip(a).map(|(n, v)| (n.to_string(), r3(*v))).collect()
}

fn setting(s: &[f64; 29], name: &str) -> f64 {
    s[SETTINGS.iter().position(|n| *n == name).expect("setting")]
}

/// A call as overrides of its intent's template: its timbre offset moves the settings an intent
/// can override, and its phrase, prosody and rhythm replace the template's.
fn call_override(body: &BodyInput, name: &str, base: &[f64; 10], resid: &[f64; 29], pos: &[f64; 10], ph: &Phrase, inv: &[SylType]) -> IntentOverride {
    let t = base_intent(name).expect("intent");
    let st = space::settings(body, base, resid);
    let sc = space::settings(body, pos, resid);
    let d = |n: &str| setting(&sc, n) - setting(&st, n);
    let mut o = IntentOverride::default();
    let set = |slot: &mut Option<f64>, base: f64, value: f64| {
        if (value - base).abs() > 1e-3 {
            *slot = Some(r3(value));
        }
    };
    let fold = -12.0 * (setting(&sc, "fold_scale") / setting(&st, "fold_scale")).log2();
    set(&mut o.pitch, t.pitch, t.pitch + d("pitch") + fold);
    let range = setting(&sc, "pitch_range") / setting(&st, "pitch_range");
    if (range - 1.0).abs() > 0.01 && !t.contour.is_empty() {
        o.contour = Some(t.contour.iter().map(|&(u, s)| (u, r3(s * range))).collect());
    }
    set(&mut o.harsh, t.harsh, t.harsh + d("harshness") + 0.5 * d("roughness"));
    set(&mut o.breath, t.breath, (t.breath + d("breathiness")).max(0.0));
    set(&mut o.nasal, t.nasal, (t.nasal + d("nasality")).clamp(0.0, 1.0));
    set(&mut o.vibrato, t.vibrato, t.vibrato + d("vibrato"));
    set(&mut o.subharmonics, t.subharmonics, (t.subharmonics + d("subharmonics")).max(0.0));
    set(&mut o.chaos, t.chaos, (t.chaos + d("chaos")).max(0.0));
    set(&mut o.biphonation, t.biphonation, (t.biphonation + d("biphonation")).max(0.0));
    set(&mut o.jumps, t.jumps, (t.jumps + d("jumps")).max(0.0));
    set(&mut o.larynx, t.larynx, (t.larynx + 2.0 * d("larynx_drop")).clamp(0.0, 1.5));

    let (vowels, prosody, _) = render_phrase(ph, inv);
    // How long the phrase runs in units of one plain syllable: each syllable's prosody length, and
    // the synth's lengthening for each vowel it glides through (1 + 0.7 per extra vowel).
    let syllables: f64 = crate::model::syllables(&vowels)
        .iter()
        .zip(&prosody)
        .map(|(s, p)| p[0] * (1.0 + 0.7 * (s.vowels.len() as f64 - 1.0)))
        .sum::<f64>()
        .max(1.0);
    o.vowels = Some(vowels);
    o.prosody = Some(prosody);
    o.sequence = Some(true);
    o.dur = Some(r3((t.dur / syllables).clamp(0.08, 1.6))); // the phrase lasts about as long as the template call
    o.gap = Some(r3(t.gap.min(0.15) * ph.gap));
    // Extra repeats only for short calls: long ones (mating, warn) already run for seconds.
    let extra = if t.dur < 1.0 { ph.extra_repeats } else { 0.0 };
    o.repeats = Some((t.repeats.0 + extra, t.repeats.1 + extra));
    if ph.accel != 1.0 {
        o.accel = Some(ph.accel);
    }
    o.rhythm = Some(ph.swell.iter().enumerate().map(|(k, &m)| (k as f64 / 4.0, r3(m))).collect());
    o
}

/// Decodes the voice section of a genome on this body.
pub fn decode(body: &BodyInput, dna: &str, versions: &Versions) -> Result<(Creature, Report), String> {
    decode_with(body, dna, versions.field.map(|_| field::builtin()))
}

/// Decodes with a given preference field (or none).
pub fn decode_with(body: &BodyInput, dna: &str, field: Option<&Field>) -> Result<(Creature, Report), String> {
    if !(body.mass > 0.0 && body.mass.is_finite()) {
        return Err(format!("body.mass must be > 0 (got {})", body.mass));
    }
    let codes = windows(dna);
    let n = codes.len();
    let mut walker = Walker { body, field, steps: 0, deflected: 0, max_badness: 0.0 };

    let everything = even(n, 0.0, 1.0);
    let base = walker.walk([0.0; 10], &codes, &everything, dim::AXES, BASE_SCALE, None);
    let resid: [f64; 29] = std::array::from_fn(|k| RESIDUAL_SCALE * score(&codes, &everything, dim::RESIDUAL + k));
    let inv = inventory(&codes, body);

    let plain: Vec<VowelSpec> = inv.iter().map(|t| vowel_spec(t.vowel)).collect();
    let mut creature = space::creature(body, &base, &resid, plain);
    let mut calls = BTreeMap::new();
    let mut speak = String::new();
    for (p, &(name, ..)) in PHRASES.iter().enumerate() {
        let ph = phrase(&codes, p, inv.len());
        let (a, b) = stretch(p);
        if name == "speak" {
            let (vowels, prosody, text) = render_phrase(&ph, &inv);
            let mut o = IntentOverride { vowels: Some(vowels), prosody: Some(prosody), ..Default::default() };
            if ph.gap != 1.0 {
                o.gap = Some(r3(base_intent("speak").unwrap().gap * ph.gap));
            }
            creature.voice.calls.insert("speak".into(), o);
            speak = text;
            continue;
        }
        let pos = walker.walk(base, &codes, &even(n, a, b), dim::OFFSET, CALL_SCALE, Some(name));
        creature.voice.calls.insert(name.into(), call_override(body, name, &base, &resid, &pos, &ph, &inv));
        let offset: [f64; 10] = std::array::from_fn(|k| pos[k] - base[k]);
        calls.insert(
            name.to_string(),
            CallReport {
                axes: named(&pos),
                offset: named(&offset),
                badness: r3(walker.badness(&pos, Some(name)).0),
                phrase: render_phrase(&ph, &inv).2,
                picks: ph.picks.clone(),
            },
        );
    }
    creature.validate()?;
    let report = Report {
        decoder_version: DECODER_VERSION,
        space_version: SPACE_VERSION,
        field_version: field.map(|f| f.version),
        bases: dna.bytes().filter(|&c| base_index(c).is_some()).count(),
        steps: walker.steps,
        deflected: walker.deflected,
        max_badness: r3(walker.max_badness),
        base: named(&base),
        base_badness: r3(walker.badness(&base, None).0),
        inventory: inv.iter().map(type_text).collect(),
        speak,
        calls,
    };
    Ok((creature, report))
}

/// Which part of the decoding reads the base at this position (a share of the section is enough):
/// the inventory, or a phrase's stretch. The base voice reads everything.
pub fn region(dna: &str, pos: usize) -> String {
    let len = dna.bytes().filter(|&c| base_index(c).is_some()).count().max(1);
    let x = pos as f64 / len as f64;
    if x < INVENTORY_END {
        let t = (x / (INVENTORY_END / MAX_TYPES as f64)) as usize;
        return format!("inventory, type {}", t + 1);
    }
    let p = (0..PHRASES.len()).find(|&p| x < stretch(p).1).unwrap_or(PHRASES.len() - 1);
    format!("{} stretch", PHRASES[p].0)
}

// ------------------------------------------------------------------ random genomes and mutations

/// A random voice section of the standard length.
pub fn random_genome(rng: &mut Rng) -> String {
    (0..SECTION_LENGTH).map(|_| BASES[rng.integer_inclusive(0, 3) as usize] as char).collect()
}

pub const MUTATIONS: [&str; 5] = ["point", "insertion", "deletion", "duplication", "inversion"];

/// One random mutation of the given kind. A duplication copies a stretch of 6-60 bases next to
/// itself; an inversion reverse-complements 3-30 bases.
pub fn mutate(dna: &str, kind: &str, rng: &mut Rng) -> String {
    let mut d = dna.as_bytes().to_vec();
    if d.is_empty() {
        return dna.to_string();
    }
    let at = |rng: &mut Rng, n: usize| rng.integer_inclusive(0, n as i64 - 1) as usize;
    let base = |rng: &mut Rng| BASES[rng.integer_inclusive(0, 3) as usize];
    match kind {
        "point" => {
            let i = at(rng, d.len());
            let old = d[i];
            d[i] = loop {
                let b = base(rng);
                if b != old {
                    break b;
                }
            };
        }
        "insertion" => {
            let i = at(rng, d.len() + 1);
            for _ in 0..rng.integer_inclusive(1, 3) {
                let b = base(rng);
                d.insert(i, b);
            }
        }
        "deletion" => {
            let i = at(rng, d.len());
            let n = (rng.integer_inclusive(1, 3) as usize).min(d.len() - i);
            d.drain(i..i + n);
        }
        "duplication" => {
            let n = (rng.integer_inclusive(6, 60) as usize).min(d.len());
            let i = at(rng, d.len() - n + 1);
            let copy = d[i..i + n].to_vec();
            d.splice(i + n..i + n, copy);
        }
        "inversion" => {
            let n = (rng.integer_inclusive(3, 30) as usize).min(d.len());
            let i = at(rng, d.len() - n + 1);
            let seg: Vec<u8> = d[i..i + n]
                .iter()
                .rev()
                .map(|&b| match b {
                    b'A' => b'T',
                    b'T' => b'A',
                    b'C' => b'G',
                    b'G' => b'C',
                    o => o,
                })
                .collect();
            d[i..i + n].copy_from_slice(&seg);
        }
        _ => {}
    }
    String::from_utf8(d).unwrap()
}

/// One generation of mutation as Unity's Genome.MutateSection does it with MutationRates.Default,
/// every rate multiplied by `scale`: per base, a deletion (1-3 bases, 0.0015), an insertion (1-3,
/// 0.0015) or a point change (0.006, which may redraw the same base); then per section, maybe a
/// duplication (0.2, 6-95 bases), a span deletion (0.12, 3-47), an inversion (0.06, 6-45) and a
/// reversal (0.06, 6-45).
pub fn mutate_generation(dna: &str, scale: f64, rng: &mut Rng) -> String {
    let base = |rng: &mut Rng| BASES[rng.integer_inclusive(0, 3) as usize];
    let len = |rng: &mut Rng, lo: usize, hi: usize, cap: usize| (rng.integer_inclusive(lo as i64, hi as i64) as usize).min(cap).max(1);
    let mut out: Vec<u8> = Vec::with_capacity(dna.len() + 64);
    let mut skip = 0;
    for &c in dna.as_bytes() {
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let mut roll = rng.random();
        if roll < 0.0015 * scale {
            skip = len(rng, 1, 3, usize::MAX) - 1;
            continue;
        }
        roll -= 0.0015 * scale;
        if roll < 0.0015 * scale {
            for _ in 0..len(rng, 1, 3, usize::MAX) {
                let b = base(rng);
                out.push(b);
            }
            out.push(c);
            continue;
        }
        roll -= 0.0015 * scale;
        out.push(if roll < 0.006 * scale { base(rng) } else { c });
    }
    let mut d = out;
    if rng.random() < (0.2 * scale).min(1.0) && d.len() > 10 {
        let n = len(rng, 6, 95, d.len());
        let i = rng.integer_inclusive(0, (d.len() - n) as i64) as usize;
        let copy = d[i..i + n].to_vec();
        d.splice(i + n..i + n, copy);
    }
    if rng.random() < (0.12 * scale).min(1.0) && d.len() > 40 {
        let n = len(rng, 3, 47, d.len() / 4);
        let i = rng.integer_inclusive(0, (d.len() - n) as i64) as usize;
        d.drain(i..i + n);
    }
    for reverse_complement in [true, false] {
        if rng.random() < (0.06 * scale).min(1.0) && d.len() > 20 {
            let n = len(rng, 6, 45, d.len());
            let i = rng.integer_inclusive(0, (d.len() - n) as i64) as usize;
            d[i..i + n].reverse();
            if reverse_complement {
                d[i..i + n].iter_mut().for_each(|b| {
                    *b = match *b {
                        b'A' => b'T',
                        b'T' => b'A',
                        b'C' => b'G',
                        b'G' => b'C',
                        o => o,
                    }
                });
            }
        }
    }
    d.truncate(4000);
    String::from_utf8(d).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> BodyInput {
        BodyInput { mass: 12.0, ..BodyInput::default() }
    }

    #[test]
    fn traits_are_standard_normal_over_random_dna() {
        let mut rng = Rng::new(2);
        let codes: Vec<Vec<u16>> = (0..400).map(|_| windows(&random_genome(&mut rng))).collect();
        for d in [dim::AXES, dim::PICK, dim::F1] {
            let s: Vec<f64> = codes.iter().map(|c| score(c, &even(c.len(), 0.0, 1.0), d)).collect();
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            let sd = (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
            assert!(mean.abs() < 0.15 && (sd - 1.0).abs() < 0.15, "dim {d}: mean {mean}, sd {sd}");
        }
    }

    #[test]
    fn decodes_deterministically() {
        let mut rng = Rng::new(3);
        for _ in 0..20 {
            let dna = random_genome(&mut rng);
            let (a, _) = decode(&body(), &dna, &Versions::default()).unwrap();
            let (b, _) = decode(&body(), &dna, &Versions::default()).unwrap();
            assert_eq!(a.to_json(), b.to_json());
        }
    }

    #[test]
    fn empty_genome_is_the_body_prior() {
        let (c, r) = decode(&body(), "", &Versions::default()).unwrap();
        assert_eq!(r.steps, 0);
        let plain = space::creature(&body(), &[0.0; 10], &[0.0; 29], vec![VowelSpec::Name("a".into())]);
        assert_eq!(c.body.tract_length, plain.body.tract_length);
    }

    #[test]
    fn a_mutation_in_one_stretch_leaves_the_other_calls_alone() {
        let mut rng = Rng::new(5);
        let dna = random_genome(&mut rng);
        let (_, before) = decode_with(&body(), &dna, None).unwrap();
        // a point mutation in the middle of the warn stretch (phrase 2)
        let (a, b) = stretch(2);
        let i = ((a + b) / 2.0 * dna.len() as f64) as usize;
        let mut m = dna.clone().into_bytes();
        m[i] = if m[i] == b'A' { b'C' } else { b'A' };
        let (_, after) = decode_with(&body(), &String::from_utf8(m).unwrap(), None).unwrap();
        for (k, c) in &before.calls {
            if k != "warn" {
                assert_eq!(c.phrase, after.calls[k].phrase, "{k}");
                assert_eq!(c.offset, after.calls[k].offset, "{k}");
            }
        }
        assert_eq!(before.inventory, after.inventory);
        assert_ne!(before.calls["warn"].offset, after.calls["warn"].offset);
        let moved = AXES.iter().map(|n| (before.base[*n] - after.base[*n]).powi(2)).sum::<f64>().sqrt();
        assert!(moved > 0.0 && moved < 0.3, "the base voice moves a little: {moved}");
    }

    #[test]
    fn every_random_genome_makes_a_valid_creature() {
        let mut rng = Rng::new(11);
        for _ in 0..60 {
            let b = space::sample_body(&mut rng);
            let mut dna = random_genome(&mut rng);
            let len = rng.integer_inclusive(0, 2000) as usize;
            dna.truncate(len);
            for _ in 0..3 {
                let k = MUTATIONS[rng.integer_inclusive(0, 4) as usize];
                dna = mutate(&dna, k, &mut rng);
            }
            let (c, _) = decode(&b, &dna, &Versions::default()).unwrap();
            c.validate().unwrap();
            for intent in ["speak", "mating", "alarm"] {
                assert!(!crate::synth::create_vocalization(&c.body, &c.voice, intent, 1, 10.0, 0.0).unwrap().is_empty());
            }
        }
    }

    #[test]
    fn beak_has_no_labials() {
        let mut rng = Rng::new(9);
        let bird = BodyInput { mass: 0.3, lips: false, ..BodyInput::default() };
        for _ in 0..40 {
            let (_, r) = decode(&bird, &random_genome(&mut rng), &Versions { field: None }).unwrap();
            assert!(r.inventory.iter().all(|t| !t.starts_with(['p', 'b', 'f'])), "{:?}", r.inventory);
        }
    }

    #[test]
    fn prosody_lengthens_a_syllable() {
        let mut c = space::creature(&body(), &[0.0; 10], &[0.0; 29], vec![VowelSpec::Name("a".into())]);
        let len = |c: &Creature| crate::synth::create_vocalization(&c.body, &c.voice, "speak", 1, 1.0, 0.0).unwrap().len();
        c.voice.vowels = vec![VowelSpec::Name("a i".into())];
        let plain = len(&c);
        c.voice.calls.insert("speak".into(), IntentOverride { prosody: Some(vec![[2.0, 0.0, 0.0, 1.0], [2.0, 3.0, -4.0, 0.8]]), ..Default::default() });
        assert!(len(&c) as f64 > 1.4 * plain as f64);
    }

    #[test]
    fn steering_slides_around_a_disliked_spot() {
        // One disliked point at roughness +4, pitch +0.5, on a 12 kg larynx voice.
        let mut x = vec![0.0; 18];
        x[4] = 4.0;
        x[1] = 0.5;
        x[10] = 12f64.log10();
        x[11] = 1.0;
        let features = r#"["resonance_scale","pitch_height","pitch_liveliness","breathy","roughness","instability","brightness","ringiness","nasality","body_resonance","log_mass","source=larynx","source=syrinx","source=pulse","source=hiss","intent=idle","intent=mating","intent=warn"]"#;
        let json = format!(
            r#"{{"field_version": 99, "space_version": [1], "features": {features}, "length_scale": {ls:?},
               "signal_variance": 1.0, "noise": 0.01, "y_mean": 0.0, "y_std": 1.0, "X": [{x:?}],
               "alpha": [-3.0], "L": [[{l}]], "badness": {{"theta": 0.2, "kappa": 1.0, "beta": 8.0}}}}"#,
            ls = vec![1.0; 18], l = 1.01f64.sqrt()
        );
        let f = Field::from_json(&json).unwrap();
        let b = body();
        let walk = |field: Option<&Field>| {
            let mut w = Walker { body: &b, field, steps: 0, deflected: 0, max_badness: 0.0 };
            let mut probe = w.probe(None);
            let mut a = [0.0; 10];
            let mut worst: f64 = 0.0;
            for _ in 0..80 {
                let mut d = [0.0; 10];
                d[4] = 0.07; // roughness +5.6 in all, straight past the spot
                w.step(&mut a, d, &mut probe);
                worst = worst.max(f.badness(&f.input(&a, 12.0, crate::model::Source::Larynx, None)).0);
            }
            (a, worst, w.deflected)
        };
        let (a_off, worst_off, _) = walk(None);
        let (a_on, worst_on, deflected) = walk(Some(&f));
        assert!(deflected > 0);
        assert!(worst_on < 0.7 * worst_off, "steered {worst_on} vs straight {worst_off}");
        assert!(a_on[1] < -0.1, "it slides away on the side it can: {a_on:?}");
        assert!((a_off[4] - 5.6).abs() < 1e-9);
    }
}
