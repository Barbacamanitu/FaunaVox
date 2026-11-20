//! Creature description: Body (physical traits), Voice (heritable traits) and the intent
//! templates. Field names and defaults match create_vocalization.py, so the same JSON works in
//! both: {"body": {...}, "voice": {...}}.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub type Points = Vec<(f64, f64)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    #[default]
    Larynx,
    Syrinx,
    Pulse,
    Hiss,
}

/// A vowel: a name (a e i o u @) or F1-F3 ratios to a neutral tube, optionally + lip rounding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VowelSpec {
    Name(String),
    Ratios(Vec<f64>),
}

/// Vowel table: (name, ASCII alias, F1-F3 relative to a uniform tube's 500/1500/2500 Hz, lip rounding).
/// The first six are adult human averages. The rest are the eleven SSBE (Standard Southern British
/// English) monophthongs: mean F1/F2 of three L1 speakers; F3 is estimated from typical SSBE values.
pub const VOWELS: [(&str, &str, [f64; 3], f64); 17] = [
    ("a", "", [1.46, 0.73, 0.98], 0.0),
    ("e", "", [1.06, 1.23, 0.99], 0.0),
    ("i", "", [0.54, 1.53, 1.20], 0.0),
    ("o", "", [1.14, 0.56, 0.96], 0.6),
    ("u", "", [0.60, 0.58, 0.90], 1.0),
    ("@", "", [1.00, 1.00, 1.00], 0.2), // schwa, neutral tube
    ("i:", "iy", [296.0 / 500.0, 2241.0 / 1500.0, 1.16], 0.0), // fleece
    ("ɪ", "ih", [396.0 / 500.0, 1839.0 / 1500.0, 1.04], 0.0),  // kit
    ("ɛ", "eh", [532.0 / 500.0, 1656.0 / 1500.0, 1.02], 0.0),  // dress
    ("æ", "ae", [667.0 / 500.0, 1565.0 / 1500.0, 1.00], 0.0),  // trap
    ("ʌ", "ah", [661.0 / 500.0, 1296.0 / 1500.0, 0.98], 0.0),  // strut
    ("ɑ:", "aa", [680.0 / 500.0, 1193.0 / 1500.0, 0.98], 0.0), // palm / start
    ("ɒ", "oh", [643.0 / 500.0, 1019.0 / 1500.0, 0.96], 0.5),  // lot
    ("ɔ:", "ao", [480.0 / 500.0, 857.0 / 1500.0, 0.96], 0.8),  // thought
    ("ʊ", "uh", [395.0 / 500.0, 1408.0 / 1500.0, 0.92], 0.7),  // foot
    ("u:", "uw", [386.0 / 500.0, 1587.0 / 1500.0, 0.90], 0.8), // goose (fronted)
    ("ɜ:", "er", [519.0 / 500.0, 1408.0 / 1500.0, 0.98], 0.1), // nurse (non-rhotic)
];

fn vowel_entry(name: &str) -> Option<&'static (&'static str, &'static str, [f64; 3], f64)> {
    VOWELS.iter().find(|(n, alias, _, _)| *n == name || (!alias.is_empty() && *alias == name))
}

/// Vowel-list separators. In a sequential call (the "speak" intent, or any intent with
/// "sequence": true) every vowel is its own syllable unless joined: "~" glides from the vowel
/// before it into the one after, inside one syllable; "|" is a pause. Any string in the list
/// may hold several tokens ("a i~u | e"); spaces and commas separate them. Other calls ignore
/// the separators and use the vowels in order.
pub const GLIDE: &str = "~";
pub const PAUSE: &str = "|";

#[derive(Clone, Debug, PartialEq)]
pub enum VowelToken {
    Vowel(VowelSpec),
    Consonant(Consonant),
    Glide,
    Pause,
}

/// Consonants for sequential calls. `H` is breath through the tract shaped like the vowel next to
/// it; s, sh, f and th are turbulence at the teeth or lips (fricatives); p, t, k, b, d and g close
/// the tract and release it with a burst (stops).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consonant {
    H,
    S,
    Sh,
    F,
    Th,
    /// Affricates: a stop (a brief silence) released into a short sh ("itch", "judge").
    Ch,
    Jh,
    /// Stops: the lips (p, b), the tongue tip (t, d) or the back of the tongue (k, g) close the
    /// tract, then release it with a burst. p, t and k are followed by breath before the voice
    /// starts; b, d and g are voiced through the closure between vowels.
    P,
    T,
    K,
    B,
    D,
    G,
}

impl Consonant {
    pub fn is_affricate(self) -> bool {
        matches!(self, Consonant::Ch | Consonant::Jh)
    }

    pub fn is_stop(self) -> bool {
        matches!(self, Consonant::P | Consonant::T | Consonant::K | Consonant::B | Consonant::D | Consonant::G)
    }

    /// b, d, g and j: the vocal folds keep vibrating.
    pub fn is_voiced(self) -> bool {
        matches!(self, Consonant::B | Consonant::D | Consonant::G | Consonant::Jh)
    }
}

/// Consonant names: (name, IPA alias, consonant).
pub const CONSONANTS: [(&str, &str, Consonant); 13] = [
    ("h", "", Consonant::H),
    ("s", "", Consonant::S),
    ("sh", "ʃ", Consonant::Sh),
    ("f", "", Consonant::F),
    ("th", "θ", Consonant::Th),
    ("ch", "tʃ", Consonant::Ch),
    ("j", "dʒ", Consonant::Jh),
    ("p", "", Consonant::P),
    ("t", "", Consonant::T),
    ("k", "", Consonant::K),
    ("b", "", Consonant::B),
    ("d", "", Consonant::D),
    ("g", "", Consonant::G),
];

pub fn consonant(name: &str) -> Option<Consonant> {
    CONSONANTS.iter().find(|(n, a, _)| *n == name || (!a.is_empty() && *a == name)).map(|e| e.2)
}

/// One syllable of a sequential call: the consonants before it, the vowels it glides through
/// (one = a steady vowel), the consonants after it, and whether a pause follows.
#[derive(Clone, Debug, PartialEq)]
pub struct Syllable {
    pub onset: Vec<Consonant>,
    pub vowels: Vec<VowelSpec>,
    pub coda: Vec<Consonant>,
    pub pause: bool,
}

/// A vowel list in JSON: a list (`["a", "i~u", [0.6, 1.4, 1.0]]`) or one string holding the whole
/// phrase (`"a i~u | e"`), which is split into vowels and separators when used.
#[derive(Deserialize)]
#[serde(untagged)]
enum VowelListJson {
    Phrase(String),
    List(Vec<VowelSpec>),
}

impl From<VowelListJson> for Vec<VowelSpec> {
    fn from(v: VowelListJson) -> Self {
        match v {
            VowelListJson::Phrase(s) => vec![VowelSpec::Name(s)],
            VowelListJson::List(l) => l,
        }
    }
}

fn vowel_list<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<VowelSpec>, D::Error> {
    VowelListJson::deserialize(d).map(Into::into).map_err(|_| {
        serde::de::Error::custom("vowels must be a string such as \"a i~u | e\" or a list such as [\"a\", \"i~u\", [0.6, 1.4, 1.0]]")
    })
}

fn opt_vowel_list<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Vec<VowelSpec>>, D::Error> {
    Option::<VowelListJson>::deserialize(d).map(|o| o.map(Into::into)).map_err(|_| {
        serde::de::Error::custom("vowels must be a string such as \"a i~u | e\" or a list such as [\"a\", \"i~u\", [0.6, 1.4, 1.0]]")
    })
}

/// Splits a vowel list into vowels, consonants and separators.
pub fn vowel_tokens(list: &[VowelSpec]) -> Vec<VowelToken> {
    let mut out = Vec::new();
    for spec in list {
        match spec {
            VowelSpec::Ratios(_) => out.push(VowelToken::Vowel(spec.clone())),
            VowelSpec::Name(s) => {
                let mut word = String::new();
                let flush = |word: &mut String, out: &mut Vec<VowelToken>| {
                    if !word.is_empty() {
                        let w = std::mem::take(word);
                        out.push(match consonant(&w) {
                            Some(c) => VowelToken::Consonant(c),
                            None => VowelToken::Vowel(VowelSpec::Name(w)),
                        });
                    }
                };
                for ch in s.chars() {
                    match ch {
                        '~' => {
                            flush(&mut word, &mut out);
                            out.push(VowelToken::Glide);
                        }
                        '|' => {
                            flush(&mut word, &mut out);
                            out.push(VowelToken::Pause);
                        }
                        c if c.is_whitespace() || c == ',' => flush(&mut word, &mut out),
                        c => word.push(c),
                    }
                }
                flush(&mut word, &mut out);
            }
        }
    }
    out
}

/// The vowels of a list in order, without separators or consonants.
pub fn plain_vowels(list: &[VowelSpec]) -> Vec<VowelSpec> {
    vowel_tokens(list)
        .into_iter()
        .filter_map(|t| if let VowelToken::Vowel(v) = t { Some(v) } else { None })
        .collect()
}

/// A vowel list as syllables: every vowel its own, except runs joined by "~". Consonants start
/// the next syllable, except those before a pause or at the end, which close the one before.
pub fn syllables(list: &[VowelSpec]) -> Vec<Syllable> {
    let mut out: Vec<Syllable> = Vec::new();
    let mut pending: Vec<Consonant> = Vec::new();
    let mut join = false;
    for t in vowel_tokens(list) {
        let glide = matches!(t, VowelToken::Glide);
        match t {
            VowelToken::Vowel(v) => match out.last_mut() {
                Some(s) if join && !s.pause && pending.is_empty() => s.vowels.push(v),
                _ => out.push(Syllable { onset: std::mem::take(&mut pending), vowels: vec![v], coda: vec![], pause: false }),
            },
            VowelToken::Consonant(c) => pending.push(c),
            VowelToken::Glide => {}
            VowelToken::Pause => {
                if let Some(s) = out.last_mut() {
                    s.coda.append(&mut pending);
                    s.pause = true;
                }
            }
        }
        join = glide;
    }
    if let Some(s) = out.last_mut() {
        s.coda.append(&mut pending);
    }
    out
}

/// Checks every vowel in a list (separators allowed) and that there is at least one.
pub fn validate_vowels(list: &[VowelSpec], what: &str) -> Result<(), String> {
    let mut any = false;
    for t in vowel_tokens(list) {
        if let VowelToken::Vowel(v) = t {
            v.validate()?;
            any = true;
        }
    }
    if any { Ok(()) } else { Err(format!("{what} must hold at least one vowel")) }
}

impl VowelSpec {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            VowelSpec::Name(n) if vowel_entry(n).is_some() => Ok(()),
            VowelSpec::Name(n) => {
                let names: Vec<String> = VOWELS.iter().map(|(n, a, _, _)| if a.is_empty() { n.to_string() } else { format!("{n} ({a})") }).collect();
                Err(format!(
                    "unknown vowel {n:?}; use one of {} or [F1, F2, F3] ratios (consonants: h s sh f th ch j; separators: \"~\" glide, \"|\" pause)",
                    names.join(" ")
                ))
            }
            VowelSpec::Ratios(r) if r.len() >= 3 => Ok(()),
            VowelSpec::Ratios(_) => Err("a custom vowel needs at least 3 ratios [F1, F2, F3(, rounding)]".into()),
        }
    }

    pub fn ratios(&self) -> [f64; 3] {
        match self {
            VowelSpec::Name(n) => vowel_entry(n).map(|e| e.2).unwrap_or([1.0; 3]),
            VowelSpec::Ratios(r) => [r[0], r[1], r[2]],
        }
    }

    /// Lip rounding: 0 = spread, 1 = fully rounded.
    pub fn rounding(&self) -> f64 {
        match self {
            VowelSpec::Name(n) => vowel_entry(n).map(|e| e.3).unwrap_or(0.0),
            VowelSpec::Ratios(r) => r.get(3).copied().unwrap_or(0.0),
        }
    }
}

fn v(name: &str) -> VowelSpec {
    VowelSpec::Name(name.to_string())
}

/// Physical traits, normally taken from the creature's actual size.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Body {
    pub mass: f64,
    pub tract_length: Option<f64>,
    pub source: Source,
    pub fold_scale: f64,
    pub larynx_drop: f64,
    pub roughness: f64,
    pub breathiness: f64,
    pub nasality: f64,
    pub lung_capacity: f64,
    pub resonator_freq: Option<f64>,
    pub resonator_harmonic: f64,
    pub buzz: f64,
    pub chest: f64,
    pub chest_width: f64,
    pub noise_color: f64,
    pub tract_damping: Option<f64>,
    pub hf_loss: f64,
    pub low_cut: f64,
    /// dB of extra resonance at each formant (tone and character).
    pub formant_gain: f64,
    /// Bandwidth multiplier for that resonance: < 1 = sharper, ringier.
    pub formant_width: f64,
    /// 0..1: how much breath noise pulses with each glottal cycle (1 = fully fused into the voice).
    pub breath_pulse: f64,
    /// Fraction of each glottal cycle the folds are open: high = soft and breathy, low = pressed and bright.
    pub open_quotient: f64,
    /// Native-only convenience (Body.from_dimensions): when tract_length is not given, the tract
    /// runs 0.8 x head_length + 0.5 x neck_length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neck_length: Option<f64>,
}

impl Default for Body {
    fn default() -> Self {
        Body {
            mass: 50.0,
            tract_length: None,
            source: Source::Larynx,
            fold_scale: 1.0,
            larynx_drop: 0.15,
            roughness: 0.3,
            breathiness: 0.25,
            nasality: 0.1,
            lung_capacity: 1.0,
            resonator_freq: None,
            resonator_harmonic: 0.0,
            buzz: 0.0,
            chest: 1.0,
            chest_width: 70.0,
            noise_color: 1.0,
            tract_damping: None,
            hf_loss: 1.0,
            low_cut: 0.0,
            formant_gain: crate::synth::FORMANT_GAIN,
            formant_width: 1.0,
            breath_pulse: crate::synth::BREATH_PULSE,
            open_quotient: crate::synth::OPEN_QUOTIENT,
            head_length: None,
            neck_length: None,
        }
    }
}

/// Python treats 0 and None alike for these optional fields (`x or default`).
fn some_nonzero(x: Option<f64>) -> Option<f64> {
    x.filter(|v| *v != 0.0)
}

impl Body {
    pub fn tract(&self) -> f64 {
        if let Some(t) = some_nonzero(self.tract_length) {
            return t;
        }
        if let (Some(h), Some(n)) = (self.head_length, self.neck_length) {
            return 0.8 * h + 0.5 * n;
        }
        0.06 * self.mass.cbrt()
    }

    /// Natural pitch. Loosely allometric: bigger animals have bigger, slower folds.
    pub fn base_f0(&self) -> f64 {
        match self.source {
            Source::Syrinx => 2500.0 * self.mass.powf(-0.2) / self.fold_scale,
            Source::Pulse => 40.0 / self.fold_scale, // pulse repetition rate
            _ => 600.0 * self.mass.powf(-0.3) / self.fold_scale,
        }
    }

    pub fn carrier(&self) -> f64 {
        some_nonzero(self.resonator_freq).unwrap_or_else(|| 400.0 * self.mass.powf(-0.3))
    }

    /// Bigger lungs sustain longer calls.
    pub fn call_scale(&self) -> f64 {
        (self.mass / 50.0).powf(0.12) * self.lung_capacity
    }

    pub fn tract_damping_or_default(&self) -> f64 {
        some_nonzero(self.tract_damping).unwrap_or(if self.source == Source::Syrinx { 4.0 } else { 1.0 })
    }

    pub fn validate(&self) -> Result<(), String> {
        let pos = |name: &str, x: f64| if x > 0.0 && x.is_finite() { Ok(()) } else { Err(format!("body.{name} must be > 0 (got {x})")) };
        pos("mass", self.mass)?;
        pos("fold_scale", self.fold_scale)?;
        pos("lung_capacity", self.lung_capacity)?;
        pos("tract", self.tract())?;
        pos("carrier", self.carrier())?;
        pos("formant_width", self.formant_width)?;
        pos("open_quotient", self.open_quotient)?;
        Ok(())
    }
}

/// Per-intent overrides of Intent fields (a species' repertoire). `like` bases the intent on
/// another one's template.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntentOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub like: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contour: Option<Points>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amp: Option<Points>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harsh: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeats: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mouth: Option<Points>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub larynx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breath: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrato: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fry: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nasal: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inhale: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glide: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", deserialize_with = "opt_vowel_list")]
    pub vowels: Option<Vec<VowelSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrato_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rumble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subharmonics: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chaos: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biphonation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jumps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voiced: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drift: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accel: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhythm: Option<Points>,
    /// Sequential calls: per syllable, in order, [length x, pitch step (semitones), pitch slope
    /// across the syllable (semitones), level x]. Syllables past the list are plain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prosody: Option<Vec<[f64; 4]>>,
}

/// Heritable vocal traits, normally decoded from the genome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Voice {
    #[serde(deserialize_with = "vowel_list")]
    pub vowels: Vec<VowelSpec>,
    pub pitch: f64,
    pub pitch_range: f64,
    pub harshness: f64,
    pub vibrato: f64,
    pub tempo: f64,
    pub sustain: f64,
    pub melody: f64,
    pub falsetto: f64,
    pub repetition: f64,
    /// Nonlinear phenomena, 0..1 each: how prone this voice is to them (added to the intent's).
    /// Episodes of period doubling/tripling: lower, rougher, bigger-sounding.
    pub subharmonics: f64,
    /// Episodes of deterministic chaos: irregular but voiced (screams, snarls).
    pub chaos: f64,
    /// A second, independent tone over the voice, with sidebands (canids, deer).
    pub biphonation: f64,
    /// Its pitch relative to the voice's (not a whole number: independent).
    pub biphonation_ratio: f64,
    /// Abrupt register breaks: the pitch leaps up or down and back (yelps, meows).
    pub jumps: f64,
    pub calls: BTreeMap<String, IntentOverride>,
}

impl Default for Voice {
    fn default() -> Self {
        Voice {
            vowels: vec![v("a")],
            pitch: 0.0,
            pitch_range: 1.0,
            harshness: 0.0,
            vibrato: 0.0,
            tempo: 1.0,
            sustain: 1.0,
            melody: 1.0,
            falsetto: 0.0,
            repetition: 1.0,
            subharmonics: 0.0,
            chaos: 0.0,
            biphonation: 0.0,
            biphonation_ratio: 1.45,
            jumps: 0.0,
            calls: BTreeMap::new(),
        }
    }
}

impl Voice {
    pub fn validate(&self) -> Result<(), String> {
        validate_vowels(&self.vowels, "voice.vowels")?;
        if !(self.tempo > 0.0) {
            return Err(format!("voice.tempo must be > 0 (got {})", self.tempo));
        }
        for (name, o) in &self.calls {
            if !crate::synth::INTENT_NAMES.contains(&name.as_str()) {
                return Err(format!("voice.calls: unknown intent {name:?}"));
            }
            if let Some(like) = &o.like {
                if !crate::synth::INTENT_NAMES.contains(&like.as_str()) {
                    return Err(format!("voice.calls.{name}.like: unknown intent {like:?}"));
                }
            }
            if let Some(v) = &o.vowels {
                validate_vowels(v, &format!("voice.calls.{name}.vowels"))?;
            }
            if let Some(p) = &o.prosody {
                if let Some(x) = p.iter().find(|x| !(x[0] > 0.0) || !(x[3] >= 0.0) || x.iter().any(|v| !v.is_finite())) {
                    return Err(format!("voice.calls.{name}.prosody: length must be > 0 and level >= 0 (got {x:?})"));
                }
            }
            if let Some(r) = &o.rhythm {
                if let Some(&(_, m)) = r.iter().find(|&&(_, m)| !(m > 0.0)) {
                    return Err(format!("voice.calls.{name}.rhythm: spacing must be > 0 (got {m})"));
                }
            }
        }
        Ok(())
    }
}

/// Call template for one behaviour. Curves are (u, value) over normalized call time u.
#[derive(Clone, Debug)]
pub struct Intent {
    pub pitch: f64,
    pub contour: Points,
    pub amp: Points,
    pub harsh: f64,
    pub dur: f64,
    pub repeats: (f64, f64),
    pub gap: f64,
    pub mouth: Points,
    pub larynx: f64,
    pub breath: f64,
    pub vibrato: f64,
    pub fry: f64,
    pub nasal: f64,
    pub inhale: bool,
    pub glide: bool,
    /// Sequential: one syllable per vowel in the list (joined by "~" to glide, "|" to pause),
    /// and `repeats` says how many times the whole phrase is said.
    pub sequence: bool,
    pub spread: f64,
    pub vowels: Option<Vec<VowelSpec>>,
    pub vibrato_rate: f64,
    pub rumble: f64,
    /// Nonlinear phenomena this behaviour brings out, added to the voice's own.
    pub subharmonics: f64,
    pub chaos: f64,
    pub biphonation: f64,
    pub jumps: f64,
    /// 0 = unvoiced (a hiss) even from a voiced body.
    pub voiced: f64,
    /// Pitch change per repeat, semitones.
    pub drift: f64,
    /// Gap multiplier per repeat (below 1 speeds up).
    pub accel: f64,
    /// Spacing over the series: (position 0..1 from first repeat to last, multiplier of the gap).
    /// Notes shorten with the square root of it, the way a trill's notes do. Empty = steady.
    pub rhythm: Points,
    /// Per-syllable [length x, pitch step, pitch slope, level x] for sequential calls.
    pub prosody: Vec<[f64; 4]>,
}

impl Default for Intent {
    fn default() -> Self {
        Intent {
            pitch: 0.0,
            contour: vec![],
            amp: vec![],
            harsh: 0.0,
            dur: 0.5,
            repeats: (1.0, 1.0),
            gap: 0.3,
            mouth: vec![(0.0, 0.3), (0.3, 0.8), (1.0, 0.4)],
            larynx: 0.0,
            breath: 0.0,
            vibrato: 0.0,
            fry: 0.5,
            nasal: 0.0,
            inhale: false,
            glide: false,
            sequence: false,
            spread: 0.0,
            vowels: None,
            vibrato_rate: 5.5,
            rumble: 0.0,
            subharmonics: 0.0,
            chaos: 0.0,
            biphonation: 0.0,
            jumps: 0.0,
            voiced: 1.0,
            drift: 0.0,
            accel: 1.0,
            rhythm: vec![],
            prosody: vec![],
        }
    }
}

impl Intent {
    pub fn apply(mut self, o: &IntentOverride) -> Intent {
        macro_rules! set {
            ($($f:ident),*) => { $( if let Some(x) = &o.$f { self.$f = x.clone(); } )* };
        }
        set!(pitch, contour, amp, harsh, dur, repeats, gap, mouth, larynx, breath, vibrato, fry, nasal,
             inhale, glide, sequence, spread, vibrato_rate, rumble, subharmonics, chaos, biphonation, jumps,
             voiced, drift, accel, rhythm, prosody);
        if let Some(x) = &o.vowels {
            self.vowels = Some(x.clone());
        }
        self
    }
}

fn pts(p: &[(f64, f64)]) -> Points {
    p.to_vec()
}

/// The intent templates (territorial is a roar bout built in code). Morton's
/// motivation-structural rules: hostile calls low and harsh, fearful/appeasing calls high and tonal.
pub fn base_intent(name: &str) -> Option<Intent> {
    let d = Intent::default();
    Some(match name {
        // Soft contact call.
        "idle" => Intent {
            pitch: 0.0, contour: pts(&[(0.0, -1.0), (0.4, 1.5), (1.0, -2.0)]),
            amp: pts(&[(0.0, 0.0), (0.15, 0.6), (0.7, 0.5), (1.0, 0.0)]), harsh: 0.0, dur: 0.5,
            repeats: (1.0, 2.0), gap: 0.35, mouth: pts(&[(0.0, 0.2), (0.4, 0.6), (1.0, 0.3)]), nasal: 0.2, fry: 0.3,
            subharmonics: 0.1,
            ..d
        },
        // Threat growl: low, harsh, mouth barely open, larynx down to sound bigger.
        "warn" => Intent {
            pitch: -6.0, contour: pts(&[(0.0, -1.0), (0.3, 1.0), (0.8, 0.5), (1.0, -2.0)]),
            amp: pts(&[(0.0, 0.0), (0.2, 0.8), (0.8, 0.9), (1.0, 0.0)]), harsh: 0.8, dur: 1.5,
            repeats: (1.0, 2.0), gap: 0.4, mouth: pts(&[(0.0, 0.3), (0.5, 0.5), (1.0, 0.35)]),
            larynx: 1.0, breath: 0.1, inhale: true, fry: 0.4, spread: 1.0, subharmonics: 0.5, chaos: 0.3,
            ..d
        },
        // Alarm: short, sharp, high, repeated barks.
        "alarm" => Intent {
            pitch: 7.0, contour: pts(&[(0.0, -3.0), (0.2, 3.0), (1.0, -4.0)]),
            amp: pts(&[(0.0, 0.0), (0.05, 1.0), (0.4, 0.7), (1.0, 0.0)]), harsh: 0.35, dur: 0.18,
            repeats: (3.0, 5.0), gap: 0.18, mouth: pts(&[(0.0, 0.5), (0.2, 1.0), (1.0, 0.5)]), fry: 0.2,
            subharmonics: 0.3, jumps: 0.2,
            ..d
        },
        // Distress scream while fleeing: very high and chaotic.
        "flee" => Intent {
            pitch: 11.0, contour: pts(&[(0.0, 0.0), (0.15, 3.0), (0.4, 1.0), (0.6, 2.5), (1.0, -5.0)]),
            amp: pts(&[(0.0, 0.0), (0.08, 1.0), (0.7, 0.85), (1.0, 0.0)]), harsh: 0.7, dur: 0.8,
            repeats: (1.0, 3.0), gap: 0.25, mouth: pts(&[(0.0, 0.6), (0.15, 1.0), (1.0, 0.6)]),
            breath: 0.2, inhale: true, fry: 0.3, subharmonics: 0.4, chaos: 0.5, jumps: 0.4,
            ..d
        },
        // Advertisement call: long, melodic, tonal, gliding through the voice's vowels.
        "mating" => Intent {
            pitch: 2.0, contour: pts(&[(0.0, -2.0), (0.25, 3.0), (0.5, 1.0), (0.75, 4.0), (1.0, -3.0)]),
            amp: pts(&[(0.0, 0.0), (0.2, 0.8), (0.8, 0.85), (1.0, 0.0)]), harsh: 0.1, dur: 1.6,
            repeats: (2.0, 3.0), gap: 0.5, mouth: pts(&[(0.0, 0.3), (0.3, 0.9), (0.7, 0.7), (1.0, 0.3)]),
            vibrato: 0.4, inhale: true, glide: true, fry: 0.4, subharmonics: 0.1,
            ..d
        },
        // Yelp: instant onset, high, steep fall.
        "pain" => Intent {
            pitch: 11.0, contour: pts(&[(0.0, -2.0), (0.1, 4.0), (1.0, -9.0)]),
            amp: pts(&[(0.0, 0.0), (0.03, 1.0), (0.3, 0.7), (1.0, 0.0)]), harsh: 0.45, dur: 0.35,
            mouth: pts(&[(0.0, 0.6), (0.1, 1.0), (1.0, 0.3)]), fry: 0.6, subharmonics: 0.3, chaos: 0.3, jumps: 0.5,
            ..d
        },
        // Submissive whine: high, rising, nasal, clean.
        "beg" => Intent {
            pitch: 8.0, contour: pts(&[(0.0, -2.0), (0.6, 3.0), (1.0, 4.0)]),
            amp: pts(&[(0.0, 0.0), (0.3, 0.5), (0.85, 0.55), (1.0, 0.0)]), harsh: -0.3, dur: 0.8,
            repeats: (1.0, 3.0), gap: 0.3, mouth: pts(&[(0.0, 0.1), (1.0, 0.3)]), nasal: 0.5,
            vibrato: 0.3, glide: true, fry: 0.1, jumps: 0.2,
            ..d
        },
        // Vocalizing: every vowel in the list in turn, one syllable each ("~" glides, "|" pauses),
        // with a small rise-fall per syllable and the pitch sinking over the phrase.
        "speak" => Intent {
            pitch: 3.0, contour: pts(&[(0.0, -1.0), (0.3, 1.0), (1.0, -1.5)]),
            amp: pts(&[(0.0, 0.0), (0.12, 1.0), (0.75, 0.85), (1.0, 0.0)]), harsh: 0.0, dur: 0.3,
            repeats: (1.0, 1.0), gap: 0.08, mouth: pts(&[(0.0, 0.3), (0.25, 0.95), (0.8, 0.85), (1.0, 0.4)]),
            vibrato: 0.1, fry: 0.3, drift: -0.3, sequence: true,
            ..d
        },
        _ => return None,
    })
}

/// A complete creature description, as stored in JSON.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Creature {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub body: Body,
    pub voice: Voice,
}

impl Creature {
    pub fn from_json(json: &str) -> Result<Creature, String> {
        let c: Creature = serde_json::from_str(json).map_err(|e| format!("creature JSON: {e}"))?;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.body.validate()?;
        self.voice.validate()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Set one field by name to a JSON value. `name` is a Body or Voice field ("mass",
    /// "pitch", "vowels", "calls", ...) or a qualified path ("body.mass", "voice.calls").
    pub fn set_field(&mut self, name: &str, value: serde_json::Value) -> Result<(), String> {
        let mut doc = serde_json::to_value(&*self).map_err(|e| e.to_string())?;
        let (section, field) = match name.split_once('.') {
            Some((s, f)) if s == "body" || s == "voice" => (s.to_string(), f.to_string()),
            _ => {
                let body_fields = serde_json::to_value(Body::default()).map_err(|e| e.to_string())?;
                let in_body = body_fields.get(name).is_some() || name == "head_length" || name == "neck_length";
                let in_voice = serde_json::to_value(Voice::default()).map_err(|e| e.to_string())?.get(name).is_some();
                match (in_body, in_voice) {
                    (true, _) => ("body".to_string(), name.to_string()),
                    (_, true) => ("voice".to_string(), name.to_string()),
                    _ => return Err(format!("unknown field {name:?}")),
                }
            }
        };
        doc[&section][&field] = value;
        let c: Creature = serde_json::from_value(doc).map_err(|e| format!("{name}: {e}"))?;
        c.validate()?;
        *self = c;
        Ok(())
    }
}
