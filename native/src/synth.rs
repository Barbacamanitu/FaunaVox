//! The vocalization engine: source -> lossy vocal tube -> radiation -> mix.
//! A line-by-line port of create_vocalization.py; comments explain the acoustics.

use std::f64::consts::PI;

use crate::dsp::{self, curve, curve_at, interp, rms};
use crate::model::{base_intent, plain_vowels, syllables, Body, Consonant, Intent, Points, Source, Voice, VowelSpec};
use crate::rng::Rng;

pub const SR: usize = 44100;
const SRF: f64 = SR as f64;
const C_SOUND: f64 = 350.0; // speed of sound in warm, humid air (m/s)
// Breath noise RMS relative to the voice: floor + scale * breathiness * (0.5 + mean harshness).
const BREATH_FLOOR: f64 = 0.02;
const BREATH_SCALE: f64 = 0.26;
const GROWL_NOISE: f64 = 0.6; // extra noise at full harshness; NLP now carry most of the roughness
const NOISE_CORNER_MAX: f64 = 2000.0;
const INHALE_LEVEL: f64 = 0.08; // inhale RMS relative to the loudest call (about -22 dB)
// Clean-voice irregularity, at healthy-voice levels; harshness adds chaos on top.
const BASE_JITTER: f64 = 0.004;
const BASE_SHIMMER: f64 = 0.03;
const DRIFT_DEPTH: f64 = 0.012;
const DRIFT_TIME: f64 = 0.15;
const SHAPE_JITTER: f64 = 0.015;
const CHAOS_JITTER: f64 = 0.15;
const FRY_JITTER: f64 = 0.25;
const FRY_SHIMMER: f64 = 0.4;
const GLOTTAL_DAMPING: f64 = 0.5; // extra F1 bandwidth, as a fraction of f0, when f0 sits on F1
const TRACT_LOSS_LOWPASS: f64 = 0.5;
const CHEST_GAIN: f64 = 0.02;
const LIP_CUTOFF_HZ: (f64, f64) = (1700.0, 9000.0); // closed/rounded .. wide open, human-size tract
// Formant resonance on top of the tube. Real calls show formant peaks about twice as prominent
// as the bare tube gives, and a flat envelope with any noise in it reads as breathy.
pub const FORMANT_GAIN: f64 = 8.0;
const FORMANT_LEVELS: [f64; 5] = [1.0, 1.0, 0.9, 0.7, 0.5]; // relative boost of F1..F5
// Breath noise mostly happens while the glottis is open. Pulsed with each cycle it fuses into a
// rough voice; a steady floor between pulses is heard as a separate hiss layer.
pub const BREATH_PULSE: f64 = 0.9;
// Fraction of each glottal cycle the folds are open (clean voice). High = smooth, sine-like
// pulses with a dominant fundamental (soft, breathy); low = sharp pulses rich in upper harmonics.
pub const OPEN_QUOTIENT: f64 = 0.5;
// Nonlinear phenomena (NLP): subharmonics, deterministic chaos, biphonation and frequency jumps.
// Real voices hold a steady regime and switch abruptly between regimes in episodes; random
// cycle-to-cycle wobble instead reads as noise and breath.
const NLP_RATE: f64 = 2.5; // regime onsets per second at full trait and full loudness (more = stuttering)
const NLP_EPISODE: (f64, f64) = (0.15, 0.6); // mean episode length (s) at low .. full trait
const SUB_DEPTH: f64 = 0.6; // strength drop of the weak cycles in a subharmonic episode
const SUB_TIMING: f64 = 0.1; // period alternation in a subharmonic episode
const CHAOS_PERIOD: f64 = 0.12; // period swing driven by the chaotic map at full chaos
const CHAOS_AMP: f64 = 0.6; // amplitude swing driven by the chaotic map at full chaos
const BIPHONATION_COUPLING: f64 = 0.3; // how strongly the second source modulates the first; high = ring-mod robot

/// How far sequential syllables' loudness is evened out (0..1): rounded vowels radiate much less.
const SEQUENCE_EVEN: f64 = 0.7;

/// A consonant's length (s, for a 50 kg animal; it grows only with the square root of size) and
/// its loudness relative to the loudest syllable (dB, RMS over the consonant). Kept well under
/// measured speech power: noise stands out from a creature's voice far more than from speech, and a
/// faint consonant is better than a loud one. The order follows Wilde (1995, table 4.1): sh 5-10 dB
/// above s, and f and th about equal, far below both. Listeners reject an s even 4 dB too loud but
/// accept f anywhere within 4 dB.
fn consonant_shape(c: Consonant) -> (f64, f64) {
    match c {
        Consonant::H => (0.08, -20.0),
        Consonant::S => (0.13, -27.0),
        Consonant::Sh => (0.13, -22.0),
        Consonant::F => (0.11, -29.0),
        Consonant::Th => (0.11, -29.0),
        // Affricates: a short burst, quieter than a full sh.
        Consonant::Ch => (0.07, -24.0),
        Consonant::Jh => (0.065, -27.0),
        // Stops: the closure (silence), and the burst's level. The burst is a click of a few ms, so
        // its RMS can sit near a fricative's and still be heard. k's is the strongest and longest,
        // p's the weakest; voiced ones are about 5 dB weaker than voiceless (Klatt 1980).
        Consonant::P => (0.065, -27.0),
        Consonant::T => (0.05, -23.0),
        Consonant::K => (0.055, -22.0),
        Consonant::B => (0.055, -31.0),
        Consonant::D => (0.04, -28.0),
        Consonant::G => (0.045, -27.0),
    }
}

/// A stop's burst length and its voice onset time: how long after the release the voice starts,
/// with breath through the next vowel's formants in between (s, for a 50 kg animal; grows with the
/// square root of size). Klatt (1979, 1980): bursts of 6, 8 and 18 ms for p, t and k; voiceless
/// stops 45-60 ms before the voice, voiced ones at once.
fn stop_timing(c: Consonant) -> (f64, f64) {
    match c {
        Consonant::P => (0.006, 0.04),
        Consonant::T => (0.008, 0.05),
        Consonant::K => (0.016, 0.055),
        Consonant::B => (0.005, 0.0),
        Consonant::D => (0.007, 0.0),
        _ => (0.014, 0.0), // g
    }
}

/// The breath after a voiceless stop's burst, before the voice starts (dB re the loudest
/// syllable). After a vowel (at the end of a syllable) the release is weaker and shorter: this much
/// quieter, and this share of the onset's breath.
const STOP_ASPIRATION: f64 = -24.0;
const CODA_RELEASE: (f64, f64) = (-6.0, 0.4);

/// Voicing heard through the closure of b, d or g between vowels: the folds keep vibrating while
/// the walls of the closed tract pass only the lowest harmonics. Its level (dB re the loudest
/// syllable; Klattara's voice bar is 16 dB under a stressed vowel) and its lowpass (Hz, human size).
const VOICE_BAR: (f64, f64) = (-22.0, 450.0);

/// s, sh, ch and j: noise from the airstream hitting the teeth, much stronger than f and th.
fn strident(c: Consonant) -> bool {
    matches!(c, Consonant::S | Consonant::Sh | Consonant::Ch | Consonant::Jh)
}

/// How much weaker a strident is next to an open back vowel ("sa" against "se"), dB at full
/// openness and backness (Wilde 1995: s 8-13 dB weaker before /a/, about 3 dB of it the vowel's F1).
const STRIDENT_OPEN_BACK: f64 = -7.0;

fn open_back(v: &VowelSpec) -> f64 {
    let r = v.ratios();
    clip((r[0] - 1.0) / 0.36, 0.0, 1.0) * clip((1.15 - r[1]) / 0.4, 0.0, 1.0)
}

/// The turbulence source falls off at 6 dB/octave above a corner that rises with the airflow, so a
/// quiet fricative is darker as well as softer (Stevens; Wilde 1995, fig. 2.6). The corner (Hz,
/// human size) for a fricative at the given level (dB re the loudest syllable), and the dB per
/// octave it moves. Gentler than the source alone: lip radiation (+6 dB/octave) undoes most of the
/// source's slope, which is why f and th measure nearly flat.
const SOURCE_CORNER: (f64, f64, f64) = (4500.0, -10.0, 16.0);

/// A fricative's noise rises and falls in about this long (s, for a 50 kg animal; it grows with the
/// square root of size): a flat plateau with short ramps, nothing extra at the edges, which
/// listeners hear as unnatural (Wilde 1995). An affricate's release bursts out faster.
const FRICATIVE_RAMP: (f64, f64) = (0.025, 0.012);

/// The near-silent moment between a fricative and the vowel it releases into: the noise stops
/// before the voice starts (s, for a 50 kg animal; grows with the square root of size).
const RELEASE_DIP: f64 = 0.012;

/// How fast a vowel after a fricative starts: the attack of its loudness curve is squeezed by this
/// much. The voice switches on abruptly after a voiceless consonant.
const ONSET_SHARPEN: f64 = 0.2;

/// Where F1 to F3 point at each fricative's place of articulation (as ratios of the neutral tube's
/// 500, 1500 and 2500 Hz, so they scale with size), and how far the vowel pulls each one at the
/// boundary: 0 keeps the locus, 1 takes the vowel's own value. F1 drops toward 300 Hz at every
/// constriction, the plainest cue that the tract closed in. These loci are wider than natural locus
/// equations: synthetic speech needs its place cues exaggerated (Klattara, after DECtalk). Next to
/// a front vowel, k and g close further forward and their F2 and F3 meet near 2300 Hz ("key"
/// against "coo").
fn locus(c: Consonant, vowel: &VowelSpec) -> Option<([f64; 3], [f64; 3])> {
    match c {
        Consonant::H => None,
        Consonant::F => Some(([0.6, 1100.0 / 1500.0, 2200.0 / 2500.0], [0.35, 0.7, 0.7])),
        Consonant::Th => Some(([0.6, 1.0, 2600.0 / 2500.0], [0.35, 0.6, 0.6])),
        Consonant::S => Some(([0.5, 1.2, 2700.0 / 2500.0], [0.3, 0.5, 0.5])),
        Consonant::Sh | Consonant::Ch | Consonant::Jh => Some(([0.56, 2000.0 / 1500.0, 2650.0 / 2500.0], [0.3, 0.55, 0.55])),
        Consonant::P | Consonant::B => Some(([0.5, 900.0 / 1500.0, 2100.0 / 2500.0], [0.3, 0.6, 0.6])),
        Consonant::T | Consonant::D => Some(([0.5, 1.2, 2700.0 / 2500.0], [0.3, 0.5, 0.5])),
        Consonant::K | Consonant::G if front_vowel(vowel) => Some(([0.5, 2250.0 / 1500.0, 2350.0 / 2500.0], [0.3, 0.3, 0.3])),
        Consonant::K | Consonant::G => Some(([0.5, 2000.0 / 1500.0, 2100.0 / 2500.0], [0.3, 0.5, 0.3])),
    }
}

/// A front vowel (i, e, ae, and the fronted SSBE u:): F2 above the neutral tube's.
fn front_vowel(v: &VowelSpec) -> bool {
    v.ratios()[1] > 1.03
}

/// How much of each formant's pull toward the vowel is kept at the boundary. Half leaves the
/// formants farther from the vowel, so they bend more sharply into each consonant and mark its place
/// more clearly (Klattara: final consonants told apart 95% -> 97% of the time).
const LOCUS_PULL: f64 = 0.5;

/// How long the vowel's F2 takes to move from the consonant's locus to its own target (s, for a
/// 50 kg animal; grows with the square root of size).
const FORMANT_TRANSITION: f64 = 0.06;



/// The breath that joins a fricative to its vowel: as the tongue or lips release, the airstream
/// passes through the opening tract and takes on the vowel's formants. Its level against the
/// fricative's (dB), and its length as a fraction of the fricative's into a vowel and out of one:
/// the glottis closes slowly after a fricative, so the breath runs on into the next vowel, but opens
/// quickly before one (Wilde 1995).
const ASPIRATION: (f64, f64, f64) = (-12.0, 0.6, 0.3);

/// Breath from the open glottis has a weak F1: the glottis damps it heavily, about 8-10 dB under
/// F2 and F3 (Wilde 1995). An h is highpassed at this multiple of its vowel's F1.
const ASPIRATION_F1_CUT: f64 = 1.5;

/// How much of a consonant overlaps the vowel it leads into or trails from.
const CONSONANT_OVERLAP: f64 = 0.25;

pub const INTENT_NAMES: [&str; 9] = ["idle", "warn", "alarm", "flee", "mating", "pain", "beg", "territorial", "speak"];

#[derive(Clone, Debug)]
pub struct Call {
    pub start: f64,
    pub dur: f64,
    pub pitch: Points, // semitones relative to the body's base f0
    pub amp: Points,
    pub harsh: Points,
    pub mouth: Points,
    pub vowels: Vec<VowelSpec>,
    pub larynx: Option<Points>,
    pub voiced: f64,
    pub breath: f64,
    pub fry: f64,
    pub vibrato: f64,
    pub nasal: f64,
    pub falsetto: f64,
    pub spread: f64,
    pub vibrato_rate: f64,
    pub rumble: f64,
    pub subharmonics: f64,
    pub chaos: f64,
    pub biphonation: f64,
    pub biphonation_ratio: f64,
    pub jumps: f64,
    /// Sequential syllables: how far their loudness is evened out after rendering (0 = as
    /// rendered, 1 = all equal), so quiet vowels (rounded u, o) aren't lost; `level` is the
    /// loudness the syllable is meant to have relative to the others.
    pub even: f64,
    pub level: f64,
    /// A consonant in a sequential call: rendered as noise and set against the syllables, at
    /// `consonant_db` relative to the loudest syllable.
    pub consonant: Option<Consonant>,
    pub consonant_db: f64,
    /// A syllable next to a consonant: F1 to F3 move from the consonant's locus (ratios, and the
    /// vowel's pull on each) to the vowel's own over the first (from) or last (to) fraction of the
    /// call.
    pub formants_from: Option<([f64; 3], [f64; 3], f64)>,
    pub formants_to: Option<([f64; 3], [f64; 3], f64)>,
    /// The voicing heard through a voiced stop's closure: rendered as a call, then lowpassed.
    pub voice_bar: bool,
    /// A stop's burst: how long the tract was closed before it (s). Only the mouth track uses it.
    pub closure: f64,
}

impl Call {
    fn new(start: f64, dur: f64, pitch: Points, amp: Points, harsh: Points, mouth: Points, vowels: Vec<VowelSpec>) -> Call {
        Call {
            start, dur, pitch, amp, harsh, mouth, vowels,
            larynx: None, voiced: 1.0, breath: 0.0, fry: 0.5, vibrato: 0.0, nasal: 0.0,
            falsetto: 0.0, spread: 0.0, vibrato_rate: 5.5, rumble: 0.0,
            subharmonics: 0.0, chaos: 0.0, biphonation: 0.0, biphonation_ratio: 1.45, jumps: 0.0,
            even: 0.0, level: 1.0, consonant: None, consonant_db: 0.0, formants_from: None, formants_to: None, voice_bar: false,
            closure: 0.0,
        }
    }

    /// Set the call's nonlinear-phenomena amounts: the voice's proneness plus the behaviour's, 0..1.
    fn nlp(mut self, voice: &Voice, sub: f64, chaos: f64, biph: f64, jumps: f64) -> Call {
        self.subharmonics = clip(voice.subharmonics + sub, 0.0, 1.0);
        self.chaos = clip(voice.chaos + chaos, 0.0, 1.0);
        self.biphonation = clip(voice.biphonation + biph, 0.0, 1.0);
        self.jumps = clip(voice.jumps + jumps, 0.0, 1.0);
        self.biphonation_ratio = voice.biphonation_ratio;
        self
    }
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn round_half_even(x: f64) -> f64 {
    // Python's round().
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Blend an intent's pitch contour toward a single swell: rise to its peak, hold with a gentle
/// sag, fall to its end. A melodic up-down contour on a pure tone reads as an instrument.
fn shape_contour(contour: &Points, melody: f64) -> Points {
    let u: Vec<f64> = (0..17).map(|i| i as f64 / 16.0).collect();
    let full = curve(contour, &u);
    let first = full[0];
    let last = full[16];
    let peak = full.iter().cloned().fold(f64::MIN, f64::max);
    let swell = curve(&vec![(0.0, first), (0.12, peak), (0.85, peak - 1.0), (1.0, last)], &u);
    u.iter().enumerate().map(|(k, &x)| (x, melody * full[k] + (1.0 - melody) * swell[k])).collect()
}

// ---------------------------------------------------------------- call structure

fn inhale_call(start: f64, dur: f64) -> Call {
    let mut c = Call::new(
        start, dur,
        vec![(0.0, 0.0), (1.0, 0.0)],
        vec![(0.0, 0.0), (0.5, 0.18), (1.0, 0.0)],
        vec![(0.0, 0.4), (1.0, 0.4)],
        vec![(0.0, 0.5), (1.0, 0.6)],
        vec![VowelSpec::Name("@".into())],
    );
    c.voiced = 0.0;
    c.breath = 0.5;
    c
}

pub fn build_calls(body: &Body, voice: &Voice, intent_name: &str, rng: &mut Rng) -> Result<Vec<Call>, String> {
    // A species override may base an intent on another one ("like"), e.g. a frog's territorial
    // call is its croak chorus rather than a roar bout.
    let overrides = voice.calls.get(intent_name).cloned().unwrap_or_default();
    let base_name = overrides.like.clone().unwrap_or_else(|| intent_name.to_string());
    if base_name == "territorial" {
        return Ok(territorial_bout(body, voice, rng));
    }
    let it: Intent = base_intent(&base_name).ok_or_else(|| format!("unknown intent {base_name:?}"))?.apply(&overrides);
    if it.sequence {
        return Ok(sequence_calls(body, voice, &it, rng));
    }
    let scale = body.call_scale() / voice.tempo;
    let base_harsh = 0.5 * body.roughness + voice.harshness;
    let lo = (round_half_even(it.repeats.0 * voice.repetition) as i64).max(1);
    let hi = (round_half_even(it.repeats.1 * voice.repetition) as i64).max(1);
    let vowels = plain_vowels(it.vowels.as_ref().unwrap_or(&voice.vowels));
    let count = rng.integer_inclusive(lo, hi);
    let mut calls = Vec::new();
    let mut t = 0.0;
    let mut gap = it.gap;
    for i in 0..count.max(0) as usize {
        let spacing = rhythm_at(&it.rhythm, i, count.max(1) as usize);
        if it.inhale {
            calls.push(inhale_call(t, 0.25 * scale));
            t += 0.3 * scale;
        }
        // Sustain and melody shape only the long tonal calls (howls, songs, whines), not barks.
        let tonal = it.glide;
        let dur = it.dur * scale * (if tonal { voice.sustain } else { 1.0 }) * (1.0 + 0.1 * rng.standard_normal())
            * spacing.sqrt();
        let dur = dur.max(0.01);
        let contour = if tonal { shape_contour(&it.contour, voice.melody) } else { it.contour.clone() };
        let shift = it.pitch + voice.pitch + it.drift * i as f64 + rng.normal(0.0, 0.7);
        let level = 0.85_f64.powi(i as i32);
        let c = Call::new(
            t, dur,
            contour.iter().map(|&(u, s)| (u, shift + s * voice.pitch_range)).collect(),
            it.amp.iter().map(|&(u, a)| (u, a * level)).collect(),
            it.amp.iter().map(|&(u, a)| (u, base_harsh + it.harsh * a)).collect(), // harshest when loudest
            it.mouth.clone(),
            if it.glide { vowels.clone() } else { vec![vowels[i % vowels.len()].clone()] },
        );
        calls.push(dress(c, &it, voice));
        t += dur + gap * spacing * scale * rng.uniform(0.8, 1.2);
        gap *= it.accel;
    }
    Ok(calls)
}

/// The rhythm curve's spacing for repeat i of n (1 with no curve).
fn rhythm_at(rhythm: &Points, i: usize, n: usize) -> f64 {
    if rhythm.is_empty() {
        return 1.0;
    }
    let pos = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
    curve_at(rhythm, pos).max(0.01)
}

/// The rest of a call's settings from its intent and voice, then its nonlinear phenomena.
fn dress(mut c: Call, it: &Intent, voice: &Voice) -> Call {
    c.larynx = Some(it.amp.iter().map(|&(u, a)| (u, it.larynx * a)).collect());
    c.breath = it.breath;
    c.fry = it.fry;
    c.vibrato = it.vibrato + voice.vibrato;
    c.nasal = it.nasal;
    c.falsetto = voice.falsetto;
    c.spread = it.spread;
    c.vibrato_rate = it.vibrato_rate;
    c.rumble = it.rumble;
    c.voiced = it.voiced;
    c.nlp(voice, it.subharmonics, it.chaos, it.biphonation, it.jumps)
}

/// A sequential call: one syllable per vowel in the list, in order, repeats included. Vowels
/// joined by "~" make one longer syllable gliding through them; "|" adds a pause. `repeats` is how
/// many times the whole phrase is said. Syllables keep their level (no fading as in a bark
/// series) and the pitch moves by `drift` per syllable.
fn sequence_calls(body: &Body, voice: &Voice, it: &Intent, rng: &mut Rng) -> Vec<Call> {
    let scale = body.call_scale() / voice.tempo;
    let base_harsh = 0.5 * body.roughness + voice.harshness;
    let syls = syllables(it.vowels.as_ref().unwrap_or(&voice.vowels));
    let n = syls.len();
    let lo = (round_half_even(it.repeats.0) as i64).max(1);
    let hi = (round_half_even(it.repeats.1) as i64).max(lo);
    let phrases = rng.integer_inclusive(lo, hi);
    let pause = (0.25_f64).max(3.0 * it.gap) * scale;
    let mut calls = Vec::new();
    let mut t = 0.0;
    for p in 0..phrases.max(1) as usize {
        if p > 0 {
            t += pause;
        }
        if it.inhale {
            calls.push(inhale_call(t, 0.25 * scale));
            t += 0.3 * scale;
        }
        let mut gap = it.gap;
        for (j, syl) in syls.iter().enumerate() {
            let k = syl.vowels.len();
            let tonal = k > 1;
            let spacing = rhythm_at(&it.rhythm, j, n);
            // This syllable's own length, pitch step, slope and level (plain without prosody).
            let [p_len, p_pitch, p_slope, p_level] = it.prosody.get(j).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
            let dur = it.dur * scale * (1.0 + 0.7 * (k - 1) as f64) * (if tonal { voice.sustain } else { 1.0 })
                * (1.0 + 0.1 * rng.standard_normal()) * spacing.sqrt() * p_len;
            let dur = dur.max(0.01);
            let mut contour = if tonal { shape_contour(&it.contour, voice.melody) } else { it.contour.clone() };
            if p_slope != 0.0 {
                if contour.is_empty() {
                    contour = vec![(0.0, 0.0), (1.0, 0.0)];
                }
                // in semitones, so divided back out of the pitch range the contour is scaled by
                let pr = voice.pitch_range.max(1e-3);
                contour.iter_mut().for_each(|(u, s)| *s += p_slope * (*u - 0.5) / pr);
            }
            let shift = it.pitch + voice.pitch + it.drift * j as f64 + rng.normal(0.0, 0.4) + p_pitch;
            let level = (1.0 - 0.2 * j as f64 / (n.max(2) - 1) as f64) * p_level;
            // Consonants before the vowel run into it.
            for (m, &cons) in syl.onset.iter().enumerate() {
                if cons.is_stop() {
                    // Voiced through the closure when the voice runs straight into it.
                    let voice_in = m == 0 && j > 0 && !syls[j - 1].pause && syls[j - 1].coda.is_empty();
                    let released = !syl.onset.get(m + 1).is_some_and(|c| c.is_stop());
                    t = push_stop(&mut calls, cons, t, scale, shift, &syl.vowels[0], voice_in, true, released);
                    continue;
                }
                let c = consonant_call(cons, t, scale, shift, &syl.vowels[0], true);
                let last = m + 1 == syl.onset.len();
                if last && cons != Consonant::H {
                    // Released into the vowel: a brief dip, then the voice starts, with the breath
                    // running on into it through its formants.
                    calls.push(aspiration_call(&c, t + c.dur - 0.1 * ASPIRATION.1 * c.dur, &syl.vowels[0], true));
                    t += c.dur + RELEASE_DIP * scale.sqrt();
                } else {
                    t += c.dur * if last { 1.0 - CONSONANT_OVERLAP } else { 1.0 };
                }
                calls.push(c);
            }
            let c = Call::new(
                t, dur,
                contour.iter().map(|&(u, s)| (u, shift + s * voice.pitch_range)).collect(),
                it.amp.iter().map(|&(u, a)| (u, a * level)).collect(),
                it.amp.iter().map(|&(u, a)| (u, base_harsh + it.harsh * a)).collect(),
                it.mouth.clone(),
                syl.vowels.clone(),
            );
            let mut c = dress(c, it, voice);
            c.even = SEQUENCE_EVEN;
            c.level = level;
            // Next to a consonant the voice starts abruptly, and the formants move from or to the
            // place of the consonant: the one closing this syllable, or else the one opening the
            // next when the voice runs straight into it.
            let transition = (FORMANT_TRANSITION * scale.sqrt() / dur).min(0.45);
            let last = &syl.vowels[syl.vowels.len() - 1];
            if let Some(&cons) = syl.onset.last().filter(|&&c| c != Consonant::H) {
                c.amp = sharpen_onset(&c.amp);
                c.formants_from = transition_to(cons, &syl.vowels[0], transition);
            }
            let next_onset = if syl.pause { None } else { syls.get(j + 1).and_then(|s| s.onset.first()) };
            if let Some(&cons) = syl.coda.first().or(next_onset).filter(|&&c| c != Consonant::H) {
                c.formants_to = transition_to(cons, last, transition);
            }
            calls.push(c);
            t += dur;
            // Consonants after it start as it fades.
            for (m, &cons) in syl.coda.iter().enumerate() {
                if cons.is_stop() {
                    let released = !syl.coda.get(m + 1).is_some_and(|c| c.is_stop());
                    t = push_stop(&mut calls, cons, t, scale, shift, last, m == 0, false, released);
                    continue;
                }
                let start = if cons.is_affricate() {
                    t // bursts out as the vowel ends
                } else if m == 0 {
                    t - CONSONANT_OVERLAP * consonant_shape(cons).0 * scale.sqrt()
                } else {
                    t
                };
                let c = consonant_call(cons, start, scale, shift, last, false);
                if m == 0 && cons != Consonant::H && !cons.is_affricate() {
                    // the vowel's breath narrowing into the fricative
                    calls.push(aspiration_call(&c, start + CONSONANT_OVERLAP * c.dur - 0.5 * ASPIRATION.2 * c.dur, last, false));
                }
                t = start + c.dur;
                calls.push(c);
            }
            // No gap before a syllable that starts with a consonant: the consonant runs on from this one.
            let run_on = !syl.pause && syls.get(j + 1).is_some_and(|s| !s.onset.is_empty());
            t += if run_on { 0.0 } else { 1.0 } * gap * spacing * scale * rng.uniform(0.8, 1.2) + if syl.pause { pause } else { 0.0 };
            gap *= it.accel;
        }
    }
    calls
}

/// A consonant of a sequential call at `start`. An h is breath through the tract shaped like the
/// vowel next to it; it swells into a vowel it leads, and fades after one it trails.
fn consonant_call(cons: Consonant, start: f64, scale: f64, pitch: f64, vowel: &VowelSpec, leads: bool) -> Call {
    let dur = consonant_shape(cons).0 * scale.sqrt();
    let amp = match (cons, leads) {
        (Consonant::H, true) => vec![(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
        (Consonant::H, false) => vec![(0.0, 1.0), (0.4, 0.7), (1.0, 0.0)],
        _ => vec![(0.0, 0.0), (0.15, 1.0), (0.85, 1.0), (1.0, 0.0)],
    };
    let mut c = Call::new(
        start, dur,
        vec![(0.0, pitch), (1.0, pitch)],
        amp,
        vec![(0.0, 0.0), (1.0, 0.0)],
        vec![(0.0, 0.7), (1.0, 0.8)],
        vec![vowel.clone()],
    );
    c.voiced = 0.0;
    c.breath = 1.0;
    c.consonant = Some(cons);
    c.consonant_db = consonant_shape(cons).1 + if strident(cons) { STRIDENT_OPEN_BACK * open_back(vowel) } else { 0.0 };
    c
}

/// A stop from `start`: the closure (silent, or voiced when `voice_in` and the stop is b, d or g),
/// the burst, and for p, t and k the breath through `vowel`'s formants until the voice starts.
/// `onset`: released into the vowel after it, rather than closing the one before (weaker and
/// shorter). A stop right before another is not `released`: its closure runs on into the next
/// one's, as in "big gate". Returns when the next sound starts.
#[allow(clippy::too_many_arguments)]
fn push_stop(
    calls: &mut Vec<Call>, cons: Consonant, start: f64, scale: f64, pitch: f64, vowel: &VowelSpec, voice_in: bool, onset: bool,
    released: bool,
) -> f64 {
    let k = scale.sqrt();
    let (closure, burst_db) = consonant_shape(cons);
    let (burst, vot) = stop_timing(cons);
    let closure = closure * k;
    if voice_in && cons.is_voiced() {
        let mut bar = Call::new(
            start, closure,
            vec![(0.0, pitch), (1.0, pitch - 1.0)],
            vec![(0.0, 1.0), (0.6, 0.8), (1.0, 0.5)],
            vec![(0.0, 0.0), (1.0, 0.0)],
            vec![(0.0, 0.1), (1.0, 0.1)],
            vec![vowel.clone()],
        );
        bar.consonant = Some(cons);
        bar.consonant_db = VOICE_BAR.0;
        bar.voice_bar = true;
        calls.push(bar);
    }
    let mut t = start + closure;
    if !released {
        return t;
    }
    let release = if onset { 0.0 } else { CODA_RELEASE.0 };
    let mut b = consonant_call(cons, t, scale, pitch, vowel, onset);
    b.dur = burst * k;
    b.closure = closure;
    b.consonant_db = burst_db + release;
    t += b.dur;
    calls.push(b);
    if vot > 0.0 {
        let mut a = consonant_call(Consonant::H, t, 1.0, pitch, vowel, onset);
        a.dur = ((vot - burst) * k).max(0.005) * if onset { 1.0 } else { CODA_RELEASE.1 };
        a.amp = if onset {
            vec![(0.0, 0.6), (0.15, 1.0), (0.7, 0.8), (1.0, 0.25)]
        } else {
            vec![(0.0, 0.6), (0.15, 1.0), (1.0, 0.0)]
        };
        a.consonant_db = STOP_ASPIRATION + release;
        t += a.dur;
        calls.push(a);
    }
    t
}

/// A loudness curve whose attack (up to its first peak) is squeezed by ONSET_SHARPEN.
fn sharpen_onset(amp: &Points) -> Points {
    let peak = amp.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    let k = amp.iter().position(|p| p.1 >= peak).unwrap_or(0);
    amp.iter().enumerate().map(|(i, &(u, a))| (if i <= k { u * ONSET_SHARPEN } else { u }, a)).collect()
}

/// Formant k (as a ratio of the neutral tube's) at position u of a call, moved towards a
/// neighbouring fricative's locus at the call's start or end.
fn formant_transition(call: &Call, k: usize, u: f64, r: f64) -> f64 {
    let toward = |loci: [f64; 3], pull: [f64; 3], r: f64, w: f64| {
        let edge = loci[k] + LOCUS_PULL * pull[k] * (r - loci[k]);
        edge * (1.0 - w) + r * w
    };
    let mut r = r;
    if let Some((loci, pull, len)) = call.formants_from {
        r = toward(loci, pull, r, smoothstep(u / len));
    }
    if let Some((loci, pull, len)) = call.formants_to {
        r = toward(loci, pull, r, smoothstep((1.0 - u) / len));
    }
    r
}

/// A consonant's locus next to `vowel`, with the transition's length, for `formants_from`/`_to`.
fn transition_to(cons: Consonant, vowel: &VowelSpec, len: f64) -> Option<([f64; 3], [f64; 3], f64)> {
    locus(cons, vowel).map(|(l, p)| (l, p, len))
}

/// The breath between a fricative and its vowel (see ASPIRATION), starting at `start`: an h through
/// the vowel's tract, loudest at the boundary.
fn aspiration_call(fricative: &Call, start: f64, vowel: &VowelSpec, leads: bool) -> Call {
    let mut c = consonant_call(Consonant::H, start, 1.0, fricative.pitch[0].1, vowel, leads);
    c.dur = if leads { ASPIRATION.1 } else { ASPIRATION.2 } * fricative.dur;
    c.amp = if leads {
        vec![(0.0, 0.0), (0.3, 1.0), (1.0, 0.0)]
    } else {
        vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
    };
    c.consonant_db = fricative.consonant_db + ASPIRATION.0;
    c
}

/// 0 -> 1 with zero slope at both ends.
fn smoothstep(x: f64) -> f64 {
    let x = clip(x, 0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// A fricative: turbulence where the airstream hits the teeth or lips, coloured by the small
/// cavity in front of the constriction rather than by the whole tract. So its colour depends on
/// the size of the mouth (smaller hisses higher) and on lip rounding from the vowel beside it,
/// which lengthens that cavity and lowers the hiss ("sue" against "see").
/// - s: a peak near 4.5 kHz (human size, the front cavity's quarter wave) and a weaker one near
///   6.5 kHz, steeply cut below 3 kHz;
/// - sh: a lower peak near 2.7 kHz with a weaker second one, cut below 2 kHz;
/// - f, th: flat and faint (no cavity in front of the lips or teeth); th a little brighter;
/// - ch, j: the sh hiss, short, bursting out and dying away.
/// The quieter it is, the darker (SOURCE_CORNER). It holds steady between short ramps, and the
/// turbulence flutters a little.
fn render_fricative(call: &Call, cons: Consonant, body: &Body, rng: &mut Rng) -> Vec<f64> {
    let n = ((call.dur * SRF) as usize).max(16);
    let size = (0.17 / body.tract()).sqrt();
    let rounding = call.vowels.first().map(|v| v.rounding()).unwrap_or(0.0);
    let lower = 1.0 - 0.15 * rounding;
    let lim = 0.45 * SRF;
    let hz = |f: f64| (f * size).min(lim);
    let noise = rng.normals(n);
    let unit = |x: Vec<f64>| -> Vec<f64> {
        let r = rms(&x) + 1e-12;
        x.into_iter().map(|v| v / r).collect()
    };
    // Two second-order highpasses in a row: 24 dB/octave below the cutoff.
    let steep = |fc: f64| [dsp::butter_highpass(2, fc), dsp::butter_highpass(2, fc)];
    // (cutoff, [(peak Hz, width Hz, weight)], broadband floor weight, floor lowpass)
    let (cut, peaks, floor, top): (f64, &[(f64, f64, f64)], f64, f64) = match cons {
        Consonant::S => (3000.0 * lower, &[(4500.0, 1000.0, 1.0), (6500.0, 1500.0, 0.3)], 0.1, 9000.0),
        Consonant::Sh | Consonant::Ch | Consonant::Jh => (2000.0 * lower, &[(2700.0, 900.0, 1.0), (4800.0, 1800.0, 0.3)], 0.1, 6500.0),
        Consonant::Th => (1300.0, &[], 1.0, 9000.0),
        _ => (1000.0, &[], 1.0, 8000.0), // f
    };
    let top = hz(top);
    let cut = hz(cut).min(0.5 * top);
    let mut y = vec![0.0; n];
    let corner = hz(SOURCE_CORNER.0 * 2f64.powf((call.consonant_db - SOURCE_CORNER.1) / SOURCE_CORNER.2));
    let source = dsp::sosfilt(&[dsp::butter_lowpass(1, corner)], &noise);
    // The top is rolled off at 12 dB/octave: the air above it is what makes a hiss harsh.
    let base = dsp::sosfilt(&[dsp::butter_lowpass(2, top)], &dsp::sosfilt(&steep(cut), &source));
    let broad = unit(base.clone());
    for (o, b) in y.iter_mut().zip(&broad) {
        *o += floor * b;
    }
    // The peaks are added in phase. Klatt (1980) flips every other parallel resonator to fill the
    // dip between peaks, but with a weak second peak that cancels it and everything above it.
    for &(f, w, weight) in peaks {
        let ring = unit(dsp::sosfilt(&[dsp::resonator(hz(f * lower).min(0.9 * top), hz(w))], &base));
        for (o, r) in y.iter_mut().zip(&ring) {
            *o += weight * r;
        }
    }
    // Short ramps in and out (an affricate bursts out and dies away), and a slow flutter of the
    // turbulence (about 30 Hz and below).
    let shape = consonant_shape(cons).0;
    let (rise, fall) = if cons.is_affricate() {
        (FRICATIVE_RAMP.1 / shape, 0.6)
    } else {
        (FRICATIVE_RAMP.0 / shape, FRICATIVE_RAMP.0 / shape)
    };
    let (rise, fall) = (rise.min(0.5), fall.min(0.6));
    let flutter = dsp::sosfilt(&[dsp::butter_lowpass(2, 30.0)], &rng.normals(n));
    let fs = dsp::std(&flutter) + 1e-12;
    (0..n)
        .map(|i| {
            let u = i as f64 / n as f64;
            let env = smoothstep(u / rise) * smoothstep((1.0 - u) / fall);
            y[i] * env * (0.15 * flutter[i] / fs).exp()
        })
        .collect()
}

/// A stop's burst: the click of air as the closure opens, a few ms long, rising at once and dying
/// away. Its spectrum marks the place (Klatt 1980; Stevens & Blumstein 1978):
/// - p, b: diffuse and falling, nothing in front of the lips to shape it;
/// - t, d: high and rising, from the small cavity behind the teeth (a peak near 4.5 kHz);
/// - k, g: one compact peak near the next vowel's F2, where the tongue meets the palate (higher
///   before front vowels).
/// Sizes scale like a fricative's, and a quieter burst is darker (SOURCE_CORNER).
fn render_burst(call: &Call, cons: Consonant, body: &Body, rng: &mut Rng) -> Vec<f64> {
    let n = ((call.dur * SRF) as usize).max(16);
    let size = (0.17 / body.tract()).sqrt();
    let lim = 0.45 * SRF;
    let hz = |f: f64| (f * size).min(lim);
    let unit = |x: Vec<f64>| -> Vec<f64> {
        let r = rms(&x) + 1e-12;
        x.into_iter().map(|v| v / r).collect()
    };
    let corner = hz(SOURCE_CORNER.0 * 2f64.powf((call.consonant_db - SOURCE_CORNER.1) / SOURCE_CORNER.2));
    let source = dsp::sosfilt(&[dsp::butter_lowpass(1, corner)], &rng.normals(n));
    let mut y = match cons {
        Consonant::P | Consonant::B => {
            dsp::sosfilt(&[dsp::butter_highpass(2, hz(300.0)), dsp::butter_lowpass(2, hz(3000.0))], &source)
        }
        Consonant::T | Consonant::D => {
            let base = dsp::sosfilt(
                &[dsp::butter_highpass(2, hz(2500.0)), dsp::butter_highpass(2, hz(2500.0)), dsp::butter_lowpass(2, hz(8000.0))],
                &source,
            );
            let ring = unit(dsp::sosfilt(&[dsp::resonator(hz(4500.0), hz(1500.0))], &base));
            unit(base).iter().zip(&ring).map(|(b, r)| 0.3 * b + r).collect()
        }
        _ => {
            let f2 = call.vowels.first().map(|v| v.ratios()[1]).unwrap_or(1.0) * 1500.0;
            let peak = clip(1.1 * f2, 1500.0, 2600.0);
            let base = dsp::sosfilt(&[dsp::butter_highpass(2, hz(1000.0)), dsp::butter_lowpass(2, hz(5000.0))], &source);
            let ring = unit(dsp::sosfilt(&[dsp::resonator(hz(peak), hz(500.0))], &base));
            unit(base).iter().zip(&ring).map(|(b, r)| 0.15 * b + r).collect()
        }
    };
    let attack = 0.0004 * SRF;
    for (i, v) in y.iter_mut().enumerate() {
        let u = i as f64 / n as f64;
        *v *= (i as f64 / attack).min(1.0) * (-3.0 * u).exp() * smoothstep((1.0 - u) / 0.15);
    }
    y
}

/// Voicing through a voiced stop's closure: the call as rendered, with only its lowest harmonics
/// left (the closed tract's walls pass nothing higher).
fn render_voice_bar(call: &Call, body: &Body, rng: &mut Rng) -> Vec<f64> {
    let mut y = render_call(call, body, rng);
    let cut = (VOICE_BAR.1 * 0.17 / body.tract()).min(0.45 * SRF);
    dsp::butter_lowpass(2, cut).process(&mut y);
    dsp::butter_lowpass(2, cut).process(&mut y);
    y
}

/// Intro moan, full roar, then accelerating grunts with inhales (from lion_roar.py).
fn territorial_bout(body: &Body, voice: &Voice, rng: &mut Rng) -> Vec<Call> {
    let s = body.call_scale() / voice.tempo / 1.17; // lion_roar.py timings are for a 190 kg lion
    let mut j = |v: f64, sd: f64| v * (1.0 + sd * rng.standard_normal());
    let p = |pts: &[(f64, f64)]| -> Points { pts.iter().map(|&(u, st)| (u, voice.pitch + st * voice.pitch_range)).collect() };
    let base = 0.5 * body.roughness + voice.harshness;
    let h = |pts: &[(f64, f64)]| -> Points { pts.iter().map(|&(u, dh)| (u, base + dh)).collect() };
    let vowels = plain_vowels(&voice.vowels);
    let first = vowels[0].clone();
    let last = vowels[vowels.len() - 1].clone();

    let mut moan = Call::new(
        0.0, j(0.9, 0.06) * s,
        p(&[(0.0, -9.9), (0.5, j(-4.6, 0.1)), (1.0, -7.6)]),
        vec![(0.0, 0.0), (0.3, 0.3), (0.8, 0.28), (1.0, 0.0)],
        h(&[(0.0, -0.3), (1.0, -0.15)]),
        vec![(0.0, 0.05), (0.6, 0.25), (1.0, 0.1)],
        vec![first],
    );
    moan.breath = 0.1;
    moan.fry = 0.4;
    moan.nasal = 0.3;
    moan.larynx = Some(vec![(0.0, 0.0), (1.0, 0.2)]);
    let moan = moan.nlp(voice, 0.1, 0.0, 0.0, 0.0);

    let mut roar = Call::new(
        1.4 * s, j(2.3, 0.06) * s,
        p(&[(0.0, -6.5), (0.15, j(2.1, 0.3)), (0.4, j(7.9, 0.1)), (0.6, 6.5), (0.85, -1.3), (1.0, -8.7)]),
        vec![(0.0, 0.0), (0.08, 0.6), (0.35, 1.0), (0.7, 0.9), (0.9, 0.4), (1.0, 0.0)],
        h(&[(0.0, -0.1), (0.3, 0.3), (0.5, 0.5), (0.75, 0.2), (1.0, 0.0)]),
        vec![(0.0, 0.3), (0.3, 0.9), (0.7, 1.0), (1.0, 0.4)],
        vowels.clone(),
    );
    roar.breath = 0.1;
    roar.fry = 0.8;
    roar.larynx = Some(vec![(0.0, 0.2), (0.3, 1.0), (0.8, 1.0), (1.0, 0.5)]);
    let roar = roar.nlp(voice, 0.5, 0.3, 0.0, 0.0);

    let mut calls = vec![moan, inhale_call(1.0 * s, 0.35 * s), roar];
    let mut t = 4.0 * s;
    for (i, gap) in [0.95, 0.85, 0.75, 0.7, 0.65].iter().enumerate() {
        let level = 1.4 - 0.15 * i as f64; // low f0 carries less energy after radiation, so push grunts harder
        calls.push(inhale_call(t - 0.3 * s, 0.28 * s));
        let mut g = Call::new(
            t, j(0.42, 0.1) * s,
            p(&[(0.0, -7.6), (0.25, j(-3.7, 0.15)), (1.0, -13.1)]),
            vec![(0.0, 0.0), (0.12, level), (0.5, level * 0.8), (1.0, 0.0)],
            h(&[(0.0, 0.25), (0.4, 0.45), (1.0, 0.15)]),
            vec![(0.0, 0.4), (0.3, 0.85), (1.0, 0.3)],
            vec![last.clone()],
        );
        g.breath = 0.1;
        g.fry = 0.9;
        g.larynx = Some(vec![(0.0, 0.5), (0.3, 0.9), (1.0, 0.6)]);
        let g = g.nlp(voice, 0.4, 0.2, 0.0, 0.0);
        calls.push(g);
        t += gap * s;
    }
    calls
}

// ---------------------------------------------------------------- sources

/// Rosenberg glottal flow pulse at phase p in [0, 1).
#[inline]
fn rosenberg(p: f64, open_quotient: f64) -> f64 {
    let tp = 0.72 * open_quotient; // opening phase
    let tn = 0.28 * open_quotient; // closing phase (shorter = brighter)
    if p < tp {
        0.5 * (1.0 - (PI * p / tp).cos())
    } else if p < tp + tn {
        (0.5 * PI * (p - tp) / tn).cos()
    } else {
        0.0
    }
}

/// Period-by-period pulse train so jitter/shimmer/period-doubling are per cycle.
/// Returns (flow, opening): `opening` is the unscaled glottal opening (0..1) that pulses the breath noise.
fn larynx_source(call: &Call, f0: &[f64], amp: &[f64], harsh: &[f64], rng: &mut Rng, base_f0: f64, open_quotient: f64) -> (Vec<f64>, Vec<f64>) {
    let n = f0.len();
    let mut flow = vec![0.0; n];
    let mut opening = vec![0.0f64; n];
    let (mut t, mut k, mut drift) = (0.0_f64, 0usize, 0.0_f64);
    let tremor_rate = 5.0 + rng.uniform(0.0, 2.0);
    let tremor_phase = rng.uniform(0.0, 2.0 * PI);
    let amp_peak = amp.iter().cloned().fold(0.0, f64::max) + 1e-9;
    // NLP regime: 0 = normal, 2/3 = subharmonic period, -1 = chaos. `cyc` counts cycles in it.
    let (mut regime, mut cyc, mut chaos_x) = (0i32, 0i32, 0.5f64);
    while t < call.dur {
        let i = ((t * SRF) as usize).min(n - 1);
        let u = t / call.dur;
        let mut a = amp[i];
        let mut h = harsh[i];

        // Creaky tail: as air runs out the folds slacken, pitch sags and cycles turn irregular.
        let tail = clip((u - 0.8) / 0.2, 0.0, 1.0) * call.fry;
        let freq = f0[i] * (1.0 - 0.45 * tail);
        let h_voice = h;
        h = (h + 0.4 * tail).min(1.0);

        // Slow wander plus cycle-to-cycle jitter; jitter explodes when harsh (chaos).
        let keep = (-1.0 / (freq * DRIFT_TIME)).exp();
        drift = keep * drift + (1.0 - keep * keep).sqrt() * rng.standard_normal();
        // Fry pulses are irregular in timing and size; evenly spaced ones motorboat.
        let fry = clip(1.0 - freq / (0.6 * base_f0), 0.0, 1.0);
        let jitter = BASE_JITTER + CHAOS_JITTER * h.powi(3) + 0.05 * tail + FRY_JITTER * fry;
        let mut period = 1.0 / (freq * (1.0 + DRIFT_DEPTH * drift + jitter * rng.standard_normal()));
        // Harshness terms are quadratic: mild harshness should barely roughen a clean voice.
        let shimmer = 1.0 + (BASE_SHIMMER + 0.25 * h * h + FRY_SHIMMER * fry) * rng.standard_normal();
        a *= 1.0 + 0.07 * (2.0 * PI * tremor_rate * t + tremor_phase).sin(); // respiratory tremor

        // NLP episodes. Loud moments (high subglottal pressure) push the folds over a bifurcation:
        // the voice switches abruptly into a subharmonic or chaotic regime, holds it, then drops back.
        let drive = (amp[i] / amp_peak).powi(2);
        let dt = 1.0 / freq;
        if regime == 0 {
            let p_sub = 1.0 - (-NLP_RATE * call.subharmonics * drive * dt).exp();
            let p_chaos = 1.0 - (-NLP_RATE * call.chaos * drive * dt).exp();
            let r = rng.random();
            if r < p_sub {
                regime = if rng.random() < 0.25 { 3 } else { 2 };
                cyc = 0;
            } else if r < p_sub + p_chaos {
                regime = -1;
                cyc = 0;
                chaos_x = rng.uniform(0.2, 0.8);
            }
        } else {
            let level = if regime < 0 { call.chaos } else { call.subharmonics };
            let mean_len = NLP_EPISODE.0 + (NLP_EPISODE.1 - NLP_EPISODE.0) * level;
            if rng.random() < 1.0 - (-dt / mean_len).exp() {
                regime = 0;
            }
        }
        cyc += 1;

        // Period doubling from harshness alone: alternate cycles differ in length and strength.
        let sub = clip((h - 0.35) / 0.5, 0.0, 1.0) * 0.55;
        if regime > 0 {
            // Subharmonic episode: one strong cycle then weaker ones, every 2 or 3 cycles -> f0/2, f0/3.
            let depth = sub.max(SUB_DEPTH * (0.5 + 0.5 * call.subharmonics));
            let pos = cyc % regime;
            if pos != 0 {
                period *= 1.0 + SUB_TIMING * depth / (regime - 1) as f64;
                a *= 1.0 - depth * if pos == 1 { 1.0 } else { 0.5 };
            } else {
                period *= 1.0 - SUB_TIMING * depth;
            }
        } else if k % 2 == 1 {
            period *= 1.0 + 0.12 * sub;
            a *= 1.0 - sub;
        } else {
            period *= 1.0 - 0.12 * sub;
        }
        if regime < 0 {
            // Deterministic chaos: a logistic map just past the onset of chaos drives period and
            // strength. Irregular but correlated cycle to cycle (banded), so it stays a voice, not hiss.
            chaos_x = (3.6 + 0.4 * call.chaos) * chaos_x * (1.0 - chaos_x);
            let swing = 2.0 * chaos_x - 1.0;
            period *= 1.0 + CHAOS_PERIOD * call.chaos * swing;
            a *= (1.0 + CHAOS_AMP * call.chaos * swing).max(0.0);
        }
        // A chaotic draw can push the period negative; never step time backwards.
        if !(period > 1.0 / SRF) {
            period = 1.0 / SRF;
        }

        // Pressed (harsh) phonation closes faster -> more high harmonics; shape varies per cycle.
        let mut oq = open_quotient * (1.0 - 0.385 * h) *(1.0 + (SHAPE_JITTER + 0.05 * h * h) * rng.standard_normal());
        // Fry register: far below its natural pitch each cycle is a sharp click.
        oq = clip(oq * (freq / (0.6 * base_f0)).min(1.0).powi(3), 0.03, 0.9);
        let i0 = (t * SRF).ceil() as usize;
        let i1 = (((t + period) * SRF).ceil() as usize).min(n);
        // Falsetto: well above natural pitch the folds vibrate almost sinusoidally (a howl).
        let thin = call.falsetto * clip(((freq / base_f0).log2() - 0.3) / 0.7, 0.0, 1.0) * (1.0 - h_voice * h_voice);
        let gain = (a * shimmer).max(0.0);
        for idx in i0..i1 {
            let p = (idx as f64 / SRF - t) / period;
            let pulse = (1.0 - thin) * rosenberg(p, oq) + thin * 0.5 * (1.0 - (2.0 * PI * p).cos());
            flow[idx] += gain * pulse;
            opening[idx] = opening[idx].max(pulse);
        }
        t += period;
        k += 1;
    }
    (flow, opening)
}

/// Moving average over `width` samples, centred (numpy convolve mode="same").
fn smooth_box(x: &[f64], width: usize) -> Vec<f64> {
    let n = x.len();
    let w = width.max(1);
    let mut prefix = vec![0.0; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + x[i];
    }
    let c = (w - 1) / 2;
    (0..n)
        .map(|i| {
            let hi = (i + c + 1).min(n);
            let lo = (i + c + 1).saturating_sub(w).min(hi);
            (prefix[hi] - prefix[lo]) / w as f64
        })
        .collect()
}

/// 0/1 mask of random on-episodes (onsets per second, mean length in s), with 5 ms ramps so
/// abrupt switches don't click.
fn episodes(n: usize, rate: f64, mean_len: f64, rng: &mut Rng) -> Vec<f64> {
    let mut mask = vec![0.0; n];
    if rate > 0.0 {
        let mut t = rng.exponential(1.0 / rate);
        while ((t * SRF) as usize) < n {
            let length = rng.exponential(mean_len);
            let (i0, i1) = ((t * SRF) as usize, (((t + length) * SRF) as usize).min(n));
            for v in &mut mask[i0..i1.max(i0)] {
                *v = 1.0;
            }
            t += length + rng.exponential(1.0 / rate);
        }
    }
    smooth_box(&mask, (0.005 * SRF) as usize)
}

/// Frequency jumps: the folds snap into another vibration mode, so the pitch leaps abruptly
/// (several semitones, mostly upward when aroused) and later snaps back. Returns semitone steps.
fn register_jumps(call: &Call, n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut steps = vec![0.0; n];
    // A few breaks per call, not one per half second: a long howl with a break every beat reads as
    // a robot. Breaks are short and never overlap, so they can't stack into a staircase.
    let count = rng.poisson(call.jumps * (0.5 + 0.25 * call.dur));
    let mut t = rng.uniform(0.1, 0.4) * call.dur;
    for _ in 0..count {
        let length = (0.04 + rng.exponential(0.12)).min(0.3).min(0.4 * call.dur);
        if t + length > 0.9 * call.dur {
            break;
        }
        let size = rng.uniform(3.0, 7.0) * if rng.random() < 0.7 { 1.0 } else { -1.0 };
        let (i0, i1) = (((t * SRF) as usize).min(n), (((t + length) * SRF) as usize).min(n));
        for v in &mut steps[i0..i1] {
            *v = size;
        }
        t += length + rng.uniform(0.15, 0.4) * call.dur;
    }
    smooth_box(&steps, (0.004 * SRF) as usize)
}

/// Biphonation: a second, independent oscillation (a vocal membrane, or the two folds
/// desynchronising) at a non-harmonic pitch ratio. It adds its own tone and modulates the main
/// flow, giving sidebands at f0 +- g. Comes and goes in long episodes.
///
/// The second tone follows only the call's slow contour (`f0_contour`: no vibrato, no register
/// jumps) and wanders on its own. Tracking every move of the main pitch would make it a parallel
/// harmony, which sounds like a robot.
fn biphonate(flow: &mut [f64], call: &Call, f0_contour: &[f64], amp: &[f64], rng: &mut Rng) {
    let n = flow.len();
    let level = call.biphonation;
    let mask = episodes(n, 0.5 + 1.0 * level, 0.6 + 1.4 * level, rng);
    let wander = smooth_noise(n, 0.5, rng); // the second pitch wanders, never locks
    let mut cum = 0.0;
    for i in 0..n {
        cum += call.biphonation_ratio * f0_contour[i] * (1.0 + 0.04 * wander[i]);
        let phase = 2.0 * PI * cum / SRF;
        let b = level * mask[i];
        flow[i] = flow[i] * (1.0 + BIPHONATION_COUPLING * b * phase.sin()) + b * amp[i] * 0.5 * (1.0 - phase.cos());
    }
}

/// Band-limited random wander, roughly `rate` changes per second.
fn smooth_noise(n: usize, rate: f64, rng: &mut Rng) -> Vec<f64> {
    let mut x = rng.normals(n);
    dsp::butter_lowpass(1, rate).process(&mut x);
    let s = dsp::std(&x) + 1e-9;
    x.iter_mut().for_each(|v| *v /= s);
    x
}

/// Two independently controlled tonal voices (left and right syrinx) for song. Harsh calls
/// (caws, quacks) cross over into pulsed, chaotic oscillation like a harsh larynx.
fn syrinx_source(call: &Call, f0: &[f64], amp: &[f64], harsh: &[f64], rng: &mut Rng, base_f0: f64, buzz: f64, open_quotient: f64) -> (Vec<f64>, Vec<f64>) {
    let n = f0.len();
    let jn = smooth_noise(n, 300.0, rng);
    let ratio = 2f64.powf(rng.uniform(3.0, 7.0) / 12.0);
    let wn = smooth_noise(n, 20.0, rng);
    let mut tonal = vec![0.0; n];
    let mut tonal_open = vec![0.0; n];
    let (mut c1, mut c2) = (0.0, 0.0);
    let mut any_pulsed = false;
    let mut pulsed = vec![0.0; n];
    for i in 0..n {
        let h = harsh[i];
        let jitter = (0.003 + 0.03 * h * h) * jn[i];
        c1 += f0[i] * (1.0 + jitter);
        c2 += f0[i] * ratio * (1.0 + 0.01 * wn[i]);
        let (ph1, ph2) = (2.0 * PI * c1 / SRF, 2.0 * PI * c2 / SRF);
        let drive = 1.0 + 2.0 * h;
        tonal[i] = (drive * ph1.sin()).tanh() / drive.tanh() + ph2.sin() * (0.05 + 0.4 * h);
        tonal_open[i] = 0.5 * (1.0 + ph1.sin());
        pulsed[i] = buzz.max(clip((h - 0.3) / 0.4, 0.0, 1.0));
        any_pulsed |= pulsed[i] != 0.0;
    }
    if !any_pulsed {
        return ((0..n).map(|i| amp[i] * tonal[i]).collect(), tonal_open);
    }
    // Fry is judged against the call's own pitch here: the songbird-scaled natural pitch would
    // turn every low quack into razor-sharp clicks.
    let (pulses, opening) = larynx_source(call, f0, amp, harsh, rng, base_f0.min(dsp::median(f0)), open_quotient);
    let m = dsp::mean(&pulses);
    (
        (0..n).map(|i| (1.0 - pulsed[i]) * amp[i] * tonal[i] + pulsed[i] * 2.0 * (pulses[i] - m)).collect(),
        (0..n).map(|i| (1.0 - pulsed[i]) * tonal_open[i] + pulsed[i] * opening[i]).collect(),
    )
}

/// Pulse train (frog trill, insect chirp): every pulse is a short burst ringing at the carrier,
/// plus `harmonic` of its octave.
fn pulse_source(call: &Call, rate: &[f64], amp: &[f64], harsh: &[f64], carrier: &[f64], rng: &mut Rng, harmonic: f64) -> Vec<f64> {
    let n = rate.len();
    let mut out = vec![0.0; n];
    let mut t = 0.0;
    while t < call.dur {
        let i = ((t * SRF) as usize).min(n - 1);
        let h = harsh[i];
        let c = carrier[i];
        let period = 1.0 / (rate[i] * (1.0 + (0.02 + 0.2 * h) * rng.standard_normal()));
        let length = (period * 0.8).min(12.0 / c);
        let len = if length > 0.0 { (length * SRF) as usize } else { 0 };
        let noise = rng.normals(len);
        let i0 = (t * SRF) as usize;
        for (k, nz) in noise.iter().enumerate() {
            let idx = i0 + k;
            if idx >= n {
                break;
            }
            let tt = k as f64 / SRF;
            let ring = (-tt * c / 3.0).exp();
            let mut burst = ((2.0 * PI * c * tt).sin() + harmonic * (4.0 * PI * c * tt).sin()) * ring;
            burst += h * 0.5 * nz * (-tt * c / 2.0).exp();
            out[idx] += amp[i] * burst;
        }
        t += period.max(1.0 / SRF);
    }
    out
}

// ---------------------------------------------------------------- filters

/// Per-sample lossy tube: a delay line with negative feedback (closed at the glottis, open at
/// the lips) resonates at every odd quarter wavelength, so the formant series never ends. A
/// one-pole lowpass in the loop broadens the upper formants. Three pole-zero sections then move
/// the neutral F1-F3 (fn) to the vowel's targets (fv).
fn tube(x: &[f64], delay: &[f64], loss: &[f64], lp_a: f64, fnr: &[Vec<f64>; 3], fv: &[Vec<f64>; 3]) -> Vec<f64> {
    let n = x.len();
    let max_delay = delay.iter().cloned().fold(0.0, f64::max) as usize;
    let mut size = 1usize;
    while size < max_delay + 4 {
        size *= 2;
    }
    let mask = size - 1;
    let mut buf = vec![0.0; size];
    let mut state = [[0.0f64; 4]; 3];
    let mut y = vec![0.0; n];
    let mut lp = 0.0;
    let two_pi = 2.0 * PI / SRF;
    let lp_delay = lp_a / (1.0 - lp_a);
    for i in 0..n {
        let d = delay[i];
        let di = d as usize;
        let frac = d - di as f64;
        let r0 = buf[i.wrapping_sub(di) & mask];
        let r1 = buf[i.wrapping_sub(di + 1) & mask];
        lp = (1.0 - lp_a) * (r0 + frac * (r1 - r0)) + lp_a * lp;
        let out = x[i] - loss[i] * lp;
        buf[i & mask] = out;

        let mut v = out;
        for k in 0..3 {
            // Match the comb's own pole radius at this formant so the zero cancels it cleanly.
            let w = two_pi * fnr[k][i];
            let lp_mag = (1.0 - lp_a) / (1.0 - 2.0 * lp_a * w.cos() + lp_a * lp_a).sqrt();
            let r = (loss[i] * lp_mag).powf(2.0 / (d + lp_delay));
            let cz = 2.0 * r * w.cos();
            let cp = 2.0 * r * (two_pi * fv[k][i]).cos();
            let r2 = r * r;
            let g = (1.0 - cp + r2) / (1.0 - cz + r2); // unity gain at DC
            let s = &mut state[k];
            let o = g * (v - cz * s[0] + r2 * s[1]) + cp * s[2] - r2 * s[3];
            s[1] = s[0];
            s[0] = v;
            s[3] = s[2];
            s[2] = o;
            v = o;
        }
        y[i] = v;
    }
    y
}

/// Cascade of peaking (bell) filters that follow the formants: adds resonance at each formant
/// without changing the level between them. Coefficients are refreshed every 8 samples.
fn formant_peaks(x: &mut [f64], freqs: &[Vec<f64>], bws: &[Vec<f64>], gains_db: &[f64]) {
    for k in 0..freqs.len() {
        if gains_db[k] == 0.0 {
            continue;
        }
        let a_lin = 10f64.powf(gains_db[k] / 40.0);
        let (mut z1, mut z2) = (0.0, 0.0);
        let (mut b0, mut b1, mut b2, mut a1, mut a2) = (1.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..x.len() {
            if i % 8 == 0 {
                let w0 = 2.0 * PI * freqs[k][i] / SRF;
                let alpha = w0.sin() * bws[k][i] / (2.0 * freqs[k][i]); // sin(w0) / (2Q), Q = f / bandwidth
                let norm = 1.0 / (1.0 + alpha / a_lin);
                b0 = (1.0 + alpha * a_lin) * norm;
                b1 = -2.0 * w0.cos() * norm;
                b2 = (1.0 - alpha * a_lin) * norm;
                a1 = b1;
                a2 = (1.0 - alpha / a_lin) * norm;
            }
            let v = x[i];
            let out = b0 * v + z1;
            z1 = b1 * v - a1 * out + z2;
            z2 = b2 * v - a2 * out;
            x[i] = out;
        }
    }
}

/// Three cascaded one-pole lowpasses with a per-sample cutoff.
fn lips(x: &[f64], cutoff: &[f64]) -> Vec<f64> {
    let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
    x.iter()
        .zip(cutoff)
        .map(|(&xi, &c)| {
            let a = (-2.0 * PI * c / SRF).exp();
            s1 = (1.0 - a) * xi + a * s1;
            s2 = (1.0 - a) * s1 + a * s2;
            s3 = (1.0 - a) * s2 + a * s3;
            s3
        })
        .collect()
}

/// Lossy tube for this tract length, shaped by the current vowel. Larynx lowering lengthens the
/// tube, jaw opening raises F1/F2, and the tract drifts slowly. When the pitch nears F1 the open
/// glottis damps the tract (source-tract interaction), so high voices don't ring like a flute.
fn vocal_tract(x: &[f64], u: &[f64], call: &Call, body: &Body, rng: &mut Rng, bw_scale: f64, f0: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mouth = curve(&call.mouth, u);
    let larynx = match &call.larynx {
        Some(l) => curve(l, u),
        None => vec![0.0; n],
    };
    let wander = smooth_noise(n, 3.0, rng);
    let tract = body.tract();

    let lp_a = clip(TRACT_LOSS_LOWPASS * body.hf_loss, 0.05, 0.9);
    let mut round_trip = vec![0.0; n];
    let mut delay = vec![0.0; n];
    let mut f1 = vec![0.0; n];
    for i in 0..n {
        let length = tract * (1.0 + body.larynx_drop * larynx[i]) * (1.0 + 0.01 * wander[i]);
        round_trip[i] = 2.0 * length / C_SOUND * SRF;
        delay[i] = (round_trip[i] - lp_a / (1.0 - lp_a)).max(2.0); // the loop lowpass adds its own delay
        f1[i] = SRF / (2.0 * round_trip[i]);
    }

    let nv = call.vowels.len();
    let table: Vec<[f64; 3]> = call.vowels.iter().map(|v| v.ratios()).collect();
    let pos: Vec<f64> = if nv > 1 { (0..nv).map(|k| k as f64 / (nv - 1) as f64).collect() } else { vec![0.0] };
    let cols: [Vec<f64>; 3] = std::array::from_fn(|k| table.iter().map(|r| r[k]).collect());
    let drift: [Vec<f64>; 3] = std::array::from_fn(|_| smooth_noise(n, 2.0, rng));
    let lim = 0.45 * SRF;
    let mut fnr: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
    let mut fv: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
    for i in 0..n {
        for k in 0..3 {
            let base = (2 * k + 1) as f64 * f1[i];
            let r = formant_transition(call, k, u[i], interp(u[i], &pos, &cols[k]));
            let mut target = base * r;
            if k == 0 {
                target *= 0.75 + 0.5 * mouth[i];
            } else if k == 1 {
                target *= 0.9 + 0.2 * mouth[i];
            }
            target *= 1.0 + 0.03 * drift[k][i];
            fv[k][i] = target.min(lim);
            fnr[k][i] = base.min(lim);
        }
    }

    // Source-tract interaction: extra F1 damping (Hz) when the pitch sits on F1.
    let glottal_damp: Vec<f64> = match f0 {
        Some(f0) => (0..n)
            .map(|i| {
                let d = (f0[i] - fv[0][i]) / (0.5 * fv[0][i]);
                GLOTTAL_DAMPING * f0[i] * (-d * d).exp()
            })
            .collect(),
        None => vec![0.0; n],
    };
    let loss: Vec<f64> = (0..n)
        .map(|i| {
            let bw1 = (60.0 + 0.07 * f1[i]) * bw_scale + glottal_damp[i];
            (-PI * bw1 * round_trip[i] / SRF).exp() // round-trip loss giving F1 that bandwidth
        })
        .collect();
    let mut y = tube(x, &delay, &loss, lp_a, &fnr, &fv);

    if body.formant_gain != 0.0 {
        // Formant resonance: F1-F3 at the vowel's targets, F4-F5 on the neutral tube. F1 keeps the
        // glottal damping, so a high voice sitting on F1 still doesn't ring like a flute.
        let top = 0.42 * SRF;
        let peaks: Vec<Vec<f64>> = (0..5)
            .map(|k| (0..n).map(|i| if k < 3 { fv[k][i] } else { (2 * k + 1) as f64 * f1[i] }.min(top)).collect())
            .collect();
        let widths: Vec<Vec<f64>> = (0..5)
            .map(|k| {
                (0..n)
                    .map(|i| body.formant_width * bw_scale * (40.0 + 0.06 * peaks[k][i]) + if k == 0 { glottal_damp[i] } else { 0.0 })
                    .collect()
            })
            .collect();
        let gains: Vec<f64> = FORMANT_LEVELS.iter().map(|l| l * body.formant_gain).collect();
        formant_peaks(&mut y, &peaks, &widths, &gains);
    }

    // A small or rounded lip opening radiates high frequencies poorly. Cutoffs scale with size.
    let rounds: Vec<f64> = call.vowels.iter().map(|v| v.rounding()).collect();
    let (closed, wide) = LIP_CUTOFF_HZ;
    let size = (0.17 / tract).powf(0.7);
    let cutoff: Vec<f64> = (0..n)
        .map(|i| {
            let rounding = interp(u[i], &pos, &rounds);
            let aperture = clip(mouth[i] * (1.0 - 0.8 * rounding * (1.0 - call.spread)), 0.0, 1.0);
            ((closed + (wide - closed) * aperture.powf(1.5)) * size).min(lim)
        })
        .collect();
    (lips(&y, &cutoff), mouth)
}

/// Sound leaking through the nose: an antiresonance (scaled to the tract) and a softer top end.
fn nasalize(y: &[f64], mouth: &[f64], call: &Call, body: &Body) -> Vec<f64> {
    let notch_f = (800.0 * 0.38 / body.tract()).min(0.35 * SRF);
    let nasal = dsp::sosfilt(&[dsp::notch(notch_f, 1.5), dsp::butter_lowpass(1, (4.0 * notch_f).min(0.45 * SRF))], y);
    y.iter()
        .zip(&nasal)
        .zip(mouth)
        .map(|((&yi, &ni), &m)| {
            let mix = clip(body.nasality + call.nasal + 0.8 * (0.3 - m), 0.0, 0.5);
            (1.0 - mix) * yi + mix * ni
        })
        .collect()
}

// ---------------------------------------------------------------- rendering

fn render_call(call: &Call, body: &Body, rng: &mut Rng) -> Vec<f64> {
    let n = (call.dur * SRF) as usize + SR / 10;
    let u: Vec<f64> = (0..n).map(|i| clip(i as f64 / (call.dur * SRF), 0.0, 1.0)).collect();
    let mut amp = curve(&call.amp, &u);
    if call.rumble > 0.0 {
        // Gravel: irregular 20-50 Hz tissue flutter, applied in log amplitude so it never gates
        // the call to silence.
        let wobble = dsp::sosfilt(&dsp::bandpass(20.0, 50.0), &rng.normals(n));
        let s = dsp::std(&wobble) + 1e-9;
        for (a, w) in amp.iter_mut().zip(&wobble) {
            *a *= (0.5 * call.rumble * w / s).exp();
        }
    }
    let harsh: Vec<f64> = u.iter().map(|&x| clip(curve_at(&call.harsh, x), 0.0, 1.0)).collect();
    // Animal vibrato wanders in rate and depth; a perfectly regular one is a musician's.
    let rate_n = smooth_noise(n, 1.5, rng);
    let depth_n = smooth_noise(n, 2.0, rng);
    let mut phase = 0.0;
    let base_f0 = body.base_f0();
    let mut semitones = vec![0.0; n];
    for i in 0..n {
        let rate = call.vibrato_rate * (1.0 + 0.15 * rate_n[i]);
        let depth = (call.vibrato * (1.0 + 0.35 * depth_n[i])).max(0.0);
        phase += rate;
        let vib = depth * (2.0 * PI * phase / SRF).sin() * clip(u[i] * 2.0 - 0.3, 0.0, 1.0);
        semitones[i] = curve_at(&call.pitch, u[i]) + vib;
    }
    if call.voiced > 0.0 && call.jumps > 0.0 {
        for (s, j) in semitones.iter_mut().zip(register_jumps(call, n, rng)) {
            *s += j;
        }
    }
    let f0: Vec<f64> = semitones.iter().map(|s| base_f0 * 2f64.powf(s / 12.0)).collect();

    if body.source == Source::Pulse {
        let carrier_hz = body.carrier();
        let carrier: Vec<f64> = semitones.iter().map(|s| carrier_hz * 2f64.powf(s / 24.0)).collect();
        let mut y = if call.voiced > 0.0 {
            pulse_source(call, &f0, &amp, &harsh, &carrier, rng, body.resonator_harmonic)
        } else {
            vec![0.0; n]
        };
        let bn = rng.normals(n);
        for i in 0..n {
            y[i] += 0.05 * call.breath * amp[i] * bn[i];
        }
        // Vocal sac / body resonance, and its octave when the call has a second band.
        let mut out = dsp::sosfilt(&[dsp::resonator(carrier_hz, 0.15 * carrier_hz)], &y);
        if body.resonator_harmonic > 0.0 && 2.0 * carrier_hz < 0.45 * SRF {
            let oct = dsp::sosfilt(&[dsp::resonator(2.0 * carrier_hz, 0.15 * carrier_hz)], &y);
            for i in 0..n {
                out[i] += body.resonator_harmonic * oct[i];
            }
        }
        return out;
    }

    let voiced = call.voiced > 0.0 && body.source != Source::Hiss;
    let (mut flow, opening) = match (voiced, body.source) {
        (true, Source::Larynx) => larynx_source(call, &f0, &amp, &harsh, rng, base_f0, body.open_quotient),
        (true, Source::Syrinx) => syrinx_source(call, &f0, &amp, &harsh, rng, base_f0, body.buzz, body.open_quotient),
        _ => (vec![0.0; n], vec![1.0; n]),
    };
    if voiced && body.source == Source::Larynx && call.biphonation > 0.0 {
        let f0_contour: Vec<f64> = u.iter().map(|&x| base_f0 * 2f64.powf(curve_at(&call.pitch, x) / 12.0)).collect();
        biphonate(&mut flow, call, &f0_contour, &amp, rng);
    }

    // Turbulence noise, pulsing with the glottal flow and louder when harsh. Its corner tracks
    // the tract's own F1, except for hissing animals; harsher calls get brighter noise.
    let harsh_mean = harsh.iter().zip(&amp).map(|(h, a)| h * a).sum::<f64>() / (amp.iter().sum::<f64>() + 1e-9);
    let corner = if body.source == Source::Hiss {
        4000.0
    } else {
        (3.0 * C_SOUND / (4.0 * body.tract())).min(NOISE_CORNER_MAX) * (1.0 + harsh_mean * harsh_mean) * body.noise_color
    };
    let noise = dsp::sosfilt(
        &[dsp::butter_highpass(2, 150.0), dsp::butter_lowpass(1, corner.min(0.4 * SRF))],
        &rng.normals(n),
    );
    // Pulsed with the glottal opening of every cycle (not the call's loudness), so quiet stretches
    // pulse as deeply as loud ones and the noise fuses into the voice instead of hissing beside it.
    let turbulence: Vec<f64> = (0..n)
        .map(|i| {
            let gate = if voiced { (1.0 - body.breath_pulse) + body.breath_pulse * opening[i] } else { 1.0 };
            noise[i] * gate * amp[i] * (0.4 + 0.8 * harsh[i])
        })
        .collect();
    let breath = body.breathiness + call.breath;

    // Voice and breath go through the same tract (same drift) separately so they can be mixed
    // at a known ratio.
    let bw_scale = body.tract_damping_or_default();
    let tract_seed = rng.next_u64() >> 33;
    let radiate = |x: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut trng = Rng::new(tract_seed);
        let (y, mouth) = vocal_tract(x, &u, call, body, &mut trng, bw_scale, if voiced { Some(&f0) } else { None });
        // Lip radiation (+6 dB/oct).
        let mut prev = 0.0;
        let y = y.iter().map(|&v| {
            let o = v - 0.97 * prev;
            prev = v;
            o
        }).collect();
        (y, mouth)
    };

    let (mut y, mouth) = if voiced {
        let (mut y, mouth) = radiate(&flow);
        if breath > 0.0 {
            // Breath relative to the voice itself; the cubic growl term keeps clean calls clean
            // while real growls come out roughly as much noise as tone.
            let (y_noise, _) = radiate(&turbulence);
            let ratio = BREATH_FLOOR + BREATH_SCALE * breath * (0.5 + harsh_mean) + GROWL_NOISE * harsh_mean.powi(3);
            let g = ratio * rms(&y) / rms(&y_noise);
            for (a, b) in y.iter_mut().zip(&y_noise) {
                *a += g * b;
            }
        }
        (y, mouth)
    } else {
        let scaled: Vec<f64> = turbulence.iter().map(|t| t * breath).collect();
        radiate(&scaled)
    };

    if matches!(body.source, Source::Larynx | Source::Syrinx) {
        y = nasalize(&y, &mouth, call, body);
    }
    if body.source == Source::Larynx {
        // Chest/body resonance driven by the raw flow: the low thump of big animals.
        let chest = dsp::sosfilt(&[dsp::resonator(95.0 * (190.0 / body.mass).cbrt(), body.chest_width)], &flow);
        let g = CHEST_GAIN * body.chest * (body.mass / 20.0).min(1.0);
        for (a, c) in y.iter_mut().zip(&chest) {
            *a += g * c;
        }
    }
    if call.consonant == Some(Consonant::H) {
        // Breath from the open glottis: its F1 is heavily damped.
        let r0 = call.vowels.first().map(|v| v.ratios()[0]).unwrap_or(1.0);
        let f1 = C_SOUND / (4.0 * body.tract()) * r0 * 1.125;
        dsp::butter_highpass(2, (ASPIRATION_F1_CUT * f1).min(0.45 * SRF)).process(&mut y);
    }
    if body.low_cut != 0.0 {
        // A small mouth or nostrils radiate low frequencies poorly (a pig's snout).
        dsp::butter_highpass(2, body.low_cut).process(&mut y);
    }
    y
}

/// Short, diffuse tail. `wet` is the tail's RMS relative to the dry sound (0.1 = -20 dB).
fn outdoor_reverb(x: &[f64], rng: &mut Rng, wet: f64) -> Vec<f64> {
    let tail_len = (0.8 * SRF) as usize;
    let mut ir: Vec<f64> = rng.normals(tail_len).iter().enumerate().map(|(i, v)| v * (-(i as f64 / SRF) / 0.1).exp()).collect();
    dsp::butter_lowpass(2, 4000.0).process(&mut ir);
    for v in ir.iter_mut().take((0.015 * SRF) as usize) {
        *v = 0.0;
    }
    let energy = ir.iter().map(|v| v * v).sum::<f64>().sqrt();
    ir.iter_mut().for_each(|v| *v *= wet / energy); // normalize by energy, not peak
    ir[0] = 1.0;
    let mut y = dsp::fft_convolve(x, &ir);
    y.truncate(x.len() + tail_len / 2);
    y
}

/// Render one vocalization. Returns mono f32 audio at [`SR`], peak-normalized to 0.9.
///
/// `distance` (m) darkens the sound by air absorption and sets the reverb mix; `reverb` scales
/// the built-in outdoor reverb (0 = dry, for when the game's mixer adds environment reverb).
pub fn create_vocalization(body: &Body, voice: &Voice, intent: &str, seed: u64, distance: f64, reverb: f64) -> Result<Vec<f32>, String> {
    if !INTENT_NAMES.contains(&intent) {
        return Err(format!("unknown intent {intent:?}; choose from {INTENT_NAMES:?}"));
    }
    body.validate()?;
    voice.validate()?;
    let mut rng = Rng::new(seed);
    let calls = build_calls(body, voice, intent, &mut rng)?;
    if calls.is_empty() {
        return Ok(Vec::new());
    }
    let mut rendered: Vec<Vec<f64>> = calls
        .iter()
        .map(|c| match c.consonant {
            Some(_) if c.voice_bar => render_voice_bar(c, body, &mut rng),
            Some(cons) if cons.is_stop() => render_burst(c, cons, body, &mut rng),
            Some(cons) if cons != Consonant::H => render_fricative(c, cons, body, &mut rng),
            _ => render_call(c, body, &mut rng),
        })
        .collect();

    // Sequential syllables: quiet vowels brought part way up to the others.
    let even_ref = calls.iter().zip(&rendered).filter(|(c, _)| c.even > 0.0).map(|(c, y)| rms(y) / c.level).fold(0.0, f64::max);
    if even_ref > 0.0 {
        for (c, y) in calls.iter().zip(rendered.iter_mut()) {
            if c.even > 0.0 {
                let g = (even_ref * c.level / (rms(y) + 1e-12)).powf(c.even);
                y.iter_mut().for_each(|v| *v *= g);
            }
        }
    }

    // Consonants are set against the loudest syllable.
    let syllable_peak = calls
        .iter()
        .zip(&rendered)
        .filter(|(c, _)| c.consonant.is_none() && c.voiced > 0.0)
        .map(|(_, y)| rms(y))
        .fold(0.0, f64::max);
    for (c, y) in calls.iter().zip(rendered.iter_mut()) {
        if let (Some(_), true) = (c.consonant, syllable_peak > 0.0) {
            let g = syllable_peak * 10f64.powf(c.consonant_db / 20.0) / (rms(y) + 1e-12);
            y.iter_mut().for_each(|v| *v *= g);
        }
    }

    // Inhales are set against the calls they precede, not in absolute terms.
    let loudest = calls.iter().zip(&rendered).filter(|(c, _)| c.voiced > 0.0).map(|(_, y)| rms(y)).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
    if let (Some(loudest), true) = (loudest, body.source != Source::Hiss) {
        for (c, y) in calls.iter().zip(rendered.iter_mut()) {
            if c.voiced <= 0.0 && c.consonant.is_none() {
                let g = INHALE_LEVEL * loudest / rms(y);
                y.iter_mut().for_each(|v| *v *= g);
            }
        }
    }

    let start = |c: &Call| (c.start * SRF).max(0.0) as usize;
    let total = calls.iter().zip(&rendered).map(|(c, y)| start(c) + y.len()).max().unwrap_or(0);
    let mut mix = vec![0.0; total];
    for (c, y) in calls.iter().zip(&rendered) {
        let s = start(c);
        for (k, v) in y.iter().enumerate() {
            mix[s + k] += v;
        }
    }

    // Nothing useful sits well below the voice's own pitch; there it only adds rumble and mud.
    dsp::butter_highpass(2, (0.35 * body.base_f0()).max(30.0)).process(&mut mix);
    let p = dsp::peak(&mix) + 1e-9;
    let peak_harsh = calls
        .iter()
        .map(|c| clip(c.harsh.iter().map(|&(_, v)| v).fold(f64::MIN, f64::max), 0.0, 1.0))
        .fold(0.0, f64::max);
    let drive = 1.0 + 0.8 * peak_harsh;
    let td = drive.tanh();
    mix.iter_mut().for_each(|v| *v = (drive * *v / p).tanh() / td); // tissue/airflow saturation grit

    // Air absorbs highs over distance; farther sources are also more reverberant.
    let cutoff = (16000.0 * (-distance / 150.0).exp()).max(1500.0);
    dsp::butter_lowpass(2, cutoff.min(0.45 * SRF)).process(&mut mix);
    let wet = reverb * (0.06 + 0.002 * distance).min(0.35);
    if wet > 0.0 {
        mix = outdoor_reverb(&mix, &mut rng, wet);
    }
    let p = dsp::peak(&mix) + 1e-9;
    Ok(mix.iter().map(|v| (0.9 * v / p) as f32).collect())
}
