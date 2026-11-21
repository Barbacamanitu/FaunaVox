//! Mouth track: how a creature's mouth moves while it makes a vocalization, for animating it.
//!
//! The track is read from the same calls the audio is rendered from (same body, voice, intent and
//! seed), so it lines up with `create_vocalization`'s audio sample for sample: frame `i` is at
//! `i / fps` seconds into the audio. It needs no rendering, so it costs well under a millisecond.
//!
//! Channels, each 0..1:
//! - `jaw`: jaw or beak opening. Follows the call's mouth curve, how open the vowel is (its F1),
//!   and the loudness, so the mouth settles as a sound fades.
//! - `round`: lips pushed forward into a small round opening (u, o, sh).
//! - `close`: lips pressed together while the jaw may stay part open (p and b), or the lower lip
//!   against the upper teeth (f, at 0.6).
//! - `snarl`: upper lip lifted off the teeth, corners pulled back (the voice's `spread`, and a
//!   little for s and th).
//! - `loud`: loudness, 1 at the loudest moment. Handy for throat, chest or ear motion.
//!
//! A creature without lips (a beak) can ignore `round`, `close` and `snarl`.

use crate::dsp::{curve, interp};
use crate::model::{Body, Consonant, Voice};
use crate::rng::Rng;
use crate::synth::{build_calls, Call, INTENT_NAMES};

pub const CHANNELS: [&str; 5] = ["jaw", "round", "close", "snarl", "loud"];

/// Internal rate the track is built at before smoothing and resampling to the caller's fps.
const RATE: f64 = 500.0;
/// Coarticulation: the mouth moves toward a sound a little before it is heard and leaves it a
/// little after. Gaussian smoothing widths (s) for the shape channels and for lip closure.
const SMOOTH_SHAPE: f64 = 0.022;
const SMOOTH_CLOSE: f64 = 0.010;
const SMOOTH_LOUD: f64 = 0.008;
/// Frames past the last call, so the mouth has time to close.
const TAIL: f64 = 0.15;

#[derive(Clone, Debug)]
pub struct MouthTrack {
    pub fps: f64,
    /// One `[jaw, round, close, snarl, loud]` per frame.
    pub frames: Vec<[f32; 5]>,
}

/// How open a vowel is, from its F1 ratio: i and u ~0, schwa ~0.5, a ~1.
fn openness(f1_ratio: f64) -> f64 {
    ((f1_ratio - 0.5) / 0.95).clamp(0.0, 1.0)
}

/// One call's mouth shape at `u` (0..1 through the call): ([jaw, round, close, snarl], loud, weight).
fn shape(c: &Call, u: f64, amp_peak: f64) -> ([f64; 4], f64, f64) {
    let nv = c.vowels.len().max(1);
    let pos: Vec<f64> = if nv > 1 { (0..nv).map(|k| k as f64 / (nv - 1) as f64).collect() } else { vec![0.0] };
    let f1s: Vec<f64> = c.vowels.iter().map(|v| v.ratios()[0]).collect();
    let rounds: Vec<f64> = c.vowels.iter().map(|v| v.rounding()).collect();
    let (f1, rounding) = if c.vowels.is_empty() { (1.0, 0.0) } else { (interp(u, &pos, &f1s), interp(u, &pos, &rounds)) };
    let open = openness(f1);
    let mouth = curve(&c.mouth, &[u])[0].clamp(0.0, 1.0);
    let amp = (curve(&c.amp, &[u])[0] / amp_peak.max(1e-9)).clamp(0.0, 1.0);
    let spread = c.spread.clamp(0.0, 1.0);
    let round = rounding * (1.0 - spread);
    let db = |d: f64| 10f64.powf(d / 20.0);

    match c.consonant {
        None if c.voiced <= 0.0 => {
            // an inhale: the mouth part open, silent
            ([0.3 * mouth.max(0.5), 0.0, 0.0, 0.0], 0.0, 0.5)
        }
        None => {
            let jaw = mouth * (0.3 + 0.7 * open) * amp.sqrt();
            let loud = curve(&c.amp, &[u])[0].max(0.0) * c.level;
            ([jaw, round, 0.0, spread], loud, 0.2 + amp)
        }
        Some(cons) => {
            let loud = curve(&c.amp, &[u])[0].max(0.0) * db(c.consonant_db);
            let vowel_jaw = 0.3 + 0.7 * open;
            match cons {
                // breath shaped like the vowel beside it
                Consonant::H => ([0.6 * vowel_jaw, round, 0.0, spread], loud, 0.4),
                Consonant::S | Consonant::Th => ([0.12, 0.3 * round, 0.0, spread.max(0.25)], loud, 3.0),
                Consonant::Sh | Consonant::Ch | Consonant::Jh => ([0.15, round.max(0.55), 0.0, spread], loud, 3.0),
                Consonant::F => ([0.12, 0.2 * round, 0.6, spread], loud, 3.0),
                // a stop's burst: the closure opening again
                Consonant::P | Consonant::B => ([0.1 + 0.3 * u * vowel_jaw, round, 1.0 - u, spread], loud, 4.0),
                Consonant::T | Consonant::D | Consonant::K | Consonant::G => ([0.25 * vowel_jaw, round, 0.0, spread], loud, 3.0),
            }
        }
    }
}

/// The closure before a stop's burst: lips shut for p and b; the tongue for the others, with the
/// jaw part closed.
fn closure_shape(c: &Call) -> [f64; 4] {
    let open = c.vowels.first().map_or(0.5, |v| openness(v.ratios()[0]));
    let round = c.vowels.first().map_or(0.0, |v| v.rounding()) * (1.0 - c.spread.clamp(0.0, 1.0));
    match c.consonant {
        Some(Consonant::P | Consonant::B) => [0.1, round, 1.0, 0.0],
        _ => [0.2 * (0.3 + 0.7 * open), round, 0.0, c.spread.clamp(0.0, 1.0)],
    }
}

fn gaussian_smooth(x: &[f64], sigma: f64) -> Vec<f64> {
    let s = sigma * RATE;
    if s < 0.5 {
        return x.to_vec();
    }
    let r = (3.0 * s).ceil() as isize;
    let k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / s).powi(2)).exp()).collect();
    let n = x.len() as isize;
    (0..n)
        .map(|i| {
            let (mut a, mut w) = (0.0, 0.0);
            for (j, kv) in k.iter().enumerate() {
                let t = i + j as isize - r;
                // outside the track the mouth is at rest (0)
                if (0..n).contains(&t) {
                    a += kv * x[t as usize];
                }
                w += kv;
            }
            a / w
        })
        .collect()
}

/// The mouth track for the calls a vocalization is rendered from.
pub fn track_from_calls(calls: &[Call], fps: f64) -> MouthTrack {
    let end = calls.iter().map(|c| c.start + c.dur).fold(0.0, f64::max);
    let n = ((end + TAIL) * RATE).ceil() as usize + 1;
    let mut acc = vec![[0.0f64; 4]; n];
    let mut weight = vec![0.0f64; n];
    let mut loud = vec![0.0f64; n];
    let mut closed = vec![None::<([f64; 4], f64)>; n];

    for c in calls {
        let amp_peak = c.amp.iter().map(|p| p.1).fold(0.0, f64::max);
        let i0 = (c.start * RATE).floor().max(0.0) as usize;
        let i1 = (((c.start + c.dur) * RATE).ceil() as usize).min(n - 1);
        for i in i0..=i1 {
            let u = ((i as f64 / RATE - c.start) / c.dur.max(1e-6)).clamp(0.0, 1.0);
            let (s, l, w) = shape(c, u, amp_peak);
            for k in 0..4 {
                acc[i][k] += w * s[k];
            }
            weight[i] += w;
            loud[i] += l;
        }
        if c.closure > 0.0 {
            // the silent closure before a stop's burst outweighs anything overlapping it
            let shape = closure_shape(c);
            let i0 = ((c.start - c.closure) * RATE).floor().max(0.0) as usize;
            let i1 = ((c.start * RATE).ceil() as usize).min(n - 1);
            for slot in closed.iter_mut().take(i1 + 1).skip(i0) {
                *slot = Some((shape, 10.0));
            }
        }
    }

    let mut ch: Vec<Vec<f64>> = (0..4)
        .map(|k| {
            (0..n)
                .map(|i| {
                    let (mut a, mut w) = (acc[i][k], weight[i]);
                    if let Some((s, cw)) = closed[i] {
                        a += cw * s[k];
                        w += cw;
                    }
                    if w > 0.0 { a / w } else { 0.0 }
                })
                .collect()
        })
        .collect();
    // fade toward rest where only a little weight is present (gaps between calls)
    for i in 0..n {
        let w = weight[i] + closed[i].map_or(0.0, |c| c.1);
        let g = (w / 0.2).min(1.0);
        for c in ch.iter_mut() {
            c[i] *= g;
        }
    }
    for (k, c) in ch.iter_mut().enumerate() {
        *c = gaussian_smooth(c, if k == 2 { SMOOTH_CLOSE } else { SMOOTH_SHAPE });
    }
    let mut loud = gaussian_smooth(&loud, SMOOTH_LOUD);
    let peak = loud.iter().cloned().fold(0.0, f64::max);
    if peak > 0.0 {
        loud.iter_mut().for_each(|v| *v /= peak);
    }

    let frames = ((end + TAIL) * fps).ceil() as usize + 1;
    let at = |x: &[f64], t: f64| {
        let p = t * RATE;
        let i = (p.floor() as usize).min(n - 1);
        let j = (i + 1).min(n - 1);
        let f = p - i as f64;
        x[i] * (1.0 - f) + x[j] * f
    };
    MouthTrack {
        fps,
        frames: (0..frames)
            .map(|f| {
                let t = f as f64 / fps;
                let v = |x: &[f64]| at(x, t).clamp(0.0, 1.0) as f32;
                [v(&ch[0]), v(&ch[1]), v(&ch[2]), v(&ch[3]), v(&loud)]
            })
            .collect(),
    }
}

/// The mouth track for `create_vocalization(body, voice, intent, seed, ..)`, at `fps` frames a
/// second. Distance and reverb don't move anything in time, so they aren't needed.
pub fn mouth_track(body: &Body, voice: &Voice, intent: &str, seed: u64, fps: f64) -> Result<MouthTrack, String> {
    if !INTENT_NAMES.contains(&intent) {
        return Err(format!("unknown intent {intent:?}; choose from {INTENT_NAMES:?}"));
    }
    if !(fps > 0.0 && fps <= 1000.0) {
        return Err(format!("fps must be in (0, 1000], got {fps}"));
    }
    body.validate()?;
    voice.validate()?;
    let mut rng = Rng::new(seed);
    let calls = build_calls(body, voice, intent, &mut rng)?;
    Ok(track_from_calls(&calls, fps))
}

impl MouthTrack {
    pub fn to_json(&self) -> String {
        let frames: Vec<String> = self
            .frames
            .iter()
            .map(|f| format!("[{}]", f.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(",")))
            .collect();
        format!(
            "{{\"fps\": {}, \"channels\": [{}], \"frames\": [\n{}\n]}}\n",
            self.fps,
            CHANNELS.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", "),
            frames.join(",\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::VowelSpec;

    fn speak(phrase: &str) -> (Body, Voice) {
        let body = Body { mass: 30.0, ..Body::default() };
        let mut voice = Voice::default();
        voice.vowels = vec![VowelSpec::Name(phrase.into())];
        (body, voice)
    }

    fn track(phrase: &str) -> MouthTrack {
        let (b, v) = speak(phrase);
        mouth_track(&b, &v, "speak", 3, 100.0).unwrap()
    }

    #[test]
    fn deterministic_and_bounded() {
        let a = track("a i~u | o");
        let b = track("a i~u | o");
        assert_eq!(a.frames, b.frames);
        assert!(a.frames.iter().flatten().all(|v| (0.0..=1.0).contains(v)));
        // starts and ends at rest
        assert!(a.frames[0][0] < 0.2 && a.frames.last().unwrap()[0] < 0.05);
    }

    #[test]
    fn open_vowel_opens_jaw_more_than_closed() {
        let peak = |t: &MouthTrack, k: usize| t.frames.iter().map(|f| f[k]).fold(0.0, f32::max);
        assert!(peak(&track("a"), 0) > 1.5 * peak(&track("i"), 0));
        assert!(peak(&track("u"), 1) > 0.6 && peak(&track("i"), 1) < 0.05);
    }

    #[test]
    fn p_closes_lips() {
        let peak = |t: &MouthTrack, k: usize| t.frames.iter().map(|f| f[k]).fold(0.0, f32::max);
        assert!(peak(&track("a p a"), 2) > 0.7);
        assert!(peak(&track("a t a"), 2) < 0.05);
    }

    #[test]
    fn covers_the_audio() {
        let (b, v) = speak("a i u");
        let audio = crate::synth::create_vocalization(&b, &v, "speak", 3, 10.0, 0.0).unwrap();
        let t = mouth_track(&b, &v, "speak", 3, 60.0).unwrap();
        let secs = audio.len() as f64 / crate::synth::SR as f64;
        let track_secs = t.frames.len() as f64 / t.fps;
        assert!((track_secs - secs).abs() < 0.4, "track {track_secs} s vs audio {secs} s");
    }
}
