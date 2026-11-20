//! `cvsynth sample` and `cvsynth sweep`: renders from voice space (faunavox::space) for the
//! rating tool and for checking the axes. See GENOME_VOICE_DESIGN.md §1-2.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use faunavox::model::{Source, VowelSpec};
use faunavox::rng::Rng;
use faunavox::space::{self, BodyInput, AXES, BODY_KINDS, SETTINGS, SPACE_VERSION};
use faunavox::{create_vocalization, INTENT_NAMES, SR};
use serde_json::{json, Value};

use super::{write_wav, Format};

pub const HELP: &str = "\
cvsynth sample - render a batch of voices from voice space for rating

USAGE:
    cvsynth sample --count N --round NAME --out-dir DIR [OPTIONS]
    cvsynth sample --points FILE --round NAME --out-dir DIR [OPTIONS]

OPTIONS:
    --count N        broad samples, plus about 10 % paired and 5 % repeats on top (default: 300)
    --points FILE    render these positions instead (JSON lines: body, axes, residual, intent)
    --round NAME     batch name, used in sample ids (default: r001)
    --seed N         random seed for the batch (default: 1)
    --intents LIST   intents to rate, spread evenly (default: idle,mating,warn)
    --vowels LIST    the creature's vowels (default: a,o)
    --out-dir DIR    where manifest.jsonl and the WAVs go

cvsynth sweep - render each axis from -3 to +3 on a few bodies, to check the axes by ear

USAGE:
    cvsynth sweep --out-dir DIR [--intent NAME] [--seed N] [--per-setting]

    One WAV per axis and body: seven calls at -3, -2, -1, 0, 1, 2, 3, each loudness-matched,
    1.5 s apart. --per-setting sweeps each of the 29 settings alone instead, over nearly its whole
    range, on two more bodies (a pulse-voiced bug and a hissing snake).
";

/// Every take is brought to this RMS (dBFS), then pulled down if its peak would clip.
const TARGET_RMS_DB: f64 = -20.0;
/// Listening distance and reverb for every rated take.
const DISTANCE: f64 = 10.0;
const REVERB: f64 = 0.5;
/// A sample this far out on the instability axis gets a second take with another seed: its
/// randomness can make one render unrepresentative.
const TWO_TAKE_INSTABILITY: f64 = 1.0;

fn loudness_matched(audio: &[f32]) -> Vec<f32> {
    let rms = (audio.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / audio.len().max(1) as f64).sqrt();
    let peak = audio.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    if rms <= 0.0 {
        return audio.to_vec();
    }
    let g = (10f64.powf(TARGET_RMS_DB / 20.0) / rms).min(0.95 / peak.max(1e-9));
    audio.iter().map(|v| (*v as f64 * g) as f32).collect()
}

struct Args {
    count: usize,
    points: Option<PathBuf>,
    round: String,
    seed: u64,
    intents: Vec<String>,
    vowels: Vec<VowelSpec>,
    out_dir: Option<PathBuf>,
    intent: String,
    per_setting: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        count: 300, points: None, round: "r001".into(), seed: 1,
        intents: vec!["idle".into(), "mating".into(), "warn".into()],
        vowels: vec![VowelSpec::Name("a".into()), VowelSpec::Name("o".into())],
        out_dir: None, intent: "idle".into(), per_setting: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut val = |name: &str| inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| format!("{name} needs a value"));
        let list = |s: String| -> Vec<String> { s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect() };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--count" => a.count = val("--count")?.parse().map_err(|_| "--count: not a number")?,
            "--points" => a.points = Some(PathBuf::from(val("--points")?)),
            "--round" => a.round = val("--round")?,
            "--seed" => a.seed = val("--seed")?.parse().map_err(|_| "--seed: not a number")?,
            "--intents" => a.intents = list(val("--intents")?),
            "--intent" => a.intent = val("--intent")?,
            "--vowels" => a.vowels = list(val("--vowels")?).into_iter().map(VowelSpec::Name).collect(),
            "--out-dir" => a.out_dir = Some(PathBuf::from(val("--out-dir")?)),
            "--per-setting" => a.per_setting = true,
            other => return Err(format!("unknown option {other} (see cvsynth sample --help)")),
        }
    }
    for i in a.intents.iter().chain([&a.intent]) {
        if !INTENT_NAMES.contains(&i.as_str()) {
            return Err(format!("unknown intent {i:?}"));
        }
    }
    Ok(a)
}

/// One sample to render: a position on a body, heard as one intent.
#[derive(Clone)]
struct Sample {
    id: String,
    body: BodyInput,
    axes: [f64; 10],
    residual: [f64; 29],
    intent: String,
    seed: u64,
    kind: &'static str,
    pair_of: Option<String>,
    repeat_of: Option<String>,
}

fn kind_of(source: Source) -> (Source, &'static str, f64, f64, f64, bool, bool) {
    *BODY_KINDS.iter().find(|k| k.0 == source).expect("body kind")
}

/// The same body-relative position on a body of the same kind at least 8x heavier or lighter, with
/// the same tract proportion, so the pair separates "bad sound" from "wrong for this size".
fn paired_body(b: &BodyInput, rng: &mut Rng) -> BodyInput {
    let kind = kind_of(b.source);
    let (lo, hi) = (kind.3.ln(), kind.4.ln());
    let m = b.mass.ln();
    let shift = rng.uniform(8f64.ln(), 30f64.ln());
    let up = m + shift <= hi && (m - shift < lo || rng.random() < 0.5);
    let mass = if up { m + shift } else { m - shift }.clamp(lo, hi).exp();
    let factor = b.tract() / (0.06 * b.mass.cbrt());
    space::body_of(kind, mass, factor)
}

fn batch(a: &Args) -> Result<Vec<Sample>, String> {
    let mut rng = Rng::new(a.seed);
    let mut out: Vec<Sample> = Vec::new();
    let id = |n: usize| format!("{}-s{:04}", a.round, n + 1);
    if let Some(path) = &a.points {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let v: Value = serde_json::from_str(line).map_err(|e| format!("{} line {}: {e}", path.display(), n + 1))?;
            let body: BodyInput = serde_json::from_value(v["body"].clone()).map_err(|e| format!("line {}: body: {e}", n + 1))?;
            let axes: [f64; 10] = serde_json::from_value(v["axes"].clone()).map_err(|e| format!("line {}: axes: {e}", n + 1))?;
            let residual: [f64; 29] = serde_json::from_value(v["residual"].clone()).unwrap_or([0.0; 29]);
            let intent = v["intent"].as_str().unwrap_or(&a.intents[n % a.intents.len()]).to_string();
            out.push(Sample { id: id(n), body, axes, residual, intent, seed: a.seed * 100_000 + n as u64, kind: "active", pair_of: None, repeat_of: None });
        }
        return Ok(out);
    }
    // Broad samples, with intents spread evenly.
    let mut broad = Vec::new();
    for n in 0..a.count {
        let body = space::sample_body(&mut rng);
        let (axes, residual) = space::sample_position(&mut rng);
        broad.push((body, axes, residual, a.intents[n % a.intents.len()].clone()));
    }
    // Pairs: the second of each pair goes in as its own sample.
    let pairs = (a.count as f64 * 0.05).round() as usize;
    let mut items: Vec<(BodyInput, [f64; 10], [f64; 29], String, &'static str, Option<usize>)> =
        broad.into_iter().map(|(b, x, r, i)| (b, x, r, i, "broad", None)).collect();
    for p in 0..pairs.min(items.len()) {
        let (b, x, r, i, _, _) = items[p].clone();
        let twin = paired_body(&b, &mut rng);
        items[p].4 = "paired";
        items.push((twin, x, r, i, "paired", Some(p)));
    }
    // Shuffle, then number them; pair partners are far apart by chance in a shuffled list of hundreds.
    let mut order: Vec<usize> = (0..items.len()).collect();
    for k in (1..order.len()).rev() {
        let j = rng.integer_inclusive(0, k as i64) as usize;
        order.swap(k, j);
    }
    let position: Vec<usize> = {
        let mut p = vec![0; items.len()];
        for (pos, &item) in order.iter().enumerate() {
            p[item] = pos;
        }
        p
    };
    for (pos, &item) in order.iter().enumerate() {
        let (b, x, r, i, kind, partner) = items[item].clone();
        // A pair's first sample points at its twin too, so either can be found from the other.
        let twin = match partner {
            Some(first) => Some(id(position[first])),
            None => items.iter().position(|it| it.5 == Some(item)).map(|t| id(position[t])),
        };
        out.push(Sample { id: id(pos), body: b, axes: x, residual: r, intent: i, seed: a.seed * 100_000 + pos as u64, kind, pair_of: twin, repeat_of: None });
    }
    // Repeats: a copy of an earlier sample, at least 40 places after it, without saying so.
    let repeats = (a.count as f64 * 0.05).round() as usize;
    let n0 = out.len();
    for _ in 0..repeats {
        if n0 < 60 {
            break;
        }
        let src = rng.integer_inclusive(0, (n0 - 50) as i64) as usize;
        let mut s = out[src].clone();
        s.kind = "repeat";
        s.repeat_of = Some(s.id.clone());
        s.pair_of = None;
        let at = rng.integer_inclusive((src + 40) as i64, out.len() as i64) as usize;
        out.insert(at, s);
    }
    for (pos, s) in out.iter_mut().enumerate() {
        s.id = id(pos);
    }
    Ok(out)
}

fn named(values: &[f64], names: &[&str]) -> Value {
    Value::Object(names.iter().zip(values).map(|(n, v)| (n.to_string(), json!((v * 1e4).round() / 1e4))).collect())
}

fn round4<const N: usize>(x: &[f64; N]) -> Vec<f64> {
    x.iter().map(|v| (v * 1e4).round() / 1e4).collect()
}

pub fn sample(args: &[String]) -> Result<(), String> {
    let a = parse(args)?;
    let out_dir = a.out_dir.clone().ok_or("--out-dir is required")?;
    fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let samples = batch(&a)?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let next = AtomicUsize::new(0);
    let lines: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(smp) = samples.get(k) else { break };
                match render_sample(smp, &a.vowels, &out_dir) {
                    Ok(line) => lines.lock().unwrap().push((k, line)),
                    Err(e) => errors.lock().unwrap().push(format!("{}: {e}", smp.id)),
                }
                if k % 25 == 0 {
                    print!(".");
                    let _ = io::stdout().flush();
                }
            });
        }
    });
    println!();
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let mut lines = lines.into_inner().unwrap();
    lines.sort_by_key(|l| l.0);
    let manifest = out_dir.join("manifest.jsonl");
    let text: String = lines.into_iter().map(|(_, l)| l + "\n").collect();
    fs::write(&manifest, text).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let kinds = |k: &str| samples.iter().filter(|s| s.kind == k).count();
    println!(
        "{} samples ({} broad, {} paired, {} repeats, {} active) -> {}",
        samples.len(), kinds("broad"), kinds("paired"), kinds("repeat"), kinds("active"), manifest.display()
    );
    Ok(())
}

fn render_sample(s: &Sample, vowels: &[VowelSpec], dir: &Path) -> Result<String, String> {
    let c = space::creature(&s.body, &s.axes, &s.residual, vowels.to_vec());
    let mut seeds = vec![s.seed];
    if s.axes[5] > TWO_TAKE_INSTABILITY {
        seeds.push(s.seed + 7919);
    }
    let mut takes = Vec::new();
    for (t, &seed) in seeds.iter().enumerate() {
        let audio = create_vocalization(&c.body, &c.voice, &s.intent, seed, DISTANCE, REVERB)?;
        let name = format!("{}_{}.wav", s.id, (b'a' + t as u8) as char);
        write_wav(&dir.join(&name), &loudness_matched(&audio), Format::Pcm16).map_err(|e| format!("{name}: {e}"))?;
        takes.push(name);
    }
    let values = space::settings(&s.body, &s.axes, &s.residual);
    let line = json!({
        "id": s.id,
        "space_version": SPACE_VERSION,
        "body": s.body,
        "tract": s.body.tract(),
        "axes": round4(&s.axes),
        "axes_named": named(&s.axes, &AXES),
        "residual": round4(&s.residual),
        "settings": named(&values, &SETTINGS),
        "intent": s.intent,
        "vowels": vowels,
        "seed": s.seed,
        "kind": s.kind,
        "pair_of": s.pair_of,
        "repeat_of": s.repeat_of,
        "takes": takes,
        "creature": serde_json::from_str::<Value>(&c.to_json()).unwrap_or(Value::Null),
        "render": {"distance": DISTANCE, "reverb": REVERB, "rms_db": TARGET_RMS_DB, "sample_rate": SR},
    });
    Ok(line.to_string())
}

/// Bodies for the sweeps: mouse, fox, wolf and bear sized mammals and a crow-sized bird; per-setting
/// sweeps add a frog-sized bug (pulse) and a snake (hiss), since some settings only act on those.
fn sweep_bodies(per_setting: bool) -> Vec<(&'static str, BodyInput)> {
    let larynx = kind_of(Source::Larynx);
    let mut v = vec![
        ("mouse", space::body_of(larynx, 0.03, 1.0)),
        ("fox", space::body_of(larynx, 6.0, 1.0)),
        ("wolf", space::body_of(larynx, 40.0, 1.0)),
        ("bear", space::body_of(larynx, 300.0, 1.0)),
        ("crow", space::body_of(kind_of(Source::Syrinx), 0.5, 1.0)),
    ];
    if per_setting {
        v.push(("frog", space::body_of(kind_of(Source::Pulse), 0.3, 1.0)));
        v.push(("snake", space::body_of(kind_of(Source::Hiss), 3.0, 1.0)));
    }
    v
}

pub fn sweep(args: &[String]) -> Result<(), String> {
    let a = parse(args)?;
    let out_dir = a.out_dir.clone().ok_or("--out-dir is required")?;
    fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    // An axis steps -3..3; a single setting's raw coordinate steps -6..6, nearly its whole range.
    let per = a.per_setting;
    let names: Vec<&str> = if per { SETTINGS.to_vec() } else { AXES.to_vec() };
    let jobs: Vec<(usize, &'static str, BodyInput)> =
        (0..names.len()).flat_map(|ax| sweep_bodies(per).into_iter().map(move |(n, b)| (ax, n, b))).collect();
    let next = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some((ax, name, body)) = jobs.get(k) else { break };
                let mut all: Vec<f32> = Vec::new();
                for step in -3..=3 {
                    let mut axes = [0.0; 10];
                    let mut residual = [0.0; 29];
                    if per {
                        residual[*ax] = 2.0 * step as f64;
                    } else {
                        axes[*ax] = step as f64;
                    }
                    let c = space::creature(body, &axes, &residual, a.vowels.clone());
                    match create_vocalization(&c.body, &c.voice, &a.intent, a.seed, DISTANCE, REVERB) {
                        Ok(audio) => {
                            all.extend(loudness_matched(&audio));
                            all.extend(std::iter::repeat(0.0).take((1.5 * SR as f64) as usize));
                        }
                        Err(e) => errors.lock().unwrap().push(format!("{} {name}: {e}", names[*ax])),
                    }
                }
                let path = out_dir.join(format!("{:02}_{}_{name}.wav", ax + 1, names[*ax]));
                if let Err(e) = write_wav(&path, &all, Format::Pcm16) {
                    errors.lock().unwrap().push(format!("{}: {e}", path.display()));
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    println!("{} sweeps -> {}", jobs.len(), out_dir.display());
    Ok(())
}
