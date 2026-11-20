//! cvsynth: render creature vocalizations to WAV files.
//!
//! Run `cvsynth --help` for usage.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use faunavox::{create_vocalization, english_to_phrase, presets, Creature, VowelSpec, INTENT_NAMES, SR};

#[path = "cvsynth/space_cli.rs"]
mod space_cli;
#[path = "cvsynth/genome_cli.rs"]
mod genome_cli;

const HELP: &str = "\
cvsynth - procedural creature vocalizations (native port of create_vocalization.py)

USAGE:
    cvsynth [--preset NAME | --json FILE | --all-presets] [OPTIONS]

CREATURE:
    --preset NAME        built-in creature (default: lion). See --list-presets
    --json FILE          creature JSON: {\"body\": {...}, \"voice\": {...}}, or a map of
                         name -> creature to render several at once
    --all-presets        render every built-in preset
    --set FIELD=VALUE    override a Body/Voice field; VALUE is JSON or a bare word
                         (repeatable), e.g. --set mass=40 --set vowels='[\"u\",\"o\"]'
                         --set source=syrinx
    --vowels LIST        shorthand for the vowel list, e.g. i,a,u
    --say TEXT           English text for the speak intent: translated to vowels and
                         consonants (CMU dictionary) and set as calls.speak.vowels;
                         renders speak unless --intent is given
    --translate TEXT     print the phrase English text translates to, and exit

RENDERING:
    --intent LIST        comma-separated intents, or 'all' (default: idle)
                         idle warn alarm flee mating pain beg territorial speak
    --seed N             random seed; the same seed always gives the same sound (default: 1)
    --variants N         render N takes per intent with seeds SEED..SEED+N-1 (default: 1)
    --distance M         listener distance in metres: air absorption + reverb mix (default: 20)
    --reverb X           built-in outdoor reverb amount, 0 = dry (default: 1)

OUTPUT:
    --out FILE           output file when rendering a single sound (default: vocalization.wav)
    --out-dir DIR        output folder when rendering several (default: vocalizations)
                         files are named <creature>_<intent>[_s<seed>].wav
    --format F           pcm16 | float32 (default: pcm16)
    --demo               same as --intent all
    --threads N          worker threads for batches (default: all cores)

COMMANDS:
    cvsynth sample ...   render a batch of voices from voice space for rating (see cvsynth sample --help)
    cvsynth sweep ...    render each voice-space axis from -3 to +3 (see cvsynth sweep --help)
    cvsynth genome ...   decode a voice from DNA, render it, or run the evolvability checks (see cvsynth genome --help)

OTHER:
    --list-presets       print preset names
    --print-json         print the resolved creature JSON and exit
    -h, --help           show this help

EXAMPLES:
    cvsynth --preset wolf --intent mating --out wolf.wav
    cvsynth --json creature.json --intent all --variants 4 --reverb 0 --out-dir cache
    cvsynth --all-presets --demo
    cvsynth --preset wolf --say \"Hey, Heath!\" --out hey.wav
";

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Format {
    Pcm16,
    Float32,
}

struct Args {
    preset: Option<String>,
    json: Option<PathBuf>,
    all_presets: bool,
    sets: Vec<(String, String)>,
    vowels: Option<String>,
    say: Option<String>,
    intent_given: bool,
    intents: Vec<String>,
    seed: u64,
    variants: u64,
    distance: f64,
    reverb: f64,
    out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    format: Format,
    threads: usize,
    list_presets: bool,
    print_json: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        preset: None, json: None, all_presets: false, sets: vec![], vowels: None, say: None, intent_given: false,
        intents: vec!["idle".into()], seed: 1, variants: 1, distance: 20.0, reverb: 1.0,
        out: None, out_dir: None, format: Format::Pcm16,
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        list_presets: false, print_json: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        // Accept both "--flag value" and "--flag=value".
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut val = |name: &str| -> Result<String, String> {
            inline.clone().or_else(|| it.next()).ok_or_else(|| format!("{name} needs a value"))
        };
        let num = |s: String, name: &str| -> Result<f64, String> { s.parse::<f64>().map_err(|_| format!("{name}: not a number: {s}")) };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--preset" => a.preset = Some(val("--preset")?),
            "--json" => a.json = Some(PathBuf::from(val("--json")?)),
            "--all-presets" => a.all_presets = true,
            "--set" => {
                let s = val("--set")?;
                let (k, v) = s.split_once('=').ok_or_else(|| format!("--set expects FIELD=VALUE, got {s}"))?;
                a.sets.push((k.trim().to_string(), v.trim().to_string()));
            }
            "--vowels" => a.vowels = Some(val("--vowels")?),
            "--say" => a.say = Some(val("--say")?),
            "--translate" => {
                println!("{}", english_to_phrase(&val("--translate")?));
                std::process::exit(0);
            }
            "--intent" => {
                let s = val("--intent")?;
                a.intent_given = true;
                a.intents = if s == "all" {
                    INTENT_NAMES.iter().map(|s| s.to_string()).collect()
                } else {
                    s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
                };
            }
            "--demo" => a.intents = INTENT_NAMES.iter().map(|s| s.to_string()).collect(),
            "--seed" => a.seed = val("--seed")?.parse().map_err(|_| "--seed must be a non-negative integer")?,
            "--variants" => a.variants = val("--variants")?.parse().map_err(|_| "--variants must be a positive integer")?,
            "--distance" => a.distance = num(val("--distance")?, "--distance")?,
            "--reverb" => a.reverb = num(val("--reverb")?, "--reverb")?,
            "--out" => a.out = Some(PathBuf::from(val("--out")?)),
            "--out-dir" => a.out_dir = Some(PathBuf::from(val("--out-dir")?)),
            "--format" => {
                a.format = match val("--format")?.as_str() {
                    "pcm16" | "16" => Format::Pcm16,
                    "float32" | "float" | "32f" => Format::Float32,
                    f => return Err(format!("--format must be pcm16 or float32, got {f}")),
                }
            }
            "--threads" => a.threads = val("--threads")?.parse().map_err(|_| "--threads must be a positive integer")?,
            "--list-presets" => a.list_presets = true,
            "--print-json" => a.print_json = true,
            other => return Err(format!("unknown argument {other} (see --help)")),
        }
    }
    for i in &a.intents {
        if !INTENT_NAMES.contains(&i.as_str()) {
            return Err(format!("unknown intent {i:?}; choose from {}", INTENT_NAMES.join(", ")));
        }
    }
    if a.say.is_some() && !a.intent_given {
        a.intents = vec!["speak".into()];
    }
    if a.variants == 0 {
        return Err("--variants must be at least 1".into());
    }
    a.threads = a.threads.max(1);
    Ok(a)
}

fn load_creatures(a: &Args) -> Result<Vec<(String, Creature)>, String> {
    let mut list: Vec<(String, Creature)> = if let Some(path) = &a.json {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "creature".into());
        let single = value.get("body").is_some() || value.get("voice").is_some();
        if single {
            let c = Creature::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            vec![(c.name.clone().unwrap_or(stem), c)]
        } else {
            let obj = value.as_object().ok_or_else(|| format!("{}: expected a creature or a map of creatures", path.display()))?;
            let mut v = Vec::new();
            for (name, cv) in obj {
                let c = Creature::from_json(&cv.to_string()).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
                v.push((name.clone(), c));
            }
            v
        }
    } else if a.all_presets {
        presets::all().into_iter().collect()
    } else {
        let name = a.preset.clone().unwrap_or_else(|| "lion".into());
        let c = presets::get(&name).ok_or_else(|| format!("unknown preset {name:?}; choose from {}", presets::names().join(", ")))?;
        vec![(name, c)]
    };

    for (name, c) in list.iter_mut() {
        for (field, raw) in &a.sets {
            // A bare word (syrinx) is taken as a string; anything else must be valid JSON.
            let value = serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.clone()));
            c.set_field(field, value).map_err(|e| format!("{name}: --set {field}: {e}"))?;
        }
        if let Some(text) = &a.say {
            let phrase = english_to_phrase(text);
            if phrase.is_empty() {
                return Err(format!("--say: nothing in {text:?} can be voiced"));
            }
            eprintln!("  say: {phrase}");
            c.voice.calls.entry("speak".into()).or_default().vowels = Some(vec![VowelSpec::Name(phrase)]);
        }
        if let Some(vs) = &a.vowels {
            let list: Vec<serde_json::Value> = vs.split(',').map(|v| serde_json::Value::String(v.trim().to_string())).collect();
            c.set_field("vowels", serde_json::Value::Array(list)).map_err(|e| format!("{name}: --vowels: {e}"))?;
        }
    }
    Ok(list)
}

pub(crate) fn write_wav(path: &Path, audio: &[f32], format: Format) -> io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let (fmt_tag, bits, fmt_len): (u16, u16, u32) = match format {
        Format::Pcm16 => (1, 16, 16),
        Format::Float32 => (3, 32, 18),
    };
    let bytes_per = (bits / 8) as u32;
    let data_len = audio.len() as u32 * bytes_per;
    let fact_len = if format == Format::Float32 { 12 } else { 0 };
    let mut w: Vec<u8> = Vec::with_capacity(data_len as usize + 64);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(4 + 8 + fmt_len + fact_len + 8 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&fmt_len.to_le_bytes());
    w.extend_from_slice(&fmt_tag.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&(SR as u32).to_le_bytes());
    w.extend_from_slice(&(SR as u32 * bytes_per).to_le_bytes());
    w.extend_from_slice(&(bytes_per as u16).to_le_bytes());
    w.extend_from_slice(&bits.to_le_bytes());
    if format == Format::Float32 {
        w.extend_from_slice(&0u16.to_le_bytes()); // cbSize
        w.extend_from_slice(b"fact");
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&(audio.len() as u32).to_le_bytes());
    }
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for &s in audio {
        match format {
            Format::Pcm16 => w.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()),
            Format::Float32 => w.extend_from_slice(&s.to_le_bytes()),
        }
    }
    fs::write(path, w)
}

fn sanitize(name: &str) -> String {
    name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    if a.list_presets {
        for n in presets::names() {
            println!("{n}");
        }
        return Ok(());
    }
    let creatures = load_creatures(&a)?;
    if a.print_json {
        for (name, c) in &creatures {
            let mut c = c.clone();
            c.name = Some(name.clone());
            println!("{}", c.to_json());
        }
        return Ok(());
    }

    struct Job {
        creature: usize,
        intent: String,
        seed: u64,
        path: PathBuf,
    }
    let mut jobs = Vec::new();
    for (ci, (name, _)) in creatures.iter().enumerate() {
        for intent in &a.intents {
            for v in 0..a.variants {
                jobs.push(Job { creature: ci, intent: intent.clone(), seed: a.seed + v, path: PathBuf::new() });
            }
        }
        let _ = name;
    }
    let single = jobs.len() == 1 && a.out_dir.is_none();
    let out_dir = a.out_dir.clone().unwrap_or_else(|| PathBuf::from("vocalizations"));
    for j in jobs.iter_mut() {
        j.path = if single {
            a.out.clone().unwrap_or_else(|| PathBuf::from("vocalization.wav"))
        } else {
            let name = sanitize(&creatures[j.creature].0);
            let suffix = if a.variants > 1 { format!("_s{}", j.seed) } else { String::new() };
            out_dir.join(format!("{name}_{}{suffix}.wav", j.intent))
        };
    }

    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let print = Mutex::new(());
    std::thread::scope(|s| {
        for _ in 0..a.threads.min(jobs.len()) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(k) else { break };
                let (name, c) = &creatures[job.creature];
                let t0 = Instant::now();
                let result = create_vocalization(&c.body, &c.voice, &job.intent, job.seed, a.distance, a.reverb)
                    .and_then(|audio| write_wav(&job.path, &audio, a.format).map(|_| audio.len()).map_err(|e| format!("{}: {e}", job.path.display())));
                let _lock = print.lock();
                match result {
                    Ok(len) => {
                        println!("{} ({:.1}s audio, {:.0} ms)", job.path.display(), len as f64 / SR as f64, t0.elapsed().as_secs_f64() * 1000.0);
                        if single {
                            let b = &c.body;
                            println!("  {name}: {:?}, {} kg, tract {:.3} m, base f0 {:.0} Hz", b.source, b.mass, b.tract(), b.base_f0());
                        }
                    }
                    Err(e) => errors.lock().unwrap().push(format!("{name} {}: {e}", job.intent)),
                }
                let _ = io::stdout().flush();
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if jobs.len() > 1 {
        println!("{} file(s) in {:.2}s", jobs.len() - errors.len(), started.elapsed().as_secs_f64());
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("sample") => space_cli::sample(&args[1..]),
        Some("sweep") => space_cli::sweep(&args[1..]),
        Some("genome") => genome_cli::genome(&args[1..]),
        _ => run(),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
