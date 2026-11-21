//! `cvsynth genome`: decode creatures from voice DNA (faunavox::genome), render them, and
//! check how the decoder behaves under mutation. See GENOME_VOICE_DESIGN.md §4-6.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use faunavox::create_vocalization;
use faunavox::genome::{self, Report, Versions, CALL_INTENTS, MUTATIONS, SECTION_LENGTH};
use faunavox::model::Source;
use faunavox::rng::Rng;
use faunavox::space::{self, BodyInput, AXES};

use super::{write_mouth, write_wav, Format};

pub const HELP: &str = "\
cvsynth genome - decode a creature's voice from its DNA, and render it

USAGE:
    cvsynth genome [BODY] --dna ACGT...   [OPTIONS]
    cvsynth genome [BODY] --random N      [OPTIONS]
    cvsynth genome mutations [--count N] [--seed N] [--out FILE]
    cvsynth genome check [--count N] [--seed N]

BODY (default: a 12 kg larynx quadruped):
    --body FILE      body JSON, as Unity sends it: {\"mass\": 12.4, \"source\": \"larynx\", ...}
    --mass KG  --source larynx|syrinx|pulse|hiss  --shape NAME  --no-lips  --no-teeth
    With --random and no body given, each genome gets a random body.

GENOME:
    --dna STR        the voice section (A, C, G, T; anything else is skipped)
    --genome FILE    body and DNA together: {\"body\": {...}, \"dna\": \"ACGT...\"} (see examples/genomes)
    --random N       N random voice sections of the standard length (1600 bases)

OPTIONS:
    --seed N         random genomes and render seed (default: 1)
    --intents LIST   which calls to render: comma-separated, or 'all' (default: speak and the 7 calls)
    --out-dir DIR    render WAVs here, with each creature's DNA, JSON and decoder report
    --no-field       decode without the preference field (no steering)
    --json           print the decoded creature JSON
    --mouth FPS      also write each call's mouth movement to NAME_INTENT.mouth.json
    --lineage LIST   also decode descendants at these generations (e.g. 1,5,20): each generation
                     mutated as Unity does (Genome.MutateSection, MutationRates.Default)
    --rate X         scale those mutation rates (default 1)

cvsynth genome mutations - mutate N random genomes once with each kind of mutation, and print what
    changed: where it hit, the base voice before and after on the 10 axes and the distance between
    them, and each call whose phrase or timbre offset changed.

cvsynth genome check - evolvability checks (GENOME_VOICE_DESIGN.md §6): the spread of decoded
    voices and phrases, the size of each kind of mutation's effect, the field's steering, and
    determinism.
";

struct Args {
    body: Option<BodyInput>,
    dna: Option<String>,
    random: usize,
    seed: u64,
    intents: String,
    out_dir: Option<PathBuf>,
    field: bool,
    json: bool,
    mouth: Option<f64>,
    count: Option<usize>,
    lineage: Vec<usize>,
    rate: f64,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        body: None, dna: None, random: 0, seed: 1, intents: "all".into(), out_dir: None, field: true, json: false, mouth: None,
        count: None, lineage: vec![], rate: 1.0,
    };
    let mut b = BodyInput { mass: 12.0, ..BodyInput::default() };
    let mut body_given = false;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut val = |what: &str| it.next().cloned().ok_or_else(|| format!("{what} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--body" => {
                let path = val("--body")?;
                let text = fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                b = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
                body_given = true;
            }
            "--mass" => {
                b.mass = val("--mass")?.parse().map_err(|_| "--mass must be a number")?;
                b.tract_length = None;
                body_given = true;
            }
            "--source" => {
                b.source = serde_json::from_value(serde_json::Value::String(val("--source")?)).map_err(|_| "unknown --source")?;
                if b.source != Source::Larynx {
                    b.lips = false;
                }
                body_given = true;
            }
            "--shape" => {
                b.shape = val("--shape")?;
                body_given = true;
            }
            "--no-lips" => {
                b.lips = false;
                body_given = true;
            }
            "--no-teeth" => {
                b.teeth = false;
                body_given = true;
            }
            "--dna" => a.dna = Some(val("--dna")?),
            "--genome" => {
                // {"body": {...}, "dna": "ACGT..."}, as in examples/genomes
                let path = val("--genome")?;
                let text = fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
                b = serde_json::from_value(v.get("body").cloned().unwrap_or_default()).map_err(|e| format!("{path}: body: {e}"))?;
                body_given = true;
                a.dna = Some(v.get("dna").and_then(|d| d.as_str()).ok_or_else(|| format!("{path}: no \"dna\" string"))?.to_string());
            }
            "--random" => a.random = val("--random")?.parse().map_err(|_| "--random must be a count")?,
            "--seed" => a.seed = val("--seed")?.parse().map_err(|_| "--seed must be a number")?,
            "--intents" => a.intents = val("--intents")?,
            "--out-dir" | "--out" => a.out_dir = Some(PathBuf::from(val("--out-dir")?)),
            "--no-field" => a.field = false,
            "--json" => a.json = true,
            "--mouth" => a.mouth = Some(val("--mouth")?.parse().map_err(|_| "--mouth must be frames per second")?),
            "--lineage" => {
                a.lineage = val("--lineage")?
                    .split(',')
                    .map(|x| x.trim().parse().map_err(|_| "--lineage takes generation numbers, e.g. 1,5,20".to_string()))
                    .collect::<Result<_, _>>()?
            }
            "--rate" => a.rate = val("--rate")?.parse().map_err(|_| "--rate must be a number")?,
            "--count" => a.count = Some(val("--count")?.parse().map_err(|_| "--count must be a number")?),
            other => return Err(format!("unknown option {other:?} (see cvsynth genome --help)")),
        }
    }
    if body_given {
        a.body = Some(b);
    }
    Ok(a)
}

fn versions(field: bool) -> Versions {
    if field { Versions::default() } else { Versions { field: None } }
}

pub fn genome(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("check") => return check(&parse(&args[1..])?),
        Some("mutations") => return mutations(&parse(&args[1..])?),
        _ => {}
    }
    let a = parse(args)?;
    let mut rng = Rng::new(a.seed);
    let mut jobs: Vec<(String, BodyInput, String)> = Vec::new();
    let fixed = a.body.clone().unwrap_or(BodyInput { mass: 12.0, ..BodyInput::default() });
    if let Some(d) = &a.dna {
        jobs.push(("genome".into(), fixed.clone(), d.trim().to_string()));
    }
    for k in 0..a.random {
        let body = a.body.clone().unwrap_or_else(|| space::sample_body(&mut rng));
        jobs.push((format!("g{:03}", k + 1), body, genome::random_genome(&mut rng)));
    }
    if jobs.is_empty() {
        return Err("give --dna or --random (see cvsynth genome --help)".into());
    }
    // Descendants: one line of single-parent generations from each genome, at Unity's default
    // mutation rates x --rate, kept at the generations asked for.
    if let Some(&last) = a.lineage.iter().max() {
        let mut more = Vec::new();
        for (name, body, dna) in &jobs {
            let mut d = dna.clone();
            for g in 1..=last {
                d = genome::mutate_generation(&d, a.rate, &mut rng);
                if a.lineage.contains(&g) {
                    more.push((format!("{name}_gen{g:02}"), body.clone(), d.clone()));
                }
            }
        }
        let mut all = Vec::new();
        for j in jobs {
            let prefix = format!("{}_gen", j.0);
            let kids: Vec<_> = more.iter().filter(|m| m.0.starts_with(&prefix)).cloned().collect();
            all.push((format!("{}_gen00", j.0), j.1, j.2));
            all.extend(kids);
        }
        jobs = all;
    }
    if let Some(dir) = &a.out_dir {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    for (name, body, dna) in &jobs {
        let (c, report) = genome::decode(body, dna, &versions(a.field))?;
        println!("== {name}: {} kg {:?} {}", fmt_mass(body.mass), body.source, body.shape);
        print_report(&report);
        if a.json {
            println!("{}", c.to_json());
        }
        let Some(dir) = &a.out_dir else { continue };
        let intents: Vec<String> = match a.intents.as_str() {
            "all" => ["speak"].iter().chain(CALL_INTENTS.iter()).map(|s| s.to_string()).collect(),
            list => list.split(',').map(|s| s.trim().to_string()).collect(),
        };
        let body_json = serde_json::to_string_pretty(body).unwrap_or_default();
        fs::write(dir.join(format!("{name}.dna.txt")), format!("{dna}\n\nbody: {body_json}\n")).map_err(|e| e.to_string())?;
        fs::write(dir.join(format!("{name}.json")), c.to_json()).map_err(|e| e.to_string())?;
        fs::write(dir.join(format!("{name}.report.json")), serde_json::to_string_pretty(&report).unwrap_or_default())
            .map_err(|e| e.to_string())?;
        for intent in intents {
            let audio = create_vocalization(&c.body, &c.voice, &intent, a.seed, 10.0, 0.4)?;
            let path = dir.join(format!("{name}_{intent}.wav"));
            write_wav(&path, &audio, Format::Pcm16).map_err(|e| format!("{}: {e}", path.display()))?;
            if let Some(fps) = a.mouth {
                write_mouth(&path, &c.body, &c.voice, &intent, a.seed, fps)?;
            }
        }
    }
    if let Some(dir) = &a.out_dir {
        println!("\nwrote {}", dir.display());
    }
    Ok(())
}

fn fmt_mass(m: f64) -> String {
    if m < 1.0 { format!("{m:.3}") } else { format!("{m:.1}") }
}

fn axes_text(m: &BTreeMap<String, f64>) -> String {
    AXES.iter().map(|n| format!("{:+.2}", m[*n])).collect::<Vec<_>>().join(" ")
}

fn print_report(r: &Report) {
    println!(
        "   {} bases | {} steps, {} deflected (max badness {:.2})",
        r.bases, r.steps, r.deflected, r.max_badness
    );
    println!("   base voice [{}]", axes_text(&r.base));
    println!("   inventory  {}", r.inventory.join("  "));
    println!("   {:7} {}", "speak", r.speak);
    for (name, c) in &r.calls {
        println!("   {name:7} {:28} offset [{}]", c.phrase, axes_text(&c.offset));
    }
}

fn distance(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> f64 {
    AXES.iter().map(|n| (a[*n] - b[*n]).powi(2)).sum::<f64>().sqrt()
}

/// The calls a mutation changed: phrase (before -> after) or timbre offset (by how much).
fn call_changes(a: &Report, b: &Report) -> Vec<String> {
    let mut out = Vec::new();
    if a.speak != b.speak {
        out.push(format!("speak: {} -> {}", a.speak, b.speak));
    }
    for (k, c) in &a.calls {
        let d = &b.calls[k];
        let off = distance(&c.offset, &d.offset);
        let mut parts = Vec::new();
        if c.phrase != d.phrase {
            parts.push(format!("{} -> {}", c.phrase, d.phrase));
        }
        if off > 1e-9 {
            parts.push(format!("offset moved {off:.2}"));
        }
        if !parts.is_empty() {
            out.push(format!("{k}: {}", parts.join(", ")));
        }
    }
    out
}

// ------------------------------------------------------------------ mutation table

fn mutations(a: &Args) -> Result<(), String> {
    let mut rng = Rng::new(a.seed);
    let n = a.count.unwrap_or(6);
    let mut md = String::from("# Mutation table\n\nBase voice positions are on the 10 axes, in this order: ");
    md.push_str(&AXES.join(", "));
    md.push_str(".\nDistance is the straight-line (Pythagorean) distance between the base voice before and after. Over random genomes the base voice spreads about 0.9 per axis.\n\n");
    for g in 0..n {
        let body = space::sample_body(&mut rng);
        let dna = genome::random_genome(&mut rng);
        let (_, ra) = genome::decode(&body, &dna, &versions(a.field))?;
        println!("== genome {} ({} kg {:?})", g + 1, fmt_mass(body.mass), body.source);
        print_report(&ra);
        md.push_str(&format!(
            "## Genome {} ({} kg {:?})\n\n`{dna}`\n\nInventory: {}  \nBase voice: {}\n\n| Kind | At | Hit | Mutated genome (changed bases in lower case) | Base voice after | Distance | Calls changed |\n|---|---|---|---|---|---|---|\n",
            g + 1, fmt_mass(body.mass), body.source, ra.inventory.join(" "), axes_text(&ra.base)
        ));
        for kind in MUTATIONS {
            let m = genome::mutate(&dna, kind, &mut rng);
            let pos = dna.bytes().zip(m.bytes()).position(|(x, y)| x != y).unwrap_or(dna.len().min(m.len()));
            let (_, rb) = genome::decode(&body, &m, &versions(a.field))?;
            let dist = distance(&ra.base, &rb.base);
            let hit = genome::region(&dna, pos);
            let mut changes = call_changes(&ra, &rb);
            if ra.inventory != rb.inventory {
                changes.insert(0, format!("inventory: {} -> {}", ra.inventory.join(" "), rb.inventory.join(" ")));
            }
            println!("   {kind:11} at {pos:4} ({hit})  base moved {dist:.2}");
            for c in &changes {
                println!("      {c}");
            }
            let mark: String = m
                .char_indices()
                .map(|(i, c)| if i >= pos && dna.as_bytes().get(i) != Some(&(c as u8)) { c.to_ascii_lowercase() } else { c })
                .collect();
            md.push_str(&format!(
                "| {kind} | {pos} | {hit} | `{mark}` | {} | {dist:.3} | {} |\n",
                axes_text(&rb.base),
                if changes.is_empty() { "none".into() } else { changes.join("<br>") }
            ));
        }
        md.push('\n');
    }
    if let Some(path) = &a.out_dir {
        fs::write(path, md).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

// ------------------------------------------------------------------ checks

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn check(a: &Args) -> Result<(), String> {
    let mut rng = Rng::new(a.seed);
    let n = a.count.unwrap_or(300).max(20);
    let samples: Vec<(BodyInput, String)> =
        (0..n).map(|_| (space::sample_body(&mut rng), genome::random_genome(&mut rng))).collect();
    let decode = |b: &BodyInput, d: &str, field: bool| genome::decode(b, d, &versions(field)).map(|x| x.1);
    let reports: Vec<Report> = samples.iter().map(|(b, d)| decode(b, d, a.field)).collect::<Result<_, _>>()?;

    println!("== 1. Coverage: {n} random genomes ({SECTION_LENGTH} bases) on random bodies");
    println!("   {:18} {:>7} {:>6} {:>8} {:>8}", "base voice axis", "mean", "sd", "median|", "90th|");
    for name in AXES {
        let vals: Vec<f64> = reports.iter().map(|r| r.base[name]).collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let sd = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let mut abs: Vec<f64> = vals.iter().map(|v| v.abs()).collect();
        println!("   {name:18} {mean:+7.2} {sd:6.2} {:8.2} {:8.2}", quantile(&mut abs, 0.5), quantile(&mut abs, 0.9));
    }
    let mut sizes = [0usize; 7];
    let mut with_cons = 0;
    let mut types = 0;
    let mut glides = 0;
    for r in &reports {
        sizes[r.inventory.len()] += 1;
        for t in &r.inventory {
            types += 1;
            with_cons += t.starts_with(|c: char| "hsftcjpkbdg".contains(c)) as usize;
            glides += t.contains('~') as usize;
        }
    }
    println!(
        "\n   inventory size 2-6: {}; {:.0} % of types start with a consonant, {:.0} % glide to a second vowel",
        (2..=6).map(|k| format!("{}%", 100 * sizes[k] / n)).collect::<Vec<_>>().join(" "),
        100.0 * with_cons as f64 / types as f64,
        100.0 * glides as f64 / types as f64
    );
    let distinct: Vec<f64> = reports
        .iter()
        .map(|r| {
            let mut ph: Vec<&String> = r.calls.values().map(|c| &c.phrase).collect();
            ph.sort();
            ph.dedup();
            ph.len() as f64
        })
        .collect();
    let runs: f64 = reports
        .iter()
        .flat_map(|r| r.calls.values())
        .map(|c| c.picks.windows(2).filter(|w| w[0] == w[1]).count() as f64 / (c.picks.len().max(2) - 1) as f64)
        .sum::<f64>()
        / (7 * n) as f64;
    println!(
        "   distinct call phrases per creature: {:.1} of 7; a syllable repeats the one before it {:.0} % of the time",
        distinct.iter().sum::<f64>() / n as f64,
        100.0 * runs
    );

    let per = 4;
    let m = n.min(150);
    println!("\n== 2. Mutation spectrum: {m} genomes x {per} mutations of each kind");
    println!("   Base voice distance moved (its spread is about 0.9 per axis, 2.8 over all ten), and what else changed\n");
    let buckets = [0.0, 0.05, 0.1, 0.2, 0.5];
    println!(
        "   {:15} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>10} {:>9} {:>9} {:>9}",
        "kind", "median", "<0.05", "<0.1", "<0.2", "<0.5", ">=0.5", "inventory", "1 phrase", "2+ phr.", "offsets"
    );
    let kinds: Vec<&str> = MUTATIONS.iter().copied().chain(["generation", "generation x0.2"]).collect();
    for kind in kinds {
        let mut dists = Vec::new();
        let (mut inv, mut one, mut many, mut off) = (0, 0, 0, 0);
        for (b, d) in samples.iter().take(m) {
            let before = decode(b, d, a.field)?;
            for _ in 0..per {
                let mutated = match kind {
                    "generation" => genome::mutate_generation(d, 1.0, &mut rng),
                    "generation x0.2" => genome::mutate_generation(d, 0.2, &mut rng),
                    k => genome::mutate(d, k, &mut rng),
                };
                let after = decode(b, &mutated, a.field)?;
                dists.push(distance(&before.base, &after.base));
                inv += (before.inventory != after.inventory) as usize;
                let phrases = before.calls.iter().filter(|(k, c)| c.phrase != after.calls[*k].phrase).count()
                    + (before.speak != after.speak) as usize;
                one += (phrases == 1) as usize;
                many += (phrases >= 2) as usize;
                off += before.calls.iter().any(|(k, c)| distance(&c.offset, &after.calls[k].offset) > 1e-9) as usize;
            }
        }
        let total = dists.len();
        let pct = |c: usize| 100.0 * c as f64 / total as f64;
        let mut hist = [0usize; 5];
        for &x in &dists {
            hist[buckets[1..].iter().filter(|&&t| x >= t).count()] += 1;
        }
        let med = quantile(&mut dists.clone(), 0.5);
        println!(
            "   {kind:15} {med:7.3} {:6.0}% {:6.0}% {:6.0}% {:6.0}% {:6.0}% | {:9.0}% {:8.0}% {:8.0}% {:8.0}%",
            pct(hist[0]), pct(hist[1]), pct(hist[2]), pct(hist[3]), pct(hist[4]), pct(inv), pct(one), pct(many), pct(off)
        );
    }

    println!("\n== 3. Steering: the same genomes with the field on and off");
    let off: Vec<Report> = samples.iter().map(|(b, d)| decode(b, d, false)).collect::<Result<_, _>>()?;
    let on: Vec<Report> = samples.iter().map(|(b, d)| decode(b, d, true)).collect::<Result<_, _>>()?;
    let bad_on = on.iter().filter(|r| r.base_badness > 0.05 || r.calls.values().any(|c| c.badness > 0.05)).count();
    let deflected = on.iter().filter(|r| r.deflected > 0).count();
    let moved: Vec<f64> = on.iter().zip(&off).map(|(x, y)| distance(&x.base, &y.base)).collect();
    let spread = |rs: &[Report]| {
        AXES.iter()
            .map(|a| {
                let v: Vec<f64> = rs.iter().map(|r| r.base[*a]).collect();
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
            })
            .sum::<f64>()
            / 10.0
    };
    println!(
        "   field v{}: {} % of voices end in a disliked region with steering on; {} % had a step deflected",
        faunavox::field::builtin().version, 100 * bad_on / n, 100 * deflected / n
    );
    println!(
        "   base voice moved by steering: mean {:.3}; spread (mean sd per axis) on {:.2}, off {:.2}",
        moved.iter().sum::<f64>() / n as f64, spread(&on), spread(&off)
    );

    let same = samples.iter().take(30).all(|(b, d)| {
        let x = genome::decode(b, d, &Versions::default()).unwrap().0.to_json();
        let y = genome::decode(b, d, &Versions::default()).unwrap().0.to_json();
        x == y
    });
    println!("\n== 4. Determinism: {}", if same { "same DNA and body give the same creature, byte for byte" } else { "FAILED" });
    Ok(())
}
