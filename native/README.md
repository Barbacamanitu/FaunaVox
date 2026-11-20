# FaunaVox native engine (Rust)

A native port of [`../create_vocalization.py`](../create_vocalization.py). It uses the same model,
parameters, intents, presets and JSON format. The Python file stays the reference: read
[`../README.md`](../README.md) for how the model works and how to tune a creature.

This one crate builds two things:

| Output | What it is |
|---|---|
| `cvsynth.exe` | A command-line renderer. It reads creatures from JSON or presets and writes WAV files. |
| `faunavox.dll` | A C ABI library for Unity (P/Invoke) or any native host, such as a Wwise or FMOD plugin. |

## Build

You need Rust 1.85 or later (`rustup`).

```powershell
./build.ps1        # cargo build --release, then copies the dll, exe, header and C# binding to ./dist
cargo test --release
```

## Command line

```powershell
cvsynth --preset wolf --intent mating --out wolf.wav
cvsynth --json examples/marsh_wolf.json --intent all --variants 4 --reverb 0 --out-dir cache
cvsynth --preset cat --set mass=12 --vowels i,a,u --intent mating
cvsynth --all-presets --demo                    # every intent of every preset into ./vocalizations
cvsynth genome --mass 30 --random 1 --lineage 1,5,20 --rate 0.1 --out-dir out
cvsynth genome --random 8 --out-dir out         # random genomes on random bodies
cvsynth genome check                            # evolvability checks
cvsynth --help
```

- `--json` takes one creature (`{"body": ..., "voice": ...}`, optionally with `"name"`), or an
  object that maps names to creatures, so a whole bestiary can live in one file.
- `--variants N` renders seeds `SEED .. SEED+N-1`, which gives each creature a set of takes per
  intent. Batches run on all cores.
- `--set FIELD=VALUE` overrides any Body or Voice field. VALUE is JSON or a bare word:
  `--set source=syrinx`, `--set vowels='["u","o"]'`.
- `--format pcm16|float32`, `--distance M`, `--reverb X` (0 = dry).
- `--print-json` prints the resolved creature, with defaults filled in.

### Example creatures (`examples/`)

| File | Shows |
|---|---|
| `minimal.json` | The smallest useful file: everything else uses its default. |
| `marsh_wolf.json` | A howler: `head_length`/`neck_length`, falsetto, sustain, and `calls` overrides. |
| `cave_bear.json` | A big, harsh mammal, with a custom `warn` and a `territorial` built from it with `"like"`. |
| `reed_warbler.json` | A bird (`"source": "syrinx"`) with a custom vowel given as F1–F3 ratios. |
| `bog_toad.json` | A frog-like pulse source with a vocal-sac resonance and its octave. |
| `bestiary.json` | Several creatures in one file (a map of name → creature). |
| `speaker.json` | A phrase for the `speak` intent with `~` glides and a `\|` pause, and a sequential `warn`. |
| `greeting.json` | "hey heath" for `speak`: `h` and `th` consonants, a glided diphthong and a pause. |

```powershell
cvsynth --json examples/cave_bear.json --intent warn --out bear_warn.wav
cvsynth --json examples/bestiary.json --intent idle,warn --variants 3 --out-dir bestiary
cvsynth --json examples/minimal.json --set mass=300 --intent all --out-dir heavy
```

### Tone and resonance settings

These `body` fields shape how breathy or resonant a voice is. The main README explains them.

| Field | Default | Effect |
|---|---|---|
| `open_quotient` | 0.5 | Lower is pressed and brighter (0.4 for roars and grunts); higher is softer and purer (0.65). |
| `formant_gain` | 8 | dB of resonance at each formant. 0 turns it off; 12 rings more. |
| `formant_width` | 1 | Below 1 gives sharper, ringier formants. |
| `breath_pulse` | 0.9 | How much breath noise pulses with each glottal cycle. Lower is airier. |

```powershell
cvsynth --preset lion --set open_quotient=0.4 --set formant_gain=12 --intent territorial --out lion.wav
```

### Nonlinear phenomena (`voice` fields, 0–1)

`subharmonics`, `chaos`, `biphonation` (plus `biphonation_ratio`, default 1.45) and `jumps`.
Each is the voice's heritable proneness. Intents add their own amounts on top; per-intent
overrides go under `calls`.

```powershell
cvsynth --preset wolf --set biphonation=0.6 --set subharmonics=0.4 --intent mating --out wolf_biphonic.wav
cvsynth --preset wolf --set chaos=0.8 --set jumps=0.6 --intent flee --out wolf_scream.wav
```

### Vowels

Names: `a e i o u @`, plus the eleven SSBE vowels by IPA name or ASCII alias:
`i:` `iy`, `ɪ` `ih`, `ɛ` `eh`, `æ` `ae`, `ʌ` `ah`, `ɑ:` `aa`, `ɒ` `oh`, `ɔ:` `ao`, `ʊ` `uh`,
`u:` `uw`, `ɜ:` `er`.

```powershell
cvsynth --preset lion --vowels aa,ao --intent territorial --out lion_aa.wav
```

How the list is used depends on the call:

- **Most intents** (idle, warn, alarm, flee, pain) give each repeat one vowel, cycling through the
  list (`vowels[i % len]`). Vowels past the number of repeats are never heard.
- **Gliding intents** (mating, beg, or any with `"glide": true`) slide through the whole list
  inside every syllable.
- **The territorial bout** moans on the first vowel, roars through the whole list and grunts on
  the last.

### Sequential vowels: the `speak` intent

`speak` voices every vowel in the list, in order, one syllable each. Repeated vowels are not
merged: `a,a,a` is three syllables. Two separators change how the phrase is said:

| Token | Effect |
|---|---|
| `~` | Glide: the vowels on either side become one longer syllable that slides from one to the next. `i:~ɑ:~u:` is one syllable. |
| `\|` | Pause: a break after the syllable before it, about 0.25 s for a 50 kg animal. |

Separators can be list items of their own (`["a", "~", "i"]`) or written inside a string
(`"a~i | u"`). Spaces and commas inside a string also separate vowels. Any `vowels` field in JSON
(the voice's or an intent's under `calls`) can be a single string holding the whole phrase,
`"vowels": "a i~u | e"`, or a list. Custom `[F1, F2, F3]` vowels
work too. Intents without `sequence` ignore the separators.

#### Consonants

Consonants can go anywhere in a sequential phrase, as separate tokens:

| Token | Sound | How it's made |
|---|---|---|
| `h` | breath | Unvoiced breath through the tract, shaped like the vowel next to it, with its F1 damped by the open glottis. About −20 dB. |
| `s` | hiss | A peak near 4.5 kHz for a human-size mouth and a weaker one near 6.5 kHz, cut at 24 dB/octave below 3 kHz and rolled off above 9 kHz. About −27 dB. |
| `sh` (`ʃ`) | hush | A peak near 2.7 kHz, with a weaker one near 4.8 kHz, cut below 2 kHz and rolled off above 6.5 kHz. About −22 dB. |
| `f` | soft hiss | Faint noise from about 1 to 8 kHz. About −29 dB. |
| `th` (`θ`) | soft hiss | Like `f`, slightly brighter. About −29 dB. |
| `ch` (`tʃ`) | affricate | A short sh burst that starts quickly and dies away. About 55 % of sh's length, about −24 dB. |
| `j` (`dʒ`) | affricate | Like `ch`, shorter and quieter, about −27 dB. |
| `p` `b` | stop (lips) | A closure, then a diffuse burst that falls off above 3 kHz. |
| `t` `d` | stop (tongue tip) | A closure, then a high burst peaking near 4.5 kHz. |
| `k` `g` | stop (back of tongue) | A closure, then a compact burst near the next vowel's F2: higher before front vowels ("key"), lower before back ones ("caw"). |

Levels are RMS against the loudest syllable, kept well under measured speech: noise stands out
from a creature's voice far more than from speech, and a faint consonant is better than a loud
one. The order and most of the shaping below follow Lorin Wilde's thesis, *Analysis and Synthesis
of Fricative Consonants* (MIT, 1995):

- **Order:** sh is 5–10 dB above s, and f and th are about equal, far below both. Listeners reject
  an s even 4 dB too loud, but accept f anywhere within 4 dB.
- **Next to open back vowels:** s, sh, ch and j are up to 7 dB weaker beside an open back vowel
  ("sa" against "se"; `STRIDENT_OPEN_BACK`).
- **Darker when quieter:** turbulence falls off above a corner frequency that rises with airflow,
  so the quieter a fricative is set, the darker it is (`SOURCE_CORNER`).
- **Steady:** the noise holds a constant spectrum and level between short ramps of about 25 ms
  (`FRICATIVE_RAMP`). Extra energy at the edges sounded unnatural to listeners. An affricate bursts
  out in about 12 ms and dies away. The turbulence flutters slightly.
- **Release into a vowel:** the noise stops, there is a near-silent dip of about 12 ms
  (`RELEASE_DIP`), then the voice switches on abruptly. A weak breath through the vowel's formants
  (`ASPIRATION`, 12 dB under the fricative, F1 damped) runs from the end of the fricative into the
  vowel; the glottis closes slowly after a fricative. Before a fricative it is shorter.
- **Formant transitions:** the vowel's F1, F2 and F3 start near the fricative's locus and move to
  their own targets over about 60 ms, and move back toward it before a fricative that follows.
  This makes the consonant sound like part of the same voice. F1 dips toward 300 Hz at every
  fricative, the plainest cue that the tract has closed in. F2 orders the places: f lowest, then
  th, s, and sh highest. The loci (`locus`) come from Klattara's DECtalk-style tables, which are
  wider than natural locus equations because synthetic speech needs its place cues exaggerated. At
  the boundary the vowel pulls each formant only halfway toward itself (`LOCUS_PULL`), so the
  formants bend more sharply into the consonant. Values are ratios of the neutral tube's
  500/1500/2500 Hz, so they scale with body size.
- **Rounding:** a rounded vowel beside s or sh lowers its hiss by up to 15 %, so "sue" hisses
  lower than "see".
- **Stops** follow Klatt's synthesizer (1979, 1980) and Klattara's tables:
  - *Closure:* 40–65 ms of silence (`consonant_shape`). Voiced stops are shorter.
  - *Burst:* 5–16 ms of noise that starts within half a millisecond and dies away
    (`stop_timing`, `render_burst`). k's is the longest and strongest. Levels are −22 to −27 dB for
    p, t and k, and about 4 dB lower for b, d and g.
  - *Breath (p, t, k):* 40–55 ms of breath through the next vowel's formants, at −24 dB, before
    the voice starts (`STOP_ASPIRATION`). That delay is what separates "tie" from "die".
  - *Voice bar (b, d, g):* between vowels, the voice keeps going through the closure at −22 dB.
    Only its lowest harmonics get through, lowpassed near 450 Hz for a human-size tract
    (`VOICE_BAR`). At the start of a phrase the closure is silent.
  - *After a vowel* (before a `|` or at the end), a stop is released more weakly: 6 dB quieter,
    with 40 % of the breath (`CODA_RELEASE`).
  - *Two stops in a row* ("big gate"): the first isn't released, and its closure runs into the
    second's.
  - *Front k and g:* before a front vowel (i, e, ae), k and g close further forward, and their F2
    locus rises to about 2250 Hz.
- **Into the next syllable:** when the voice runs straight into a consonant that starts the next
  syllable, the vowel's formants already bend toward that consonant's place, as well as away from
  the one before.
- **Timing:** consonants before a vowel lead into it (an h overlaps its start). Consonants before a
  `|` pause or at the end of the phrase trail the vowel before them. In `"a s a"` the s starts the
  second syllable.
- **Size:** smaller mouths hiss higher; frequencies scale with √(0.17 m / tract length).
  Consonant length grows with the square root of body size. A tiny creature's s goes up to the top
  of the audible range.
- **Tuning:** levels and lengths are in `consonant_shape` and `stop_timing` in `synth.rs`, and
  spectra in `render_fricative` and `render_burst`.
- **Spacing:** a consonant must be its own token, separated by spaces, so write `"h eh~ih"`, not
  `"heh~ih"`.
- **Other intents** ignore consonants, like the separators.

```powershell
cvsynth --preset wolf --intent speak --vowels "a,a,i,u" --out wolf_aaiu.wav
cvsynth --preset wolf --intent speak --vowels "i:~ɑ:~u:,a,a|e,i,i" --out wolf_phrase.wav
cvsynth --json examples/speaker.json --intent speak,warn --out-dir speaker
cvsynth --json examples/greeting.json --intent speak --out hey_heath.wav
```

How it is said:

- Each syllable has a small rise and fall in pitch. The pitch sinks 0.3 semitones per syllable
  over the phrase (`drift`), and the level drops by up to 20 % by the end.
- A glided syllable lasts 0.7 × `dur` longer for each extra vowel, scaled by the voice's `sustain`.
- Rounded vowels (u, o) radiate far less through the lips. Syllables are evened out 70 % towards
  equal loudness, so a `u` sits about 3 dB under an `a` rather than 10–12 dB.
- Syllables are `gap` apart, except that a syllable starting with a consonant follows straight on
  from the one before; the consonant fills the gap.
- `repeats` is how many times the whole phrase is said, separated by a pause. `voice.repetition`
  does not apply.
- Every other intent field works as usual (`dur`, `gap`, `accel`, `rhythm`, `contour`, `mouth`, `inhale`,
  NLP…), and the intent can have its own `"vowels"` phrase under `calls.speak`.
- `"prosody"` gives each syllable, in order, `[length ×, pitch step (semitones), pitch slope across
  the syllable (semitones), level ×]`, e.g. `[[1, 0, 0, 1], [1.8, 3, -6, 0.8]]` for a plain syllable
  then a long, higher, falling, softer one. Syllables past the list are plain. Native only: the
  Python version doesn't read it yet.

Any intent can be made sequential with `"sequence": true`. For example, a warn call that barks
through its own phrase:

```json
"calls": { "warn": { "sequence": true, "vowels": ["a", "a", "ɑ:~ʌ"], "dur": 0.35, "repeats": [1, 1] } }
```

Shorten `dur` when you do this. The template's `dur` becomes the length of every syllable, and its
`repeats` the number of times the whole phrase is said. A warn growl is 1.5 s, so a three-vowel
phrase said twice would run over 12 s.

### English text to a phrase

`--say` translates English into a `speak` phrase and renders it. `--translate` only prints the
phrase.

```powershell
cvsynth --preset wolf --say "Hey, Heath!" --out hey.wav    # say: h eh~ih | h iy th
cvsynth --translate "I see you. Leave my cave!"          # a~ih s iy iy~uw | iy f a~ih eh~ih f
```

`--say` sets `calls.speak.vowels` and renders `speak` unless `--intent` is given.

**Where pronunciations come from:**
- Words are looked up in the [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
  (134k words). The copy in `data/cmudict.dict` is embedded in the exe and the DLL; its licence is
  in `data/cmudict.LICENSE`.
- Words it doesn't know, such as made-up creature names, go through the NRL letter-to-sound rules
  in `data/english_rules.txt`.
- Digits are read one by one.

**How the phonemes become synth tokens:**

| English sounds | Become |
|---|---|
| Vowels | The SSBE vowel names. Unstressed "uh" (AH0) becomes schwa `@`. |
| Diphthongs | Glides: day `eh~ih`, my `a~ih`, now `a~uh`, boy `ao~ih`, go `o~uh`. |
| w, y | A glide into the next vowel: we `uw~iy`, you `iy~uw`. |
| h s sh f th | Themselves. |
| z v dh zh | The nearest voiceless consonant: `s` `f` `th` `sh`. |
| ch, j | The affricates `ch` and `j`. |
| p b t d k g | The stops themselves. |
| m n ng l r | Dropped, because the synth can't make them. |
| `, ; : . ! ?` and dashes | A pause, `\|`. |

Since nasals, l and r are dropped and voiced fricatives become voiceless ones, the result keeps
the rhythm, vowels, hisses and stops of the sentence, but it isn't fully intelligible speech. "The
quick brown fox" becomes `th @ k uw~ih k b a~uh f aa k s`.

The same translation is available in other places:
- In the DLL as `cv_english_to_phrase`. The dictionary loads on the first call, in about 40 ms.
- In Unity as `NativeCreatureVoice.EnglishToPhrase`.
- In Python as `english_phrase.py`, or `create_vocalization.py --say`, which read the same data
  files.

To update the dictionary, pull `third_party/cmudict` (a clone of the repository, git-ignored),
copy its `cmudict.dict` into `data/` and rebuild.

### JSON additions

- `body.head_length` and `body.neck_length`: when `tract_length` is not given, the tract length is
  0.8 × head + 0.5 × neck (Python's `Body.from_dimensions`).
- Unknown fields are rejected, with a list of the valid names.

## Unity

1. Copy `faunavox.dll` to `Assets/Plugins/x86_64/`.
2. Copy `unity/CreatureVoiceNative.cs` anywhere under `Assets/`.

```csharp
using var voice = NativeCreatureVoice.FromJson(json);      // or FromPreset("wolf")
voice.Set("mass", body.Mass).Set("tract_length", tract);    // genome / body driven
voice.SetJson("vowels", "[\"u\",\"o\"]");
float[] samples = await voice.RenderAsync("mating", seed);  // worker thread
AudioClip clip = NativeCreatureVoice.ToClip("mating", samples);  // main thread
```

Notes on the binding:

- `Render` is thread-safe for a given creature, so render a creature's whole set of calls in
  parallel when it spawns. Don't call `Set` while that creature is rendering.
- For Wwise or FMOD, pass `reverb: 0` and let the middleware add reverb. Feed the samples in
  through the Audio Input plugin or programmer sounds, or through a small playback plugin that
  reads them.
- Unity reloads the DLL only when the editor restarts. To replace the DLL, close Unity first.

The C API is in [`src/faunavox.h`](src/faunavox.h):

| Function | Purpose |
|---|---|
| `cv_creature_from_json` / `_from_preset` / `_clone` / `_free` | Create, copy and free creature handles |
| `cv_creature_from_genome` | Decode a creature from its body and the voice section of its genome ([GENOME_VOICE_DESIGN.md](../GENOME_VOICE_DESIGN.md)) |
| `cv_genome_report` | What the decoder did for that creature, as JSON |
| `cv_decoder_version` / `cv_field_version` | The decoder and preference field built in. Store them with a save. |
| `cv_creature_set` / `cv_creature_set_json` | Set a field by name |
| `cv_render` | Render one call. Returns a buffer handle. |
| `cv_buffer_len` / `_data` / `_copy` / `_free` | Read and free a rendered buffer |
| `cv_last_error` | Why the last call failed, per thread |

Every entry point catches panics, so a bad input returns an error instead of crashing the editor.

## Differences from the Python version

- **Random stream:** this version uses its own seeded RNG (xoshiro256++), not numpy's. A given seed
  is deterministic on every platform, but it gives a different take than the same seed in Python.
  Across many seeds, the call plans are identical: the same call-count distributions and lengths.
- **Dry clips:** with `reverb` 0, the 0.4 s of trailing silence that Python pads on is not added.
- **Gravel bandpass:** the 20–50 Hz bandpass for `rumble` is a 2nd-order highpass and lowpass in
  series, not scipy's bandpass design. The noise is renormalized afterwards, so this barely matters.
- **Negative periods:** a chaotic period draw is clamped to at least one sample. Python could step
  time backwards.

Checked against Python: all 14 presets × 8 intents, 4 seeds each, with no reverb. The median
spectral centroid ratio is 1.003, and the median 1/3-octave spectrum difference is 0.9 dB. That is
less than Python's own seed-to-seed spread of 1.1 dB.

Speed: about 25 ms of CPU per second of audio on one core, roughly twice as fast as the
numba-compiled Python. A 16 s wolf howl takes about 0.4 s.

## Layout

```
src/model.rs      Body, Voice, intents, JSON (serde)
src/synth.rs      sources, vocal tube, radiation, mix (the model)
src/dsp.rs        Butterworth/notch/resonator biquads, interpolation, FFT convolution
src/rng.rs        deterministic RNG
src/presets.json  built-in presets (same as PRESETS in the Python)
src/english.rs    English text -> speak phrase (CMU dictionary + letter-to-sound rules)
src/space.rs      voice space: 10 perceptual axes relative to the body, mapped to synth settings
src/genome.rs     genome decoder: 4-base windows -> base voice walk, syllable inventory, call patterns
src/field.rs      the preference field (fitted by ratings/fit.py), for steering the walk
data/field_v1.json  the fitted field (copied from ratings/; embedded)
data/cmudict.dict CMU Pronouncing Dictionary (embedded; licence in data/cmudict.LICENSE)
data/english_rules.txt  letter-to-sound rules for words not in the dictionary
src/ffi.rs        C ABI
src/bin/cvsynth.rs  command-line tool (cvsynth/space_cli.rs: sample, sweep; cvsynth/genome_cli.rs: genome)
unity/CreatureVoiceNative.cs  Unity binding
```
