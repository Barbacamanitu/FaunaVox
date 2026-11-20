# FaunaVox

**Procedural creature vocalizations.** Animal voices made from scratch, from a physical model of how voices are produced. Nothing is sampled. A
creature's body sets its size and sound source, its genome sets its voice and its calls, and the game's AI picks
what it's saying: an idle contact call, a warning growl, an alarm, a mating song.

- **Physical model:** larynx, syrinx, pulse and hiss sources through a lossy vocal tube, with formants, vowels,
  consonants (fricatives and stops), nonlinear phenomena (subharmonics, chaos, biphonation, register breaks), lip
  radiation and an outdoor mix.
- **Genomes:** a plain string of DNA bases decodes into a whole creature voice: its timbre, a small inventory of
  syllables, and a phrase, rhythm and timbre for each of its calls. Mutations make small, mostly local changes, so
  lineages drift audibly but gradually.
- **Steering by taste:** a preference model fitted to your own ratings of sample voices steers decoded voices
  around the kinds you dislike, without forbidding them.
- **Native and fast:** a Rust library and DLL (`faunavox.dll`) with a C API, a C# binding for Unity, and a
  command-line tool, `cvsynth`. A call renders in tens of milliseconds, and a genome decodes in about 20 ms.

| Part | What it is |
|---|---|
| [`native/`](native/README.md) | The Rust engine, DLL, C API, Unity binding and `cvsynth`. This is the main implementation. |
| [`GENOME_VOICE_DESIGN.md`](GENOME_VOICE_DESIGN.md) | How genomes decode into voices, the voice space, the preference model, and evolvability measurements. |
| [`ratings/`](ratings/README.md) | The rating page and the preference-model fit. |
| `create_vocalization.py` | The original Python engine. The Rust version is ported from it and has more features: it decodes genomes and reads per-syllable `prosody`; Python does neither. |
| `english_phrase.py` | English text to a phrase of vowels and consonants, for the `speak` intent. |
| `lion_roar.py` | The first single-purpose prototype, kept for reference. |

## Quick start

Build the DLL and the command-line tool with a recent stable Rust:

```powershell
native\build.ps1          # faunavox.dll, cvsynth.exe, the C header and the Unity binding, in native\dist
```

Render one of the example creatures from its genome: every call it has, as WAVs, with its decoded spec and a
report of what the decoder did.

```powershell
native\target\release\cvsynth.exe genome --genome native\examples\genomes\dusk_hound.json --out-dir out
```

Or render a hand-made creature, or a preset:

```powershell
native\target\release\cvsynth.exe --json native\examples\marsh_wolf.json --intent all --out-dir out
native\target\release\cvsynth.exe --preset wolf --intent mating --out wolf.wav
```

## Example genomes

Each file in [`native/examples/genomes/`](native/examples/genomes) is a body and a 1,600-base voice section:

```json
{"name": "dusk_hound",
 "body": {"mass": 30.0, "source": "larynx", "shape": "quadruped", "lips": true, "teeth": true},
 "dna": "CATTGTCTTTGGCTTGCA…"}
```

What they decode to (vowels are shown by their nearest named vowel, `~` glides from one vowel into the next, and
`|` is a pause):

| Creature | Body | Syllable inventory | Mating call | Alarm | Warn |
|---|---|---|---|---|---|
| `dusk_hound` | 30 kg mammal | jih, er, to | jih jih to | jih \| jih | jih jih to |
| `burrow_fox` | 4 kg mammal | e, h@, hih, taa, fao | fao \| taa taa e | h@ \| e | hih hih |
| `reed_finch` | 38 g bird (no lips) | hae~iy, hae, e, eh, uw | hae~e | uw | hae~iy e |
| `sand_adder` | 3.4 kg hissing serpent | er, h@, ih | er h@ ih | ih ih er | h@ h@ er |
| `dew_cricket` | 9 g insect (pulse source) | huw, iy | huw iy iy | iy iy~uw | huw huw |

Each syllable type also has its own length, pitch step, pitch slope and loudness, and each call has its own small
shift in timbre from the creature's base voice. `dusk_hound` is the voice heard in the lineage tests in the design
document: its descendants keep `jih` for generations while the other syllables drift.

**Make your own:** `cvsynth genome --random 8 --out-dir out` decodes 8 random genomes on random bodies.
`--lineage 1,5,20 --rate 0.1` adds descendants mutated as a game would mutate them, so you can hear how a voice
drifts. Any string of A, C, G and T decodes; 1,600 bases is the length the scales are tuned for.

## In Unity

Copy `native/dist/faunavox.dll` to `Assets/Plugins/x86_64/` and `native/dist/CreatureVoiceNative.cs` anywhere
under `Assets/`. Unity loads a native DLL once per editor session, so close the editor before replacing it.

```csharp
// The body from the creature you've built; the voice section from its genome.
string body = "{\"mass\": 30, \"head_length\": 0.16, \"neck_length\": 0.12, \"source\": \"larynx\", "
            + "\"shape\": \"quadruped\", \"lips\": true, \"teeth\": true}";
using var voice = NativeCreatureVoice.FromGenome(body, voiceDna);

// Render on a worker thread, make the clip on the main thread.
float[] samples = await voice.RenderAsync("mating", seed);
AudioClip clip = NativeCreatureVoice.ToClip("mating", samples);

// Store these with the save: the same DNA, body and versions always decode to the same voice.
int decoder = NativeCreatureVoice.DecoderVersion, field = NativeCreatureVoice.FieldVersion;
```

The intents are `idle`, `warn`, `alarm`, `flee`, `mating`, `pain`, `beg`, `territorial` and `speak`. The same
seed always gives the same take; vary it (creature id plus a call counter) for natural variation between calls.
`voice.ToJson()` gives the decoded creature, and `voice.GenomeReport()` shows what the decoder did.

**Mutation rates for the voice section.** Every base of the section is read, so per-base mutation rates tuned for
body genes are far too fast here: at 0.006 point changes per base, nearly every child sounds different from its
parent. About 0.1–0.2 of those rates gives gradual drift. Insertions, deletions and duplications shift everything
after them, so a voice section is best mutated with point changes, inversions and reversals only, at a fixed
length. See [`GENOME_VOICE_DESIGN.md`](GENOME_VOICE_DESIGN.md) §6 for the measurements.

## Without genomes

A creature can also be written by hand: a body, a voice, and per-call overrides, as JSON or in code. The rest of
this document describes that model. The Rust version reads the same JSON as the Python one.

```python
from create_vocalization import Body, Voice, create_vocalization

body = Body.from_dimensions(mass=40, head_length=0.25, neck_length=0.12, roughness=0.25)
voice = Voice(vowels=["u", "o"], pitch=8, falsetto=0.6, sustain=3.0, melody=0.1)
audio = create_vocalization(body, voice, intent="mating", seed=creature_id, reverb=0)
# mono float32, 44.1 kHz, peak-normalized to 0.9
```

| Input | What it is | Where the game gets it |
|---|---|---|
| `Body` | physical traits | the creature's actual size and body plan, plus a few species-level tissue traits |
| `Voice` | heritable vocal traits | decoded from the genome, or written by hand |
| `intent` | what the call means | the AI's current behaviour |

## License

FaunaVox is by Russ Martelli and released under the [MIT License](LICENSE). It includes the CMU Pronouncing
Dictionary (BSD-style license) and is built with MIT-licensed Rust crates; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which must travel with any copy, including binary builds.

---

## How the engine works

Every voiced sound goes through the same **source → filter → radiation** chain that real
vertebrate voices use:

```
intent template ─┐
voice overrides ─┼─> calls (timing, pitch contour, loudness, harshness, mouth, vowels)
body + voice ────┘        │
                          v
   SOURCE   larynx pulses | syrinx | pulse train | hiss        (+ breath turbulence)
                          │
   FILTER   lossy vocal tube (size → formants) → vowel shaping → lip aperture
                          │
   RADIATE  lip radiation, nasal coupling, chest resonance, low cut
                          │
   MIX      calls + inhales → saturation → air absorption → short outdoor reverb → normalize
```

### 1. Intent → calls

An intent is a template (`INTENTS` in the code). It sets a pitch offset, a pitch contour, a
loudness envelope, harshness, length, how many repeats with what gaps, and mouth movement. The
templates follow Morton's motivation-structural rules: hostile calls are low and harsh, fearful
or appeasing calls are high and tonal.

| Intent | Default character |
|---|---|
| `idle` | soft contact call |
| `warn` | low, harsh growl; lips pulled back in a snarl, larynx lowered |
| `alarm` | short, sharp, high barks in a series |
| `flee` | very high, chaotic distress scream |
| `mating` | long, melodic advertisement call that glides through the voice's vowels |
| `pain` | yelp: instant onset, steep pitch fall |
| `beg` | submissive whine: high, rising, nasal |
| `territorial` | full roar bout: moan, roar, then accelerating grunts (lion-style) |
| `speak` | every vowel in the list in turn, one syllable each (see "Sequential vowels" below) |

A species can reshape any intent with `Voice.calls` (see "Species repertoire" below). The model
stays the same; only the parameters change.

#### Sequential vowels

Most intents give each repeat a single vowel, cycling through the list, so vowels past the
number of repeats are never heard. Gliding intents slide through the whole list inside every
syllable. `speak`, and any intent with `"sequence": true`, voices every vowel in the list in
order, one syllable each. Repeated vowels stay separate syllables: `a, a, a` is three.

Two separators shape the phrase. They can be list items of their own, or written inside a string
such as `"a i~u | e"`, where spaces and commas also separate vowels. A `vowels` field can also be
one string holding the whole phrase (`"vowels": "a i~u | e"`) instead of a list:

- `~` glides: the vowels on either side become one longer syllable that slides between them.
  `i:~ɑ:~u:` is one syllable.
- `|` pauses after the syllable before it.

Sequential phrases can also hold consonants, as separate tokens:
- `h` is breath through the tract, shaped like the vowel next to it.
- `s`, `sh` (`ʃ`), `f` and `th` (`θ`) are noise made at the teeth or lips (`render_fricative`).
  Their colour scales with mouth size.
- `ch` (`tʃ`) and `j` (`dʒ`) are affricates: a short sh burst that starts abruptly and dies
  away.
- `p`, `t`, `k`, `b`, `d` and `g` are stops: a short silence, then a burst of a few ms whose
  colour marks the place (`render_burst`). After p, t and k, breath runs through the next vowel
  before the voice starts. b, d and g hum through the silence between vowels.

A consonant leads into the vowel after it. Consonants before a `|` or at the end of the phrase
trail the vowel before them instead. For example, "hey heath" is `"h eh~ih | h iy th"`. Each
consonant has a fixed length and a level against the loudest syllable (`CONSONANT_SHAPE`).
Levels, spectra and the release into the vowel follow Lorin Wilde's thesis on fricatives (MIT,
1995). Stops and the vowel's formant transitions (F1 to F3 bending toward each consonant's
place) follow Klatt's synthesizer and the Klattara synthesizer's tables; see `native/README.md`.

Other intents ignore the separators and consonants.

English text can be turned into such a phrase with `english_phrase.py`, or with
`create_vocalization.py --say "Hey, Heath!"`, which renders it as `speak`. Words come from the CMU
Pronouncing Dictionary (`native/data/cmudict.dict`), and unknown words go through letter-to-sound
rules (`native/data/english_rules.txt`). Sounds the synth can't make (nasals, l, r) are dropped.
The native README describes the mapping.

Each syllable rises and falls a little in pitch. The pitch sinks over the phrase (`drift`
-0.3 semitones per syllable), and the level drops by up to 20 %. A glided syllable is 0.7 × `dur`
longer for each extra vowel. Syllable loudness is evened out 70 % (`SEQUENCE_EVEN`), so rounded
vowels, which radiate much less, aren't lost. `repeats` is how many times the whole phrase is said.

Durations are scaled by body size (`call_scale`, bigger lungs give longer calls) and by
`Voice.tempo`. The harshness of each call rises and falls with its loudness, so every voice is
roughest at the peak of a call.

### 2. Sources

| `Body.source` | Used for | How it's made |
|---|---|---|
| `larynx` | mammals | One glottal pulse per vocal cycle (Rosenberg pulse). Each cycle gets jitter, shimmer, slow pitch drift and tremor. Harshness adds period doubling (a subharmonic an octave down), chaotic jitter and a pressed, brighter pulse. There are two automatic registers. **Fry:** well below the natural pitch, pulses get short and irregular. **Falsetto:** well above it, pulses become nearly sinusoidal; this only applies if `Voice.falsetto` > 0. |
| `syrinx` | birds | Two independent tonal voices for song. `buzz`, or high harshness, switches to the pulsed larynx-style source, because syrinx labia vibrate like vocal folds. That's what caws and quacks are. |
| `pulse` | frogs, insects | A train of short bursts ringing at `resonator_freq` (a vocal sac or body resonance), with an optional second band at its octave. |
| `hiss` | reptiles | Turbulent noise only. |

**Breath noise** is mixed at a set ratio *to the voice*, not at an absolute level. The ratio grows
with breathiness and strongly (cubically) with harshness, so clean calls stay clean and full
growls come out roughly as much noise as tone. If a call's total breath (`Body.breathiness` +
the intent's `breath`) is 0 or less, no breath noise is added at all.

The noise is **pulsed with every glottal cycle**: it is strongest while the folds are open and
drops to `1 − breath_pulse` (default 0.1) between pulses. Pulsed noise fuses into the voice as
roughness. A steady floor is heard as a separate hiss on top of the voice, which reads as breathy.

**Open quotient** (`Body.open_quotient`, default 0.5) is the fraction of each cycle the folds are
open. A high value gives smooth, sine-like pulses with a dominant fundamental, which sounds soft
and breathy. A low value gives sharp pulses with strong upper harmonics, which sounds pressed and
bright. Real roars and grunts measured lower (about 0.4) than the old fixed value of 0.65.

### Nonlinear phenomena

Real animal voices don't wobble at random. They hold a steady vibration regime and switch
abruptly into another one for a stretch. The research literature names four such regimes, and
each one has its own control:

| Phenomenon | What happens | Sounds like |
|---|---|---|
| `subharmonics` | Every 2nd (sometimes 3rd) cycle is weaker and longer, which adds spectral lines at f0/2 or f0/3 | Lower, rougher and bigger, without extra breath |
| `chaos` | Deterministic chaos: a logistic map just past the onset of chaos drives period and strength from cycle to cycle | Irregular but still voiced: screams and snarls |
| `biphonation` | A second, independent tone at `biphonation_ratio` × f0 that wanders and never locks to f0. It also modulates the main voice, which adds sidebands at f0 ± g | Two voices at once (canid howls and whines, deer and elk calls) |
| `jumps` | Register breaks: the pitch snaps up (or down) several semitones and back | Yelps, meow breaks, voice cracks |

- **Episodes:** subharmonics and chaos come in episodes. Onsets are most likely at loud moments,
  as with high air pressure in a real larynx. Episodes last about 0.15–0.6 s and switch abruptly; switching more often than about twice a second
  stutters and sounds robotic.
  Biphonation comes in longer episodes.
- **Where the amounts come from:** each amount is the voice's trait (`Voice.subharmonics` etc.,
  heritable) plus the intent's own amount. Aroused intents bring them out: `warn` adds
  subharmonics and chaos, `flee` and `pain` add chaos and jumps, and the territorial roar adds
  subharmonics.
- **Harshness:** it still adds its own period doubling and cycle jitter, as before. The extra
  turbulence noise on harsh calls was halved, because the nonlinear phenomena now carry most of
  the roughness.

### 3. Vocal tract filter

- **The tube:** the tract is a per-sample lossy tube, a delay line with feedback. Its resonances
  (formants) fall at every odd quarter wavelength of `tract_length`, all the way up. Tract length
  is the strongest *size* cue: a longer tract gives lower, closer formants and sounds bigger.
- **Vowels:** F1–F3 are moved from the neutral tube toward each vowel's targets. Vowels are
  stored as ratios to a neutral tube (500/1500/2500 Hz), so the same vowel scales correctly to any
  body size. The built-in vowels are:
  - `a e i o u @`: adult human averages.
  - The **eleven SSBE monophthongs** (Standard Southern British English): mean F1/F2 of three L1
    speakers. F3 isn't in that source, so it is estimated from typical SSBE values. Use either the
    IPA name or the ASCII alias:

  | IPA | ASCII | F1 / F2 (Hz) | | IPA | ASCII | F1 / F2 (Hz) |
  |---|---|---|---|---|---|---|
  | `i:` | `iy` | 296 / 2241 | | `ɒ` | `oh` | 643 / 1019 |
  | `ɪ` | `ih` | 396 / 1839 | | `ɔ:` | `ao` | 480 / 857 |
  | `ɛ` | `eh` | 532 / 1656 | | `ʊ` | `uh` | 395 / 1408 |
  | `æ` | `ae` | 667 / 1565 | | `u:` | `uw` | 386 / 1587 |
  | `ʌ` | `ah` | 661 / 1296 | | `ɜ:` | `er` | 519 / 1408 |
  | `ɑ:` | `aa` | 680 / 1193 | | | | |
- **Mouth:** opening the jaw raises F1 and F2.
- **Larynx lowering:** during loud calls the larynx pulls down, which lengthens the tube.
- **Drift:** the whole tract drifts slightly, the way a real jaw and tongue never hold still.
- **Source-tract interaction:** when the pitch sits near F1, F1 is damped. Without this, a high
  voice on a sharp F1 rings like a flute.
- **Lip aperture:** a small or rounded opening ("u", "o", a nearly closed mouth) radiates high
  frequencies poorly, so it acts as a lowpass. `spread` (a snarl) cancels rounding.
- **Formant resonance:** peaking filters follow F1–F5 and add `formant_gain` dB at each one
  (default 8). They are tapered for F4–F5, and their width is set by `formant_width`. In real calls
  the formant peaks stand out about twice as much as the bare tube makes them, and a flat spectral
  envelope with any noise in it sounds breathy and characterless. F1 keeps the source-tract damping
  above.

### 4. Radiation and mix

- **Radiation:** lip radiation adds +6 dB per octave.
- **Nasal coupling:** an antiresonance plus a softer top end. It applies when the mouth is
  nearly closed, or according to `nasality`.
- **Chest resonance** (larynx only): low weight under big voices.
- **Low cut:** an optional highpass for small mouths or nostrils.
- **Mix:** inhales are set about 22 dB below the loudest call. Then come mild saturation, air
  absorption over `distance`, and a short outdoor reverb.

The output is **peak-normalized**, so the game must set loudness itself (see "Integration notes").

---

## What the game should supply for a new creature type

The traits fall into three groups. The first group comes from the creature you've actually
built. The second is what the genome should carry. The third is fixed per species or archetype,
and is set once by whoever designs that creature type.

### A. From the creature's real body (compute these, don't guess)

| Trait | How to get it | Effect |
|---|---|---|
| `mass` (kg) | the creature's mass | Natural pitch (bigger is lower), call length, and the default tract length if none is given. Also sets the chest resonance frequency. |
| `tract_length` (m) | `Body.from_dimensions(mass, head_length, neck_length)` uses 0.8 × head + 0.5 × neck | **The main size cue.** Formant spacing is what makes a creature sound big or small. |
| `lung_capacity` | torso volume relative to a typical animal of that mass (1 = typical) | Longer or shorter calls |
| `source` | the species' class: mammal → `larynx`, bird → `syrinx`, frog or insect → `pulse`, reptile → `hiss` | Picks the sound source |

### B. From the genome (heritable, varies between individuals)

These belong on `Voice`, plus a few tissue traits on `Body`. Ranges are what the presets use;
going outside them is allowed but untested.

| Trait | Range | What it does |
|---|---|---|
| `Voice.vowels` | list of `a e i o u @`, or `(F1, F2, F3[, rounding])` ratios | The creature's timbre. Calls cycle through the list; gliding calls sweep it. This is where the game's genome vowel list goes. |
| `Voice.pitch` | −45 … +30 semitones | Pitch relative to the body's natural pitch. Big negative values are normal for birds, whose natural pitch is songbird-scaled. |
| `Voice.pitch_range` | 0.3 … 2.0 | How far pitch contours swing (flat drone vs expressive). |
| `Voice.harshness` | −1 … +0.3 | Added to body roughness. Low is clean; high is gravelly and chaotic. |
| `Voice.vibrato` | −0.3 … 0.5 semitones | Added to each intent's vibrato. A negative value cancels it (steady howls). |
| `Voice.tempo` | 0.7 … 1.5 | Above 1 gives shorter, quicker calls. |
| `Voice.sustain` | 1 … 3 | Length of long tonal calls (`mating`, `beg`, and any intent with `glide`). |
| `Voice.melody` | 0 … 1 | 1 keeps the intent's up-down melody; 0 makes one smooth swell (howl). |
| `Voice.falsetto` | 0 … 1 | High notes flip to a thin, nearly pure register (howls). |
| `Voice.repetition` | 1 … 3 | Scales how many calls each intent repeats (frogs, ducks, crows). |
| `Voice.subharmonics` | 0 … 1 | Proneness to subharmonic episodes: lower, rougher, bigger. The cat uses 0.3. |
| `Voice.chaos` | 0 … 1 | Proneness to chaotic episodes: snarling, screaming irregularity that stays voiced. |
| `Voice.biphonation`, `biphonation_ratio` | 0 … 1, 1.2 … 2.5 | A second independent tone over the voice, and its pitch ratio (keep it off whole numbers). |
| `Voice.jumps` | 0 … 1 | Register breaks: abrupt pitch leaps and returns. |
| `Body.fold_scale` | 0.4 … 1.8 | Bigger, heavier vocal folds give a lower natural pitch. Also sets where fry starts. |
| `Body.roughness` | 0.05 … 0.8 | Baseline tissue irregularity (half of it becomes baseline harshness). |
| `Body.breathiness` | 0.1 … 1.0 | Breath noise in the voice. |
| `Body.nasality` | 0 … 0.45 | Nasal quality. |
| `Body.larynx_drop` | 0.15 … 0.5 | How far the larynx lowers on loud calls (red deer roar). |

### C. Per species or archetype (set once by the designer)

These describe anatomy that differs by kind of animal rather than by individual. Defaults are
fine for an ordinary mammal.

| Trait | Default | When to change it |
|---|---|---|
| `Body.buzz` | 0 | Syrinx only: 1 for pulsed calls (caws, quacks) even when not harsh. |
| `Body.resonator_freq`, `resonator_harmonic` | from mass, 0 | Pulse sources: vocal sac pitch, plus a second band at its octave (frogs about 1.3). |
| `Body.chest`, `chest_width` | 1, 70 Hz | Chest weight. Widen it (about 180 Hz) to add bass without a ringing beat against the voice. Use 0 if the voice sits near the chest frequency and throbs. |
| `Body.tract_damping` | 1 (syrinx 4) | Formant sharpness: lower gives sharp, peaky formants (duck bill); higher gives soft ones. |
| `Body.hf_loss` | 1 | Below 1 for stiff, bright tracts (pig snout). |
| `Body.open_quotient` | 0.5 | Voice source softness. About 0.4 for pressed roars and grunts (tiger, pig); 0.65 for soft, clean voices (cat, crow). Raise it if a voice is buzzy; lower it if it sounds breathy or like a pure tone. |
| `Body.formant_gain`, `formant_width` | 8 dB, 1 | Resonance at the formants: tone and character. Use 0 for tracts that already ring sharply (duck bill), and about 12 for strongly resonant calls (crow). |
| `Body.breath_pulse` | 0.9 | How much breath noise pulses with the voice. Lower it for an airy, whispery voice. |
| `Body.noise_color` | 1 | Above 1 for bright snorting turbulence through narrow nostrils. |
| `Body.low_cut` | 0 | Hz: small mouths and nostrils radiate lows poorly. |
| `Voice.calls` | {} | The species repertoire. See below. |

### Species repertoire (`Voice.calls`)

`calls` maps an intent name to overrides of any `Intent` field. Use it when a species' calls
don't fit the default templates:

```python
calls={
    # A pig's contact call is a low grunt, far below its squeal.
    "idle": {"pitch": -8.5, "dur": 0.3, "rumble": 0.4, "breath": -0.05, "vowels": ["e"]},
    # A frog's territorial call is its croak chorus, not a roar bout.
    "territorial": {"like": "mating"},
}
```

Overridable fields:

- **Shape:** `pitch`, `contour`, `amp`, `dur`, `repeats`, `gap`, `mouth`, `vowels`
- **Voice quality:** `harsh`, `breath`, `nasal`, `fry`, `rumble`, `vibrato`, `vibrato_rate`
- **Nonlinear phenomena:** `subharmonics`, `chaos`, `biphonation`, `jumps`. These are added to
  the voice's own traits, so 0 doesn't turn off a trait the voice already has.
- **Behaviour:** `larynx`, `inhale`, `glide`, `sequence`, `spread`
- **Series:** `drift` (semitones of pitch change per repeat), `accel`, `rhythm` (below)

#### Tempo over a series

Two fields change how a series of repeats (or the syllables of a sequential phrase) speeds up
or slows down:

- `accel` multiplies the gap by a fixed factor after each repeat: 0.85 speeds up steadily, 1.15
  slows down. It only changes the gaps.
- `rhythm` is a curve of points `[position, spacing]`. Position runs from 0 at the first repeat
  to 1 at the last, and spacing multiplies the gap. Each note's length is multiplied by the
  square root of the spacing, so notes shorten as they crowd together, the way a trill's do. No
  curve (the default) keeps the spacing steady.

A trill that starts slowly, races, then slows at the end (`native/examples/trill_bird.json`):

```json
"mating": {"dur": 0.09, "repeats": [18, 18], "gap": 0.16, "drift": -0.25,
           "rhythm": [[0, 1.6], [0.65, 0.3], [1, 1.1]]}
```

The two combine: `rhythm` shapes the series, and `accel` scales the gaps on top of it.

`"like": "<intent>"` builds the call from another intent's template. Overrides on `territorial`
itself are ignored unless it uses `like`, because the roar bout is a fixed sequence.

The whole description serializes to JSON, so a creature type can live in a data file:

```json
{"body": {"mass": 40, "tract_length": 0.26, "roughness": 0.25},
 "voice": {"vowels": ["u", "o"], "pitch": 8, "calls": {"idle": {"dur": 0.6}}}}
```

```bash
python create_vocalization.py --json creature.json --intent warn
```

---

## Making a new creature type sound right

1. **Start from the closest preset** (`PRESETS` in the code): lion, tiger, wolf, cow, deer, pig,
   cat, mouse, songbird, crow, duck, frog, cricket, snake. Copy it and set the real `mass` and
   `tract_length`.
2. **Get a few real recordings** of the animal, or of whatever it should resemble, and compare
   against them rather than tuning by intuition.
3. **For noisy calls, get the pitch from harmonic spacing.** A pitch tracker can mistake a
   roughness rate for the pitch.
4. **Render every intent:**

   ```bash
   python create_vocalization.py --preset <name> --demo
   ```

   Listen to the files in `vocalizations/`.
5. **If a call is hard to get right, render several distinct candidates and pick by ear.** That
   converges much faster than repeated blind tweaks.

Lessons from tuning the existing presets:

- **Separate beats, a "propeller" or a "weedeater":** the pitch is too low for the pulses to
  fuse, or a resonance is beating against the voice. Keep voiced calls at roughly 90 Hz or above
  unless you want a creak. Widen or remove `chest` if the voice sits near it.
- **Staccato puffs:** too much breath, or modulation that drops the sound to silence. Get
  roughness from the voice instead (`harsh`, `rumble`).
- **Air escaping a balloon:** a clean tone with breath on top. Set `breath` ≤ 0 for that call
  and raise `harsh`.
- **Woodwind or instrument:** a regular vibrato or a melodic contour on a pure tone. Lower
  `melody`, and keep vibrato irregular (it is by default).
- **Muddy:** the tract is too long for the animal, or the mouth is too closed or rounded.
- **Small-sounding:** the tract is too short. Lengthen `tract_length` or raise `larynx_drop`.
  Formant spacing reads as size far more than pitch does.
- **Growls:** real ones are mostly noise (harmonicity near 0 dB). Use high `harsh` and
  `spread`.
- **Howls:** real ones are steady and nearly pure. Use `falsetto`, low `melody` and negative
  `vibrato`.

---

## Integration notes

- **Cost:** one intent renders in about 0.02–0.25 s of CPU (for 1–7 s of audio), and numba
  compiles the tube filter on first use (cached afterwards). Pre-render each creature's set of intents when it spawns, or in
  a background job, rather than synthesizing during gameplay.
- **Loudness:** output is peak-normalized. Set volume in the game from mass and intent (big
  animals and `warn`/`territorial` louder, `idle`/`beg` softer) and from distance.
- **Reverb:** pass `reverb=0` and let the game's audio mixer apply environment reverb.
  `distance` also darkens the sound (air absorption). Leave it at about 20 m and do distance in
  the engine, or set it from the listener.
- **Requirements:** Python 3 with numpy, scipy and numba. `sounddevice` is optional (playback
  only). The algorithm can be ported to C#: the tube and lip filters are simple per-sample loops,
  and the rest is standard IIR filtering.
