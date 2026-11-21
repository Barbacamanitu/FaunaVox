# Genome → voice, natively: design

Status, 2026-11-20:
- **Built:**
  - voice space version 1 (`native/src/space.rs`);
  - `cvsynth sample` and `cvsynth sweep`;
  - the rating page (`ratings/rate.py`) and the axis check (`ratings/axis_check.py`);
  - the first batch, `ratings/batches/r001`, rated;
  - the preference model (`ratings/fit.py` → `field_v1.json`), evaluated natively (`native/src/field.rs`);
  - the genome decoder, version 2 (`native/src/genome.rs`), with steering: no delimiters, windows read as walks, a
    syllable inventory and call patterns;
  - per-syllable length, pitch step, pitch slope and level in the synth (`prosody`);
  - the DLL functions and `cvsynth genome` ([§5](#5-dll-and-unity-interface));
  - the evolvability checks, in voice-space units (`cvsynth genome check`), the mutation table
    (`cvsynth genome mutations`) and lineages (`cvsynth genome --lineage`).
- **Not built yet:**
  - the Unity side: `GenomeVoiceDesigner` calling `cv_creature_from_genome`, the voice section's mutation settings,
    and starter voices made by `cvsynth genome design` (also not built). The DLL and C# binding are ready;
  - the audio-fingerprint version of the mutation check.

Field v1 is inert: no region is confidently disliked yet ([§3](#3-preference-model)).

This moves the voice genome's decoding out of Unity (`VoiceGenome.cs`, `VocalProfile`, `NativeVoiceSpec`) into
`faunavox.dll`. Unity still builds the creature's body and still owns mutation and breeding. It passes the DLL
the body's measurements and the voice section of the genome. The DLL decodes the voice section into a complete
creature spec (body settings, voice traits, a call per intent with its phrase and rhythm), then renders it with the
synth as it is now.

A preference field built from rated samples shapes the decoding. Regions of voice space you've rated badly repel the
walk, so random genomes and mutations tend to steer around the worst-sounding voices without being forbidden from them.

Order of work, and of this document:

1. [Rating tool](#1-rating-tool), with the voice space it samples from ([§2](#2-voice-space)). It doesn't need the
   genome decoder, so it comes first: ratings take the longest to gather.
2. [Preference model](#3-preference-model): fitted from the ratings and frozen into the DLL as a versioned field.
3. [Genome decoder](#4-genome-decoder): overlapping windows of plain DNA, read as walks through voice space, a
   syllable inventory and a pattern per call.
4. [DLL and Unity interface](#5-dll-and-unity-interface).
5. [Evolvability checks](#6-evolvability-checks).

---

## 1. Rating tool

### What is rated

Each sample is one point in voice space ([§2](#2-voice-space)) on one body, heard as one call. You score **how much
you like it as the voice of this creature**, the one whose size and shape are shown.

Samples come from the voice space's own sampling distribution, the spread the genome walk will produce, with the
preference field **off**. Ratings belong to a place in voice space, not to how it was reached, so they don't need
the genome decoder. Once the decoder exists, later rounds can sample through it instead: same space, same ratings.

### Generating a batch

A new `cvsynth sample` command writes a batch folder:

```
ratings/batches/r001/
  manifest.jsonl     one line per sample (below)
  s0001_a.wav        take a (fixed seed per sample)
  s0001_b.wav        take b, only for samples flagged for two takes
  ...
```

```powershell
cvsynth sample --count 300 --seed 1 --round r001 --out-dir ratings/batches/r001
```

Each manifest line holds everything that shaped the sound, so the rating can be reused after anything changes:

```json
{
  "id": "r001-s0001",
  "space_version": 1,
  "body": {"mass": 12.4, "source": "larynx", "shape": "quadruped", "head_length": 0.14, "neck_length": 0.09,
           "lips": true, "teeth": true},
  "axes": [0.4, -1.1, 0.2, 0.9, -0.3, 0.0, 1.7, -0.6, 0.1, 0.8],
  "residual": [0.02, -0.01, ...],
  "intent": "mating",
  "phrase": "aa eh~ih",
  "seed": 17,
  "kind": "broad | paired | repeat | active",
  "pair_of": null,
  "creature": { "...": "the full creature JSON the synth rendered" },
  "takes": ["s0001_a.wav"]
}
```

Rules for a batch:

- **Bodies:** mass spread evenly on a log scale over the range the game spawns. Each body's source and shape class
  come from that range too.
- **Intent and phrase:** one intent and one short fixed phrase per batch, or a small fixed set. You're judging the
  timbre, not one unlucky rhythm. Rhythm and phrases are rated separately later ([§7](#7-open-questions)).
- **Loudness:** every take normalized to the same loudness, with the same distance and reverb.
- **Mix:**

  | Kind | Share | What it's for |
  |---|---|---|
  | Broad | ~85 % | Random points from the sampling distribution. |
  | Paired | ~10 % | The same body-relative point on two bodies at least 8× apart in mass, shown far apart in the order. They separate "bad sound" from "wrong for this size". |
  | Repeat | ~5 % | A sample already rated, shown again later without saying so. They measure how consistent ratings are, which sets the model's noise level. |

- **Takes:** points with strong instability (subharmonics, chaos, biphonation) get two takes with different seeds,
  played one after the other, so one odd render doesn't condemn a region.
- **Order:** shuffled.

### The page

A small local page, served by a stdlib-only Python script (`ratings/rate.py`) that reads a batch's manifest and
serves its WAVs. Run it, open the page, rate, close. It stores no state except the ratings file.

```powershell
python ratings/rate.py ratings/batches/r001
```

Layout, top to bottom:

1. **The creature:**
   - A silhouette of its shape class, scaled against a human outline, plus its mass in words and numbers ("12 kg,
     about fox-sized").
   - The intent ("mating call").
   - Nothing else about its settings: you rate the sound, not the numbers.
2. **Play:** it plays automatically when the sample opens. Space replays it, and for two-take samples it plays both.
3. **Score:** a slider from −1 to 1, starting at 0 and marked at −1, −0.5, 0, 0.5 and 1. Keys 1–5 jump straight to
   those marks; the arrow keys nudge it by 0.1.
4. **Wrong-size tick:** two buttons, of which at most one can be on:
   - **"Would suit a bigger animal"** (key B)
   - **"Would suit a smaller animal"** (key S)

   Use one only when the sound itself is one you'd probably like, just not on this body. The score still means "how
   much I like it on *this* creature", so a ticked sample usually has a low score. The tick tells the model why.
   Leave it off when the sound is bad whatever animal it came from, or when it's fine here.
5. **Skip** (key X): can't judge it (glitch, nothing heard). Recorded but not used to fit.
6. **Note** (optional text): anything worth remembering, such as "nice but too long".
7. **Next** (Enter) saves and moves on. **Back** (Backspace) returns to the previous sample to change it.
8. **Progress:** "112 / 300" and a running count of skips.

### Ratings file

One line appended per rating to `ratings/ratings.jsonl`, never rewritten. A changed rating is a new line with a later
time, and the newest line for an id wins:

```json
{"id": "r001-s0001", "score": -0.5, "size": "bigger", "skip": false, "note": "", "time": "2026-11-21T19:04:11Z",
 "listens": 2, "rater": "russ"}
```

`size` is `"bigger"`, `"smaller"` or `null`. `listens` counts how many times it was played: a lot of replays marks a
hard call. The manifests stay next to it, so ratings can be refitted against any later version of the space.

### Rounds

| Round | Samples | How they're chosen |
|---|---|---|
| r001 | ~300 | Broad, paired and repeats, as above. Roughly an hour and a half of listening. |
| r002 onward | ~150 each | Mostly chosen by the model ([§3](#3-preference-model)): where it's least sure, or near the line between liked and disliked. Plus broad samples (about a quarter) so nothing goes unchecked. |

Stop when a round barely moves the fitted field: the test is how much its predictions on a fixed set of check points
change.

---

## 2. Voice space

The synth has 31 settings that shape the voice itself (21 on the body, 10 on the voice). Two come from Unity and
aren't walked:
- `mass` is the creature's;
- `source` (larynx, syrinx, pulse or hiss) comes from its shape class.

That leaves 29 continuous settings. The walk and the ratings don't move them directly. They move:

- **10 perceptual axes**, each a direction that turns several settings together the way listeners hear one quality
  changing;
- **29 small residual offsets**, one per setting, heavily damped, so odd combinations stay reachable.

### Body prior

Every position is relative to what the body predicts. The **body prior** maps Unity's measurements to the expected
settings:

- the tract length expected from head and neck length;
- the pitch expected from mass;
- the source from the shape class;
- the usual defaults for everything else.

Axis 0 everywhere is "an ordinary voice for this body". So the same coordinates mean "pitched 5 semitones above what
it should be" on a mouse and on a bear, and a rating carries over between sizes.

### The axes

| # | Axis | Settings it moves, and which way (+) | Heard as |
|---|---|---|---|
| 1 | Resonance scale | `tract_length` up (relative to the prior), `larynx_drop` up | Bigger or smaller than it is |
| 2 | Pitch height | `fold_scale` down, `pitch` up, `falsetto` up | Higher, apart from size |
| 3 | Pitch liveliness | `pitch_range`, `vibrato`, `jumps` up | Monotone → expressive, yelpy |
| 4 | Pressed ↔ breathy | `breathiness` and `breath_pulse` up; `open_quotient` up; `buzz` down | Tight and buzzy → soft and airy |
| 5 | Roughness | `roughness`, `harshness` up | Smooth → gravelly |
| 6 | Instability | `subharmonics`, `chaos`, `biphonation` up | Clean → screams, snarls, two tones at once |
| 7 | Brightness | `hf_loss` down, `noise_color` brighter, `low_cut` up | Dark and muffled → bright and thin |
| 8 | Ringiness | `formant_gain` up, `formant_width` down, `tract_damping` down | Dull → clear, ringing vowels |
| 9 | Nasality | `nasality` up | Oral → through the nose |
| 10 | Body resonance | `chest` up, `chest_width`, `resonator_freq`, `resonator_harmonic` | No thump → chest thump or air-sac boom |

`biphonation_ratio` and `lung_capacity` are residual-only. Lung capacity changes timing, not timbre.

**From coordinates to settings:** each axis is a vector of weights over the 29 settings. A position `a` (10 axes) plus
residual `r` (29) gives a raw value per setting:

```
raw[k] = Σ_i W[i][k] · a[i] + r[k]
```

That raw value then goes through the setting's own curve, centred on the body prior. The curve is `tanh` for bounded
settings, so every coordinate is valid and extremes take more pushing to reach. A log scale with `tanh` limits is
used for frequencies and lengths. For example:

```
tract_length = prior.tract × 2^(0.35 · tanh(raw))
```

`W` and the curves are tables in Rust (`space.rs`) with a version number (`space_version`). Ratings keep the version
they were made under. A new space version means re-mapping old ratings by their stored creature JSON (a
least-squares fit back onto the new axes), or rating again.

**Per source:** some axes do nothing for some sources:
- resonance scale on a hiss;
- larynx drop on a syrinx;
- most of the voice settings on a pure hiss.

The space stays the same shape, but the preference model is told the source, so it doesn't spread a judgment made on
one source to another.

**Sampling distribution:** axes are drawn from a Student-t distribution (ν = 4, scale 1): mostly near the middle, with
some far out, about what a genome walk produces. Residuals are drawn from a normal distribution with scale 0.1. This
is what `cvsynth sample` uses, and what the genome walk should roughly match. [§6](#6-evolvability-checks) checks
that.

### Checking the axes before rating

Before round r001:

1. **Sweeps:** for each axis, render −3 to +3 in 7 steps on 4 bodies (a mouse, a fox, a wolf and a bear sized
   creature). Listen: each should sound like one thing changing steadily. About 30 short files.
2. **Independence:** on 500 random points, measure audio features: spectral centroid, harmonics-to-noise ratio, a
   roughness measure, and F1/F2 estimates. Check that every axis moves its own features. If two axes move the same
   ones, merge them; if an axis barely changes anything, drop it.

### Axis check, space version 1

Measured with `cvsynth sweep --intent mating` and `ratings/axis_check.py`, on mouse, fox, wolf and bear sized bodies.
Effects are in rough just-noticeable differences per axis step:

| Axis | Effect | Mostly moves |
|---|---|---|
| Pitch height | 14 | pitch |
| Roughness | 7 | pitch spread, flatness, roughness |
| Ringiness | 3.3 | pitch spread, flatness |
| Instability | 2.2 | pitch spread, pitch |
| Pitch liveliness | 2.1 | pitch, pitch spread |
| Resonance scale | 2.0 | centroid, low band, flatness |
| Breathy | 1.9 | low band, flatness |
| Brightness | 0.9 | low band, tilt |
| Body resonance | 0.5 | low band |
| Nasality | 0.4 | flatness |

**Changes this made to the ranges:**
- The nonlinear phenomena (subharmonics, chaos, biphonation) reach 1.0, and falsetto and jumps do too. Episodes are
  random and rare at low trait levels, so short calls often had none.
- Vibrato reaches 1.2.
- Chest reaches 8: its resonance is barely audible below about 3.
- Nasality reaches 1.0: the synth subtracts mouth openness and clips the mix at 0.5, so with an open mouth
  values up to 0.5 usually did nothing.

**Settings with no measurable effect on mammals:**
- *Expected:* `buzz` and `resonator_*` (other sources only), `larynx_drop` (warn calls only), and
  `biphonation_ratio` (only when there is biphonation).
- *Possibly real but outside what was measured:* `breath_pulse`, `falsetto` (high pitches only), `vibrato`, and
  `low_cut`.

**Nasality and body resonance stay weak.** They're kept for now, and the ratings' fitted length scales will show
whether they matter. If not, they merge into other axes in space version 2.

---

## 3. Preference model

### Fitting

`ratings/fit.py` (numpy and scikit-learn) fits a Gaussian process to the ratings.

**Inputs:**
- the 10 axis coordinates;
- log mass;
- the source, as one-hot values;
- the intent, as one-hot values. Some things are only bad in some calls: chaos suits a threat and ruins a mating
  call.

The residual offsets are left out, because their effect is small by design.

**Kernel and noise:**
- The kernel is an RBF with a separate length scale for each input. The fit learns which axes your taste depends on.
- The noise level comes from the repeat samples' spread.
- Skipped samples are left out.

**Score:** the rated score, used as it is.

**The wrong-size tick** adds one pseudo-rating, at the same body-relative point with the mass moved in the ticked
direction:
- **"Would suit a bigger animal":** mass × 6.
- **"Would suit a smaller animal":** mass ÷ 6.
- **Score:** `max(score, 0.4)`.
- **Weight:** half a normal rating, through a larger noise term for that point.

This teaches the model that the valley here leans with size instead of being bad everywhere. The factor and the score
floor are tunable after round r001.

The paired samples need no special handling: they're two ordinary points at different masses, and the length scale on
log mass decides how much they agree.

**Output:** a predicted score `μ(x)` and its uncertainty `σ(x)` at any point.

### Badness, the field the walk uses

The walk only cares where it's confidently bad:

```
badness(x) = softplus(β · (−μ(x) − κ · σ(x) − θ)) / β
```

| Symbol | Meaning |
|---|---|
| θ (about 0.2) | Scores above −0.2 count as fine. |
| κ (about 1) | Where the model is unsure, `σ` is large and badness drops to zero. Unexplored regions stay open. |
| β | The softness of the switch-on. |

Only dislikes repel. There's **no attraction toward liked regions**, which would pull every creature into a few sweet
spots. Liked voices are used another way, as archetype anchors ([§4.3](#43-the-walk)).

### Freezing it into the DLL

`fit.py` exports the fitted model as `field_vN.json`:
- the training inputs (or a reduced set of inducing points);
- their weights;
- the kernel's length scales and variance;
- θ, κ and β;
- the space version it belongs to.

`build.rs` compiles it into the DLL as data. Rust evaluates `badness` and its gradient exactly: an RBF kernel's
gradient has a closed form. With about 1,000 points over 13 inputs, that's microseconds per step.

Every field has a version number. Every decoded creature records the field version (and space and decoder versions)
it was decoded under. A save keeps its voice until the game deliberately upgrades.

### Choosing later rounds

`fit.py --propose 150` writes the next batch's points:
- About 75 % are chosen for what the model would learn most from: high `σ` near `μ ≈ θ`, the region where the
  model is least sure whether a voice crosses the line. Points are kept spread out from each other.
- About 25 % are broad samples.

`cvsynth sample --points proposals.jsonl` renders them.

---

## 4. Genome decoder

Decoder version 2. The voice section is a plain string of bases: no promoters, stop codons, genes or brackets. Anything
but A, C, G and T is skipped. Unity's mutation and recombination work on it unchanged.

*History:* version 1 read TATA genes as rewrite rules for an L-system walk, starting from an anchor picked by three
codons. Its mutation table showed the problems: a single codon or frameshift could swap the whole starting voice, a
call op could swallow the rest of the walk, and most mutations landed in rules nothing used. Version 2 replaced it.

### 4.1 Windows

The section is read as overlapping 4-base windows: bases 1–4, 2–5, 3–6 and so on. Each of the 256 possible windows has
a fixed vector of 73 trait values, drawn once from a seeded generator and centred and scaled to unit variance per trait.

A **trait** reads some windows with weights `w`: `Σ w·v / √(Σ w²)`. Over random DNA every trait is then spread like a
standard normal, whatever the length. A mutation changes only the windows that overlap it: 4 for a point change, and
frameshifts don't matter.

### 4.2 Layout

| Part | Reads (as shares of the section) | Gives |
|---|---|---|
| Base voice | All of it | The 10 axes as a walk, and the 29 residuals (spread 0.1) |
| Inventory | The first 25 % | 2–6 syllable types |
| Phrases | The rest, in 8 equal stretches: `speak`, then mating, warn, alarm, idle, flee, pain, beg | Each call's pattern, syllable count, rhythm and timbre offset |

Positions are shares of the section's length, so the layout stretches with it. The standard length is 1,600 bases
(`SECTION_LENGTH`); any length decodes.

### 4.3 The base voice and call offsets are walks

The windows are read in DNA order, each one a step: its 10 axis values × 0.9 / √n. Without the field the walk ends at
the plain sum, so the base voice spreads 0.9 per axis over random DNA and a point mutation moves it about 0.06 per
axis (about 0.2 in 10 dimensions).

Each call's timbre offset is a second walk, from the base voice, through its own stretch's windows with the offset
trait values × 0.3. It's deflected under its own intent in the field. A call's offset then depends only on its stretch.

**Deflection,** at every step `Δ`:

```
g     = ∇badness(x)          # x = axes + log mass + source + intent (an even mix for the base voice)
if |g| > 0 and Δ·g > 0:      # heading into a disliked region
    ĝ  = g / |g|
    λ  = λmax · (1 − e^(−badness(x) / b0))       # stronger the closer it is
    Δ' = Δ − λ (Δ·ĝ) ĝ                           # remove that share of the downhill part
    s  = sqrt(1 − (Δ·ĝ / |Δ|)²)                  # room to slide sideways: 1 glancing, 0 head-on
    Δ' = Δ' · (|Δ'| + (|Δ| − |Δ'|) · s) / |Δ'|  # give back the lost length in proportion to s
```

- `λmax` 0.9, `b0` 0.3. Steps that lead away from a disliked region are never changed.
- A glancing step slides past at full length and a head-on one slows down. The plan restored the full length, but
  head-on that gives back exactly the removed step, so it did nothing.
- Order matters only near disliked regions: without the field the endpoint is a sum. Near one, two nearly identical
  paths can occasionally split around opposite sides of it.
- Speed: about 2,650 field evaluations per creature. Each skips training points too far away in mass, source or
  intent, and the costly uncertainty term wherever even its floor rules badness out. About 20 ms per creature.

### 4.4 The inventory

The type count is `round(4 + 1.2 z)`, clamped to 2–6, read from the whole inventory. Six type slots sit evenly along
the inventory, and each type reads its slot's windows through a bell (sd 0.45 of a slot), so types barely share
windows. Types past the count are decoded but unused. Each type:

| Trait | Mapping |
|---|---|
| Vowel | F1 ratio `1 + 0.3 z` (0.5–1.5), F2 ratio `1.05 + 0.3 z` (0.55–1.6). F3 and lip rounding follow F2. |
| Glide | When its score is above 0.8 (21 %), the syllable glides to a second vowel, read the same way. |
| Consonant | Constriction, cut into bands ordered by how closed the mouth gets: none (z < −0.2), breath (h), fricative, stop (z > 0.9). Place in five equally likely bands: lips → teeth → ridge → palate → velum. Voicing for stops: z > 0. A small change moves one band, so "a" can become "ha", but never "ta" in one step. |
| Prosody | Length `e^(0.35 z)` (0.5–2), pitch step `3 z` semitones (±8), pitch slope across the syllable `4 z` (±10), level `e^(0.2 z)` (0.5–1.2). |

| Manner | Lips | Teeth | Ridge | Palate | Velum |
|---|---|---|---|---|---|
| Stop | p / b | t / d | t / d | ch / j | k / g |
| Fricative | f | th | s | sh | h |
| Breath | h | h | h | h | h |

**Gated by the body:** no lips (a beak) moves labials to the teeth: p and b become t and d, f becomes th. *Not built
yet:* no teeth should weaken s and sh by 6 dB; the synth has no level per consonant.

### 4.5 Phrases

Each phrase's stretch gives:
- **Syllable count:** `round(typical + spread · z)`, clamped to the intent's range:

  | Phrase | Range | Typical | Spread | Over random DNA |
  |---|---|---|---|---|
  | speak | 2–8 | 5 | 2 | |
  | mating | 2–8 | 4 | 1.6 | 2: 16 %, 3: 21 %, 4: 27 %, 5: 21 %, 6: 11 %, 7: 4 %, 8: 1 % |
  | alarm | 1–5 | 2 | 1.2 | 1: 31 %, 2: 35 %, 3: 25 %, 4: 8 %, 5: 1 % |
  | warn, idle | 1–3 | 2 | 0.67 | |
  | flee, beg | 1–4 | 2.5 | 1 | |
  | pain | 1–2 | 1.5 | 0.33 | |

  Mating and alarm reach their longest phrases only at z = 2.5, about 1 creature in 200 at random. The long songs
  and alarm series are there for selection to find.
- **Slots:** as many as the range's most, evenly along the stretch. Each reads its windows through a bell (sd 0.6 of a
  slot), so neighbours share some windows and often pick the same type: runs and trills emerge (a syllable repeats
  the one before it about a third of the time). Each slot picks an inventory type by which of `count` equally likely
  bands its score falls in, and a link to the next: glide into it (z > 1, 16 %) or pause after it (z < −1.2, 12 %).
  Slots past the count are decoded but silent, so a syllable that appears when the count grows has been drifting all
  along.
- **Rhythm:** the gap between syllables × `e^(0.4 z)` (0.4–2.5); acceleration 0.85 (z < −0.8), 1.15 (z > 0.8) or 1;
  extra repeats `round(0.9 z)` (0–2, short calls only); and a 5-point swell curve, `e^(0.25 z)` each.

**Glides:** a glide-linked slot joins the syllable before it, keeps that syllable's prosody and drops its own
consonant. A syllable glides through at most 3 vowels, because the synth lengthens a syllable for each one.

### 4.6 Output

The decoder's output is an ordinary creature spec, the same `Creature` the JSON loader makes:

- **`body` and `voice`:** the body prior plus the base voice and residuals, through the curves. `voice.vowels` is the
  inventory's vowels, for intents the genome doesn't shape (territorial).
- **`voice.calls.speak`:** the speak phrase and its prosody.
- **Each of the 7 calls,** as an `IntentOverride`, always sequential:
  - its timbre offset as the per-intent settings: pitch, contour range, `harsh`, `breath`, `nasal`, `vibrato`, the
    four nonlinear-phenomena amounts and `larynx`. Offsets in brightness, ringiness and body resonance have no effect
    on a call yet;
  - `vowels` and `prosody` from its phrase;
  - `dur`: the template's call length divided by the phrase's length in plain syllables (each syllable's prosody
    length, and 0.7 more per extra glide vowel), so a phrase lasts about as long as the template's call;
  - `gap`, `repeats`, `accel` and the `rhythm` curve.

The synth's change is per-syllable `prosody` in sequential calls: `[length ×, pitch step, pitch slope, level ×]`,
plain past the end of the list. Without it, everything renders as before. The same DNA, body and versions always give
the same creature; variation between calls stays in the render seed.

**Call lengths over 300 random creatures** (median / 95th percentile / longest): mating 6.4 / 11.8 / 15.4 s, warn
3.7 / 6.6 / 9.7 s, alarm 2.4 / 5.5 / 15.2 s, speak 1.9 / 3.7 / 6.1 s.

---

## 5. DLL and Unity interface

New C functions, in the style of the existing `cv_*` ones:

| Function | Purpose |
|---|---|
| `cv_creature_from_genome(body_json, dna, versions_json)` | Decodes and returns a creature handle, like `cv_creature_from_json`. `versions_json` is optional: by default, the current space, field and decoder versions. The existing `cv_creature_to_json` then gives the decoded creature as JSON, for debugging and `cvsynth --json`, and `cv_render` renders it. |
| `cv_genome_report(handle)` | The base voice and each call on the axes (and each call's offset), badness at each, the inventory and phrases as text, the steps and deflections, and the versions used. |
| `cv_field_version()`, `cv_decoder_version()` | The field and decoder built into this DLL. |

All three are built, with matching C# methods in `native/unity/CreatureVoiceNative.cs` (`FromGenome`, `GenomeReport`,
`DecoderVersion`, `FieldVersion`). The field is compiled in from `native/data/field_vN.json`. After refitting, copy the
new file there and rebuild.

**`body_json` from Unity:**

```json
{"mass": 12.4, "head_length": 0.14, "neck_length": 0.09, "shape": "quadruped", "lips": true, "teeth": true}
```

**`cvsynth`** gets matching commands:
- `cvsynth genome --body body.json --dna "…" --out-dir DIR` renders a creature from its genome (speak and the 7 calls),
  with its DNA, JSON and decoder report. `--random N` makes random sections, `--no-field` turns steering off, and
  `--lineage 1,5,20 [--rate X]` adds descendants mutated as Unity does;
- `cvsynth genome mutations` prints the mutation table, and `cvsynth genome check` runs [§6](#6-evolvability-checks);
- `cvsynth sample`, from [§1](#1-rating-tool).

**Unity:**
- `GenomeVoiceDesigner` builds `body_json` from the creature and calls `cv_creature_from_genome` with the voice section.
  It no longer goes through `VocalProfile` and `NativeVoiceSpec` for version 2 genomes.
- `VoiceGenome.cs` keeps mutation and recombination. Version 2 sections can't be hand-written, so `Compile` goes. The
  starter voices become DNA strings found by a design tool (`cvsynth genome design`, not built yet), which mutates
  and selects DNA until it decodes to a target voice and phrases.
- The voice section's mutation rates need scaling down; see the results below.
- Saves store the versions each creature was decoded under.

---

## 6. Evolvability checks

Built as `cvsynth genome check [--count N]`, in voice-space units. The audio-fingerprint version (MFCC distance)
is still to do. Run it after any change to the codon table, the op shares or the field:

1. **Coverage:** decode 2,000 random genomes on mixed bodies and plot their axis positions against the sampling
   distribution from [§2](#2-voice-space). The walk should fill the space, not one corner.
2. **Mutation spectrum:**
   - Take 200 genomes. Apply every kind of single mutation (point, insertion, deletion, gene duplication, inversion)
     several times each.
   - Render before and after, and measure the sound distance between them (a spectral fingerprint such as MFCC
     statistics).
   - The histogram should be heavy-tailed: mostly small, a few large, almost no "everything changed". Report it per
     mutation kind.
3. **Steering:** the same random genomes decoded with the field on and off. Report:
   - the share that land in disliked regions (it should drop a lot);
   - the overall spread of voices (it should barely shrink). Shrinking spread is the sign the field has become an
     attractor in disguise.
4. **Determinism:** decode the same DNA twice and compare the outputs bit for bit.

### Results, decoder version 2, field v1

300 random 1,600-base sections on random bodies (`cvsynth genome check`).

**Coverage:** the base voice spreads 0.83–0.96 per axis, centred within ±0.13. Inventories hold 2–6 types (12 %, 21 %,
33 %, 20 % and 12 %). 57 % of types start with a consonant, and 22 % glide to a second vowel. A creature's 7 calls have
6.5 distinct phrases on average.

**Mutation spectrum.** Base voice distance moved in 10 dimensions (the base voice spreads about 2.8), and what else
changed:

| Kind | Median | < 0.2 | < 0.5 | ≥ 0.5 | Inventory changed | 1 phrase changed | 2+ phrases changed |
|---|---|---|---|---|---|---|---|
| Point | 0.20 | 51 % | 49 % | 0 % | 13 % | 30 % | 12 % |
| Insertion (1–3) | 0.18 | 68 % | 32 % | 0 % | 24 % | 26 % | 63 % |
| Deletion (1–3) | 0.18 | 66 % | 34 % | 0 % | 24 % | 24 % | 62 % |
| Duplication (6–60) | 0.38 | 7 % | 74 % | 19 % | 69 % | 1 % | 99 % |
| Inversion (3–30) | 0.40 | 4 % | 70 % | 26 % | 20 % | 45 % | 21 % |
| One generation, Unity's default rates | 0.69 | 0 % | 14 % | 86 % | 85 % | 0 % | 99 % |
| One generation, rates × 0.2 | 0.29 | 21 % | 71 % | 7 % | 33 % | 15 % | 63 % |

- **Point mutations behave as designed.** They move the base voice a little and change at most one phrase, unless
  they hit the inventory (13 %), which changes every phrase using that type.
- **Insertions and deletions ripple.** Positions are shares of the length, so every window after the site shifts by
  a base or so against the slot bells. Each downstream score changes slightly, and with many thresholds downstream, a
  syllable somewhere flips about half the time.
- **Per generation it's far too fast at Unity's rates.** Those rates are per base. On 1,600 bases they give each child
  about 10 point changes, 5 small insertions or deletions, and a 20 % chance of a duplication. Every child then has a
  changed inventory and changed phrases. How much a trait drifts per generation doesn't depend on section length:
  longer sections take more mutations, each smaller. So the rate is the lever. Hear it with
  `cvsynth genome --genome native/examples/genomes/dusk_hound.json --lineage 1,2,5,20 --out-dir out`, with and
  without `--rate 0.1`.
- **Decided:** in the game, the voice section gets 0.1–0.2 of the default rates and no insertions, deletions,
  duplications or span deletions, so its length stays fixed and nothing ripples. Point changes, inversions and
  reversals still add syllables, vowels and consonants, through the scores (§4.5).

**Steering:** field v1 is inert. 28 % of genomes had a step deflected, by a negligible amount (mean 0.004), and the
spread of voices is unchanged. A unit test with a made-up disliked spot shows a glancing walk sliding around it and a
head-on walk slowing before it.

**Determinism:** byte for byte.

---

## 7. Open questions

- **Rhythm and phrase ratings:** after timbre is settled. The same tool, with a fixed timbre and walked rhythms.
- **Version 1 genomes:** existing saves and the hand-written starters. Either keep the old C# path for them, or write a
  one-time migration that compiles each version 1 voice to the nearest version 2 genome (by its decoded settings).
- **Shape classes:** the exact list (quadruped, bird, insect, reptile, …) and each one's source and body-prior
  defaults. They come from `Archetypes.cs` and the body genome.
- **Anchor set:** which presets, and how many top-rated voices, become archetype anchors. Too many top voices and
  creatures start sounding alike.
- **Size range:** the mass range for sampling, so it matches what the game actually spawns.
