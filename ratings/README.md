# Voice ratings

Your scores for voices sampled from voice space. They become the preference field that steers
genome-decoded voices away from regions you dislike. The design is in
[GENOME_VOICE_DESIGN.md](../GENOME_VOICE_DESIGN.md).

## Files

| Path | What it is |
|---|---|
| `rate.py` | The rating page: a local web page, Python standard library only. |
| `axis_check.py` | Measures what each axis (or setting) does in the sweeps. |
| `batches/<round>/` | A batch from `cvsynth sample`: `manifest.jsonl` and the WAVs. |
| `ratings.jsonl` | Every rating, appended. Never edited by hand: the newest line for an id wins. |
| `sweeps/`, `sweeps_settings/` | Renders from `cvsynth sweep`, for checking the axes by ear. |

## Rating a batch

From the repository root, make the batch once:

```powershell
native\target\release\cvsynth.exe sample --count 300 --round r001 --out-dir ratings\batches\r001
```

Then rate it, in as many sittings as you like. The page resumes at the first unrated sample:

```powershell
python ratings\rate.py ratings\batches\r001
```

The page opens in your browser. Each voice plays when it opens.

- **Score**, from −1 to 1: how much you like it **as the voice of the creature shown**. Keys 1–5 jump
  to −1, −0.5, 0, 0.5 and 1. The arrow keys nudge it by 0.1.
- **Wrong size:** "Would suit a bigger animal" (B) or "Would suit a smaller animal" (S). Tick one only
  when you'd probably like the sound on a creature of another size. The score still means "on this
  creature", so a ticked sample usually scores low; the tick tells the model why.
- **Space** replays. **X** skips a sample you can't judge. **Enter** saves and moves on.
  **Backspace** goes back to change the last one.
- **Note** (optional): anything worth remembering.

A batch of 300 also holds:
- about 30 pairs, the same voice on two bodies of very different sizes;
- about 15 repeats of earlier samples.

They aren't marked. The pairs separate "bad sound" from "wrong for its size", and the repeats measure
how consistent ratings are.

## What a batch holds

Each line of `manifest.jsonl` has everything that shaped one sample:

| Field | Contents |
|---|---|
| `body` | The body input: mass, tract, source, shape, lips, teeth. |
| `axes`, `residual` | The position in voice space. |
| `settings` | The resulting synth settings. |
| `intent`, `vowels`, `seed` | What was rendered. |
| `kind` | `broad`, `paired`, `repeat` or `active`. |
| `pair_of`, `repeat_of` | Links between related samples. |
| `takes` | The WAV files. |
| `creature` | The full creature JSON. |
| `space_version` | The voice space it was made under. |

Every take is loudness-matched to −20 dBFS RMS, and heard from 10 m with light reverb. Samples far out
on the instability axis get two takes with different seeds, played one after the other.

## Checking the axes

```powershell
native\target\release\cvsynth.exe sweep --out-dir ratings\sweeps --intent mating
python ratings\axis_check.py ratings\sweeps mouse,fox,wolf,bear
```

**What `sweep` renders:** each axis from −3 to +3 in seven calls, 1.5 s apart, on a mouse, fox, wolf,
bear and crow sized body. Add `--per-setting` to sweep each of the 29 settings alone instead.

**What `axis_check.py` reports:**
- how far each axis moves pitch, pitch spread, centroid, harmonicity, tilt, flatness, low-band energy,
  F1, F2 and roughness, in rough just-noticeable differences per step;
- which axes overlap;
- which are too weak to hear.
