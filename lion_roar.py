"""Source-filter lion roar synthesizer.

Signal chain:
    glottal pulse train (jitter, shimmer, subharmonics, tremor, creaky tails, breath noise)
    -> time-varying cascade formant filter (vocal tract with jaw opening, larynx lowering
       and slow articulator wobble)
    -> lip radiation + chest resonance + nasal coupling when the mouth is nearly closed
    -> soft saturation -> air absorption -> outdoor reverb

A roar bout is modelled the way lions actually do it: a soft intro moan,
one full-throated roar, then a series of grunts with audible inhales between them.

Usage:
    python lion_roar.py                    # synthesize, save lion_roar.wav, play
    python lion_roar.py --seed 7 --no-play
    python lion_roar.py --length 0.30      # shorter vocal tract = smaller cat
    python lion_roar.py --pitch 0.8        # scale all f0 contours
    python lion_roar.py --distance 150     # far away: darker, more reverb
    python lion_roar.py --reverb 0         # completely dry
"""
import argparse
import wave
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, iirnotch, lfilter, sosfilt

SR = 44100
C_SOUND = 350.0  # speed of sound in warm, humid air (m/s)
NUM_FORMANTS = 6
LARYNX_DROP = 0.15  # fractional vocal tract lengthening when the larynx is pulled down
CHEST_GAIN = 0.02   # radiation strips the lows; the chest/body resonance puts weight back


@dataclass
class Call:
    """One vocalization. Curves are (u, value) points over normalized time u in [0, 1]."""
    start: float
    dur: float
    f0: list
    amp: list
    harsh: list          # 0 = clean modal voice, 1 = chaotic/subharmonic growl
    mouth: list          # 0 = closed/nasal, 1 = jaw fully open
    larynx: list = None  # 0 = resting, 1 = fully lowered; defaults to following amp
    voiced: float = 1.0  # 0 for unvoiced sounds such as inhales
    breath: float = 0.25
    fry: float = 0.6     # strength of the creaky, pitch-dropping tail
    seed_offset: int = 0


def curve(points, u):
    xs, ys = zip(*points)
    return np.interp(u, xs, ys)


def rosenberg_pulse(p, open_quotient):
    """Rosenberg glottal flow pulse over phase p in [0, 1)."""
    tp = 0.62 * open_quotient   # opening phase
    tn = 0.38 * open_quotient   # closing phase (shorter = brighter)
    g = np.zeros_like(p)
    opening = p < tp
    g[opening] = 0.5 * (1.0 - np.cos(np.pi * p[opening] / tp))
    closing = (p >= tp) & (p < tp + tn)
    g[closing] = np.cos(0.5 * np.pi * (p[closing] - tp) / tn)
    return g


def glottal_source(call, pitch_scale, rng):
    """Period-by-period pulse train so jitter/shimmer/period-doubling are per cycle."""
    n = int(call.dur * SR)
    flow = np.zeros(n + SR // 10)
    t, k, drift = 0.0, 0, 0.0
    tremor_rate = 5.0 + rng.uniform(0.0, 2.0)
    tremor_phase = rng.uniform(0.0, 2.0 * np.pi)
    while t < call.dur:
        u = t / call.dur
        f0 = curve(call.f0, u) * pitch_scale
        a = curve(call.amp, u)
        h = curve(call.harsh, u)

        # Creaky tail: as air runs out the folds slacken, pitch sags and cycles turn irregular.
        tail = np.clip((u - 0.8) / 0.2, 0.0, 1.0) * call.fry
        f0 *= 1.0 - 0.45 * tail
        h = min(1.0, h + 0.4 * tail)

        # Slow wander plus fast cycle-to-cycle jitter; jitter explodes when harsh (chaos).
        drift = 0.95 * drift + 0.3 * rng.standard_normal()
        jitter = 0.008 + 0.09 * h ** 3 + 0.12 * tail
        period = 1.0 / (f0 * (1.0 + 0.03 * drift + jitter * rng.standard_normal()))
        shimmer = 1.0 + (0.04 + 0.25 * h) * rng.standard_normal()
        a *= 1.0 + 0.07 * np.sin(2.0 * np.pi * tremor_rate * t + tremor_phase)  # respiratory tremor

        # Period doubling: alternate cycles differ in length and strength -> f0/2 subharmonic.
        sub = np.clip((h - 0.35) / 0.5, 0.0, 1.0) * 0.55
        if k % 2:
            period *= 1.0 + 0.12 * sub
            a *= 1.0 - sub
        else:
            period *= 1.0 - 0.12 * sub

        # Pressed (harsh) phonation closes faster -> more high harmonics; shape varies per cycle.
        oq = (0.65 - 0.25 * h) * (1.0 + 0.06 * rng.standard_normal())
        oq = float(np.clip(oq, 0.25, 0.9))
        i0 = int(np.ceil(t * SR))
        i1 = min(int(np.ceil((t + period) * SR)), len(flow))
        idx = np.arange(i0, i1)
        flow[idx] += max(a * shimmer, 0.0) * rosenberg_pulse((idx / SR - t) / period, oq)
        t += period
        k += 1
    return flow


def source_signal(call, pitch_scale, rng):
    """Returns (voiced glottal flow, aspiration noise, normalized time)."""
    n_total = int(call.dur * SR) + SR // 10
    u = np.clip(np.arange(n_total) / (call.dur * SR), 0.0, 1.0)
    amp = curve(call.amp, u)
    harsh = curve(call.harsh, u)

    flow = call.voiced * glottal_source(call, pitch_scale, rng) if call.voiced > 0 else np.zeros(n_total)

    # Aspiration/turbulence noise, pulsing with the glottal flow, louder when harsh.
    noise = sosfilt(butter(2, 250, "highpass", fs=SR, output="sos"), rng.standard_normal(n_total))
    flow_norm = flow / (np.max(flow) + 1e-9)
    gate = 0.3 + 0.7 * flow_norm if call.voiced > 0 else 1.0
    # Noise is broadband, so radiation boosts it far more than the low voiced energy; keep it small.
    aspiration = 0.3 * noise * gate * amp * call.breath * (0.4 + 0.8 * harsh)

    return flow, aspiration, u


def resonator(freq, bw):
    """Klatt-style two-pole resonator with unity gain at DC."""
    r = np.exp(-np.pi * bw / SR)
    b1 = 2.0 * r * np.cos(2.0 * np.pi * freq / SR)
    b2 = -r * r
    return [1.0 - b1 - b2], [1.0, -b1, -b2]


def vocal_tract(x, mouth, larynx, tract_length, rng, block=128):
    """Cascade formant filter. Formants of a uniform tube closed at the glottis; lowering the
    larynx lengthens the tube (all formants drop, sounding bigger), jaw opening raises F1/F2,
    and each formant wanders slowly the way a real tongue and jaw never hold still."""
    tube = np.array([(2 * i - 1) * C_SOUND / 4.0 for i in range(1, NUM_FORMANTS + 1)])
    states = [np.zeros(2) for _ in range(NUM_FORMANTS)]
    wobble = np.zeros(NUM_FORMANTS)
    y = np.empty_like(x)
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        mid = (s + e) // 2
        o = mouth[mid]
        f = tube / (tract_length * (1.0 + LARYNX_DROP * larynx[mid]))
        f[0] *= 1.0 + 1.3 * o
        f[1] *= 1.0 + 0.4 * o
        f[2] *= 1.0 + 0.15 * o
        wobble = 0.995 * wobble + 0.1 * rng.standard_normal(NUM_FORMANTS)
        f *= 1.0 + 0.03 * wobble
        bw = 60.0 + 0.07 * f
        seg = x[s:e]
        for i in range(NUM_FORMANTS):
            b, a = resonator(f[i], bw[i])
            seg, states[i] = lfilter(b, a, seg, zi=states[i])
        y[s:e] = seg
    return y


def nasalize(y, mouth):
    """With the jaw nearly shut, sound leaks through the nose: an antiresonance around
    800 Hz and a dull, muffled top end."""
    b, a = iirnotch(800.0, 1.5, fs=SR)
    nasal = sosfilt(butter(2, 1800, "lowpass", fs=SR, output="sos"), lfilter(b, a, y))
    mix = np.clip(1.0 - 2.0 * mouth, 0.0, 1.0) * 0.7
    return (1.0 - mix) * y + mix * nasal


def render_call(call, tract_length, pitch_scale, seed):
    rng = np.random.default_rng(seed + call.seed_offset)
    flow, aspiration, u = source_signal(call, pitch_scale, rng)
    mouth = curve(call.mouth, u)
    larynx = curve(call.larynx if call.larynx else call.amp, u)

    y = vocal_tract(flow + aspiration, mouth, larynx, tract_length, rng)
    y = lfilter([1.0, -0.97], [1.0], y)  # lip radiation (+6 dB/oct)
    y = nasalize(y, mouth)

    # Chest/body resonance driven by the raw flow: the low thump you feel more than hear.
    b, a = resonator(95.0, 70.0)
    return y + CHEST_GAIN * lfilter(b, a, flow)


def build_bout(rng):
    """Intro moan, full roar, then accelerating grunts with inhales between them."""
    j = lambda v, s=0.06: v * (1.0 + s * rng.standard_normal())
    calls = [
        # Soft, low, mostly closed-mouth (nasal) moan.
        Call(start=0.0, dur=j(0.9), breath=0.3, fry=0.4,
             f0=[(0, 70), (0.5, j(95)), (1, 80)],
             amp=[(0, 0), (0.3, 0.3), (0.8, 0.28), (1, 0)],
             harsh=[(0, 0.15), (1, 0.3)],
             mouth=[(0, 0.05), (0.6, 0.25), (1, 0.1)],
             larynx=[(0, 0), (1, 0.2)]),
        # Inhale before the big one.
        Call(start=1.0, dur=0.35, voiced=0.0, breath=0.5, seed_offset=1,
             f0=[(0, 1), (1, 1)], amp=[(0, 0), (0.5, 0.25), (1, 0)],
             harsh=[(0, 0.4), (1, 0.4)], mouth=[(0, 0.5), (1, 0.6)]),
        # Full-throated roar: pitch and loudness swell, larynx drops, voice breaks into chaos.
        Call(start=1.4, dur=j(2.3), breath=0.3, seed_offset=2, fry=0.8,
             f0=[(0, 85), (0.15, j(140)), (0.4, j(195)), (0.6, j(180)), (0.85, 115), (1, 75)],
             amp=[(0, 0), (0.08, 0.6), (0.35, 1.0), (0.7, 0.9), (0.9, 0.4), (1, 0)],
             harsh=[(0, 0.35), (0.3, 0.75), (0.5, 0.95), (0.75, 0.65), (1, 0.45)],
             mouth=[(0, 0.3), (0.3, 0.9), (0.7, 1.0), (1, 0.4)],
             larynx=[(0, 0.2), (0.3, 1.0), (0.8, 1.0), (1, 0.5)]),
    ]
    t = 4.0
    gaps = [0.95, 0.85, 0.75, 0.7, 0.65]
    for i, gap in enumerate(gaps):
        level = 1.4 - 0.15 * i  # low f0 carries less energy after radiation, so push grunts harder
        calls.append(Call(start=t - 0.3, dur=0.28, voiced=0.0, breath=0.45, seed_offset=10 + 2 * i,
                          f0=[(0, 1), (1, 1)], amp=[(0, 0), (0.5, 0.12 + 0.02 * i), (1, 0)],
                          harsh=[(0, 0.4), (1, 0.4)], mouth=[(0, 0.4), (1, 0.5)]))
        calls.append(Call(start=t, dur=j(0.42, 0.1), breath=0.35, seed_offset=11 + 2 * i, fry=0.9,
                          f0=[(0, 80), (0.25, j(100)), (1, 58)],
                          amp=[(0, 0), (0.12, level), (0.5, level * 0.8), (1, 0)],
                          harsh=[(0, 0.7), (0.4, 0.9), (1, 0.6)],
                          mouth=[(0, 0.4), (0.3, 0.85), (1, 0.3)],
                          larynx=[(0, 0.5), (0.3, 0.9), (1, 0.6)]))
        t += gap
    return calls


def outdoor_reverb(x, rng, wet):
    """Ground reflection plus a short, dark, decaying tail (open savanna, some trees)."""
    tail_len = int(1.3 * SR)
    tt = np.arange(tail_len) / SR
    tail = rng.standard_normal(tail_len) * np.exp(-tt / 0.18)
    tail = sosfilt(butter(2, 2500, "lowpass", fs=SR, output="sos"), tail)
    tail[: int(0.02 * SR)] = 0.0
    ir = np.zeros(tail_len)
    ir[0] = 1.0
    ir[int(0.006 * SR)] = 0.3  # ground bounce
    ir += wet * tail / np.max(np.abs(tail))
    return fftconvolve(x, ir)[: len(x) + tail_len // 2]


def synthesize(seed, tract_length, pitch_scale, distance, reverb):
    rng = np.random.default_rng(seed)
    calls = build_bout(rng)
    rendered = [(c, render_call(c, tract_length, pitch_scale, seed)) for c in calls]
    total = max(int(c.start * SR) + len(y) for c, y in rendered)
    mix = np.zeros(total)
    for c, y in rendered:
        s = int(c.start * SR)
        mix[s:s + len(y)] += y

    mix = sosfilt(butter(2, 35, "highpass", fs=SR, output="sos"), mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    drive = 1.8
    mix = np.tanh(drive * mix) / np.tanh(drive)  # tissue/airflow saturation grit

    # Air absorbs highs over distance; farther sources are also more reverberant.
    cutoff = max(1500.0, 9000.0 * np.exp(-distance / 150.0))
    mix = sosfilt(butter(2, cutoff, "lowpass", fs=SR, output="sos"), mix)
    mix = outdoor_reverb(mix, rng, wet=reverb * min(0.3, 0.03 + 0.002 * distance))
    return 0.9 * mix / (np.max(np.abs(mix)) + 1e-9)


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def play(path, audio):
    try:
        import sounddevice as sd
        sd.play(audio.astype(np.float32), SR)
        sd.wait()
    except Exception:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--length", type=float, default=0.38, help="resting vocal tract length in metres")
    ap.add_argument("--pitch", type=float, default=1.0, help="f0 scale factor")
    ap.add_argument("--distance", type=float, default=20.0, help="listener distance in metres")
    ap.add_argument("--reverb", type=float, default=1.0, help="reverb amount multiplier (0 = dry)")
    ap.add_argument("--out", default="lion_roar.wav")
    ap.add_argument("--no-play", action="store_true")
    args = ap.parse_args()

    audio = synthesize(args.seed, args.length, args.pitch, args.distance, args.reverb)
    write_wav(args.out, audio)
    formants = ", ".join(f"{(2 * i - 1) * C_SOUND / (4 * args.length):.0f}" for i in range(1, NUM_FORMANTS + 1))
    print(f"Wrote {args.out} ({len(audio) / SR:.1f}s). Resting-tract formants: {formants} Hz")
    if not args.no_play:
        play(args.out, audio)


if __name__ == "__main__":
    main()
