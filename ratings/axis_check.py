"""Check the voice-space axes from the sweeps (cvsynth sweep): how much each axis moves a set of audio
features, and which axes move the same ones.

    python ratings/axis_check.py ratings/sweeps

Each sweep file holds seven calls, at -3 .. +3 on one axis, separated by silence. For every call it
measures: pitch, spectral centroid, harmonicity, spectral tilt, spectral flatness (noisiness), F1 and
F2 estimates, and roughness (fast loudness flutter). An axis's effect on a feature is the slope across its seven steps, in units
of roughly the smallest change a listener notices in that feature (JND), averaged over the bodies. Then:

- the effect table: which features each axis moves (under 0.3 of a just-noticeable difference per step is shown as .);
- the overlap table: cosine similarity of the axes' effect vectors (above 0.8 means the two axes
  sound alike and are candidates to merge);
- weak axes: an axis whose largest effect is small barely changes anything.
"""
import os
import sys
import wave

import numpy as np

FEATURES = ["pitch", "pitch_spread", "centroid", "harmonicity", "tilt", "flatness", "low", "F1", "F2", "roughness"]
# Rough just-noticeable differences: octaves for pitch, centroid and formants, dB for tilt, log10 for
# flatness, and the 0-1 scales of harmonicity and roughness.
JND = {"pitch": 0.03, "pitch_spread": 0.03, "centroid": 0.1, "harmonicity": 0.05, "tilt": 1.0, "flatness": 0.1, "low": 1.0, "F1": 0.1, "F2": 0.1,
       "roughness": 0.03}


def load(path):
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    return x, w.getframerate()


def split_calls(x, sr):
    """The seven calls, split at the silences between them (each sweep step is followed by 1.5 s)."""
    hop = int(0.01 * sr)
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    loud = env > 0.01 * env.max()
    calls, start, quiet = [], None, 0
    for k, on in enumerate(loud):
        if on:
            if start is None:
                start = k
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= 100:  # 1 s of quiet ends a step
                calls.append(x[start * hop:(k - quiet + 1) * hop])
                start, quiet = None, 0
    if start is not None:
        calls.append(x[start * hop:])
    return calls


def f0_and_harmonicity(seg, sr, fmin=40.0, fmax=3000.0, threshold=0.15):
    """YIN: the first lag whose cumulative-mean-normalized difference dips under the threshold (or the
    lowest one). Harmonicity is 1 minus that dip: near 1 for a clean voice, lower for breath and chaos."""
    seg = seg - seg.mean()
    w = len(seg) // 2
    lo, hi = max(2, int(sr / fmax)), min(int(sr / fmin), w - 1)
    d = np.array([np.sum((seg[:w] - seg[t:t + w]) ** 2) for t in range(hi + 1)])
    cm = np.cumsum(d[1:]) / np.arange(1, hi + 1)
    cmndf = np.ones(hi + 1)
    cmndf[1:] = d[1:] / (cm + 1e-12)
    k = None
    for t in range(lo, hi):
        if cmndf[t] < threshold:
            while t + 1 < hi and cmndf[t + 1] < cmndf[t]:
                t += 1
            k = t
            break
    if k is None:
        k = lo + int(np.argmin(cmndf[lo:hi]))
    return sr / k, float(1.0 - cmndf[k])


def lpc_formants(seg, sr, order=12):
    step = max(1, sr // 11025)
    s = seg[::step]
    fs = sr / step
    s = np.append(s[0], s[1:] - 0.9 * s[:-1]) * np.hamming(len(s))
    r = np.correlate(s, s, "full")[len(s) - 1:len(s) + order]
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    a = np.linalg.solve(R + 1e-9 * np.eye(order) * r[0], r[1:order + 1])
    roots = np.roots(np.concatenate([[1], -a]))
    roots = roots[np.imag(roots) > 0]
    f = np.angle(roots) * fs / (2 * np.pi)
    bw = -np.log(np.abs(roots)) * fs / np.pi
    f = sorted(fi for fi, b in zip(f, bw) if 120 < fi < 5000 and b < 700)
    return (f + [np.nan, np.nan])[:2]


def features(call, sr):
    """Medians over 40 ms frames of the call's loud part, so pitch movement doesn't smear them."""
    n = int(0.04 * sr)
    frames = [call[i:i + n] for i in range(0, len(call) - n, n // 2)]
    rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    loud = [f for f, r in zip(frames, rms) if r > 0.3 * rms.max()]
    rows = []
    for seg in loud:
        sp = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2 + 1e-12
        fr = np.fft.rfftfreq(n, 1 / sr)
        m = fr > 50
        centroid = (sp[m] * fr[m]).sum() / sp[m].sum()
        tilt = 10 * np.log10(sp[fr > 2000].sum() / sp[(fr > 50) & (fr <= 2000)].sum())
        flat = np.exp(np.mean(np.log(sp[m]))) / np.mean(sp[m])  # 1 = noise, 0 = pure tones
        f0, harm = f0_and_harmonicity(seg, sr)
        low = 10 * np.log10(sp[(fr > 20) & (fr < 1.3 * f0)].sum() / sp[fr > 20].sum() + 1e-12)  # at and under f0, dB
        f1, f2 = lpc_formants(seg, sr)
        rows.append([np.log2(f0), np.log2(centroid), harm, tilt, np.log10(flat), low, np.log2(f1), np.log2(f2)])
    # roughness over the whole loud part: share of loudness fluctuation between 20 and 150 Hz
    body = np.concatenate(loud) if loud else call
    env = np.convolve(np.abs(body), np.ones(int(sr / 400)) / int(sr / 400), "same")
    es = np.abs(np.fft.rfft(env - env.mean())) ** 2
    ef = np.fft.rfftfreq(len(env), 1 / sr)
    rough = es[(ef > 20) & (ef < 150)].sum() / (es[(ef > 1) & (ef < 400)].sum() + 1e-12)
    arr = np.array(rows) if rows else np.full((1, 8), np.nan)
    med = np.nanmedian(arr, axis=0)
    spread = np.nanstd(arr[:, 0])  # how far the pitch moves within the call, octaves
    return [med[0], spread] + list(med[1:]) + [rough]


def main(folder, only=None):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
    data = {}  # axis -> body -> (7, features)
    for f in files:
        stem = f[:-4]
        num, rest = stem.split("_", 1)
        axis, body = rest.rsplit("_", 1)
        if only and body not in only:
            continue
        x, sr = load(os.path.join(folder, f))
        calls = split_calls(x, sr)
        if len(calls) != 7:
            print(f"{f}: found {len(calls)} calls, not 7; skipped")
            continue
        data.setdefault((int(num), axis), {})[body] = np.array([features(c, sr) for c in calls])
    # Roughly the smallest change a listener notices in each feature: effects are in these units.
    spread = np.array([JND[f] for f in FEATURES])
    steps = np.arange(-3, 4)
    axes = sorted(data)
    effect = np.zeros((len(axes), len(FEATURES)))
    for i, key in enumerate(axes):
        slopes = []
        for body, m in data[key].items():
            row = []
            for j in range(len(FEATURES)):
                y = m[:, j]
                ok = np.isfinite(y)
                row.append(np.polyfit(steps[ok], y[ok], 1)[0] / spread[j] if ok.sum() >= 4 else np.nan)
            slopes.append(row)
        effect[i] = np.nanmean(slopes, axis=0)
    names = [a for _, a in axes]
    w = max(len(n) for n in names)
    print("Effect per step, in just-noticeable differences (averaged over bodies)\n")
    print(" " * w + "".join(f"{f:>13s}" for f in FEATURES))
    for n, row in zip(names, effect):
        print(f"{n:<{w}s}" + "".join(f"{v:13.2f}" if abs(v) >= 0.3 else f"{'.':>13s}" for v in row))
    print("\nOverlap (cosine of effect vectors; above 0.8 = sounds alike)\n")
    e = np.nan_to_num(effect)
    unit = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    cos = unit @ unit.T
    print(" " * w + "".join(f"{k + 1:>6d}" for k in range(len(names))))
    for k, (n, row) in enumerate(zip(names, cos)):
        print(f"{n:<{w}s}" + "".join(f"{v:6.2f}" if j != k else f"{'-':>6s}" for j, v in enumerate(row)))
    print("\nStrength (largest effect per step):")
    for n, row in zip(names, e):
        flag = "   <- weak" if np.max(np.abs(row)) < 0.5 else ""
        print(f"  {n:<{w}s} {np.max(np.abs(row)):.2f}{flag}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "sweeps")
    # Optional: only these bodies, e.g. "frog" or "mouse,fox,wolf,bear"
    main(folder, set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None)
