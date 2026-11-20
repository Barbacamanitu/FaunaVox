"""Fit the preference model to the ratings and report what it learned. See GENOME_VOICE_DESIGN.md §3.

    python ratings/fit.py                 # fit, report, and write ratings/field_v1.json
    python ratings/fit.py --no-export     # report only

The model is a Gaussian process over:
- the 10 voice-space axes;
- log10 of mass;
- the source (larynx, syrinx, pulse, hiss) and the intent, as one-hot inputs.

It uses one RBF kernel with a length scale per input (short = the score changes quickly along it, so it
matters), plus learned noise. A wrong-size tick adds a half-weight pseudo-rating at the same position
on a body 6x bigger or smaller, scored max(score, 0.4).

The report covers:
- how consistent the ratings are, from the hidden repeats;
- how well the model predicts ratings it hasn't seen (10-fold cross-validation);
- which inputs matter (length scales);
- the score along each axis with everything else averaged over the rated samples;
- the regions predicted to be confidently disliked or liked, from 20,000 random positions.

Needs numpy and scikit-learn.
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ["resonance_scale", "pitch_height", "pitch_liveliness", "breathy", "roughness", "instability",
        "brightness", "ringiness", "nasality", "body_resonance"]
SOURCES = ["larynx", "syrinx", "pulse", "hiss"]
INTENTS = ["idle", "mating", "warn", "alarm", "flee", "pain", "beg", "territorial", "speak"]
FIELD_VERSION = 1
# Badness: only confidently disliked regions repel (§3). Scores above -THETA count as fine; KAPPA
# standard deviations of doubt cancel badness; BETA sets how softly it switches on.
THETA, KAPPA, BETA = 0.2, 1.0, 8.0
SIZE_FACTOR, SIZE_FLOOR = 6.0, 0.4
# Body kinds and mass ranges, as in space.rs BODY_KINDS (for sampling the space).
KINDS = [("larynx", 0.55, 0.05, 800.0), ("syrinx", 0.2, 0.02, 20.0), ("pulse", 0.15, 0.001, 10.0), ("hiss", 0.1, 0.3, 100.0)]


def load():
    manifest = {}
    for path in sorted(glob.glob(os.path.join(HERE, "batches", "*", "manifest.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    manifest[s["id"]] = s
    ratings = {}
    with open(os.path.join(HERE, "ratings.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ratings[r["id"]] = r
    return manifest, ratings


def features(axes, mass, source, intent, intents):
    return np.concatenate([np.asarray(axes, float), [np.log10(mass)],
                           [1.0 if source == s else 0.0 for s in SOURCES],
                           [1.0 if intent == i else 0.0 for i in intents]])


def names(intents):
    return AXES + ["log_mass"] + [f"source={s}" for s in SOURCES] + [f"intent={i}" for i in intents]


def dataset(manifest, ratings, intents):
    X, y, extra_noise, ids = [], [], [], []
    for rid, r in ratings.items():
        s = manifest.get(rid)
        if s is None or r.get("skip"):
            continue
        b = s["body"]
        X.append(features(s["axes"], b["mass"], b["source"], s["intent"], intents))
        y.append(r["score"])
        extra_noise.append(0.0)
        ids.append(rid)
        if r.get("size") in ("bigger", "smaller"):
            mass = b["mass"] * (SIZE_FACTOR if r["size"] == "bigger" else 1.0 / SIZE_FACTOR)
            X.append(features(s["axes"], mass, b["source"], s["intent"], intents))
            y.append(max(r["score"], SIZE_FLOOR))
            extra_noise.append(1.0)  # half weight: its noise is doubled once the base noise is known
            ids.append(rid + "+size")
    return np.array(X), np.array(y), np.array(extra_noise), ids


def make_gp(n_features, noise_var=None):
    kernel = (ConstantKernel(0.2, (1e-3, 10.0))
              * RBF(length_scale=np.full(n_features, 2.0), length_scale_bounds=(0.1, 100.0))
              + WhiteKernel(0.05, (1e-4, 1.0)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


def fit(X, y, extra):
    gp = make_gp(X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    # Pseudo-ratings get extra noise equal to the learned noise (half weight): refit with it fixed.
    if extra.any():
        noise = gp.kernel_.k2.noise_level
        gp2 = GaussianProcessRegressor(kernel=gp.kernel_, normalize_y=True, optimizer=None,
                                       alpha=extra * noise / np.var(y))
        gp2.fit(X, y)
        return gp2
    return gp


def consistency(manifest, ratings):
    diffs = []
    for rid, s in manifest.items():
        src = s.get("repeat_of")
        if src and rid in ratings and src in ratings and not ratings[rid]["skip"] and not ratings[src]["skip"]:
            diffs.append(ratings[rid]["score"] - ratings[src]["score"])
    return np.array(diffs)


def sample_space(n, rng, intents):
    """Random positions from the sampling distribution, on random bodies, as in space.rs."""
    shares = np.array([k[1] for k in KINDS])
    kind = rng.choice(len(KINDS), n, p=shares / shares.sum())
    lo = np.log(np.array([KINDS[k][2] for k in kind]))
    hi = np.log(np.array([KINDS[k][3] for k in kind]))
    mass = np.exp(lo + rng.random(n) * (hi - lo))
    axes = rng.standard_t(4, size=(n, 10))
    intent = rng.choice(intents, n)
    X = np.array([features(axes[i], mass[i], KINDS[kind[i]][0], intent[i], intents) for i in range(n)])
    return X, [KINDS[k][0] for k in kind], intent


def badness(mu, sd):
    z = BETA * (-mu - KAPPA * sd - THETA)
    return np.logaddexp(0.0, z) / BETA


def describe_region(Xr, Xall, src, src_all, intent, intent_all, feat):
    """How a region differs from the space overall: axes whose mean moved most, and over-represented groups."""
    lines = []
    for j, n in enumerate(AXES + ["log_mass"]):
        d = Xr[:, j].mean() - Xall[:, j].mean()
        sd = Xall[:, j].std() + 1e-9
        lines.append((abs(d) / sd, f"{n} {'+' if d > 0 else '-'}{abs(d) / sd:.1f} sd (mean {Xr[:, j].mean():+.2f})"))
    lines.sort(reverse=True)
    out = [l for z, l in lines[:4] if z > 0.3]
    for groups, all_groups, what in ((src, src_all, "source"), (intent, intent_all, "intent")):
        vals, counts = np.unique(groups, return_counts=True)
        for v, c in zip(vals, counts):
            share, base = c / len(groups), np.mean(np.asarray(all_groups) == v)
            if share > 1.5 * base and share > 0.2:
                out.append(f"{what} {v}: {100 * share:.0f}% of the region (vs {100 * base:.0f}% overall)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    manifest, ratings = load()
    intents = [i for i in INTENTS if any(s["intent"] == i for s in manifest.values())]
    feat = names(intents)
    X, y, extra, ids = dataset(manifest, ratings, intents)
    print(f"{int((extra == 0).sum())} ratings (+{int(extra.sum())} wrong-size pseudo-ratings), mean score {y[extra == 0].mean():+.2f}")

    d = consistency(manifest, ratings)
    if len(d):
        print(f"Hidden repeats: {len(d)}, re-rated within {np.mean(np.abs(d)):.2f} on average "
              f"(spread {np.std(d):.2f}); exact matches {np.mean(np.abs(d) < 0.05) * 100:.0f}%")

    # Cross-validation: how well it predicts ratings it hasn't seen.
    real = extra == 0
    pred = np.zeros(real.sum())
    Xr, yr = X[real], y[real]
    for tr, te in KFold(10, shuffle=True, random_state=0).split(Xr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = make_gp(X.shape[1]).fit(Xr[tr], yr[tr])
        pred[te] = g.predict(Xr[te])
    rmse = np.sqrt(np.mean((pred - yr) ** 2))
    base = np.sqrt(np.mean((yr - yr.mean()) ** 2))
    corr = np.corrcoef(pred, yr)[0, 1]
    print(f"Cross-validated: correlation {corr:.2f}, error {rmse:.2f} vs {base:.2f} for always guessing the mean "
          f"({100 * (1 - rmse / base):.0f}% better)")

    gp = fit(X, y, extra)
    k = gp.kernel_
    rbf = k.k1.k2 if hasattr(k, "k1") else k
    ls = np.atleast_1d(rbf.length_scale)
    noise = k.k2.noise_level if hasattr(k, "k2") else float("nan")
    print(f"Noise (normalized): {noise:.2f}; signal variance {k.k1.k1.constant_value:.2f}")
    print("\nWhat matters (length scale: short = the score changes quickly along it; axes span about -3..3):")
    for i in np.argsort(ls):
        tag = "strong" if ls[i] < 2 else "some" if ls[i] < 6 else "little"
        print(f"  {feat[i]:<20s} {ls[i]:7.2f}  {tag}")

    # The score along each axis, everything else averaged over the rated samples.
    steps = np.array([-3, -2, -1, 0, 1, 2, 3])
    print("\nPredicted score along each axis (other inputs as in the rated samples):")
    print(" " * 18 + "".join(f"{s:>7d}" for s in steps))
    for j, n in enumerate(AXES):
        row = []
        for s in steps:
            Xs = Xr.copy()
            Xs[:, j] = s
            row.append(gp.predict(Xs).mean())
        print(f"  {n:<16s}" + "".join(f"{v:+7.2f}" for v in row))

    # Regions, from random positions across the space.
    rng = np.random.default_rng(0)
    Xs, src, intent = sample_space(20000, rng, intents)
    mu, sd = gp.predict(Xs, return_std=True)
    bad = badness(mu, sd)
    disliked = (mu + KAPPA * sd < -THETA)
    liked = (mu - KAPPA * sd > 0.7)
    print(f"\nAcross 20,000 random positions: {100 * disliked.mean():.1f}% confidently disliked "
          f"(score + 1 sd under {-THETA}), {100 * liked.mean():.1f}% confidently liked (score - 1 sd over 0.7); "
          f"badness > 0 at {100 * (bad > 0.01).mean():.1f}%")
    for mask, label in ((disliked, "Disliked region"), (liked, "Liked region")):
        if mask.sum() >= 20:
            print(f"\n{label} ({mask.sum()} positions), compared with the space overall:")
            for line in describe_region(Xs[mask], Xs, np.array(src)[mask], src, np.array(intent)[mask], intent, feat):
                print("  " + line)

    print("\nYour lowest-rated samples:")
    order = np.argsort(yr)
    rid = [i for i, e in zip(ids, extra) if e == 0]
    for i in order[:8]:
        s = manifest[rid[i]]
        top = sorted(zip(AXES, s["axes"]), key=lambda t: -abs(t[1]))[:3]
        print(f"  {rid[i]} {yr[i]:+.1f}  {s['body']['source']:6s} {s['body']['mass']:8.3f} kg {s['intent']:6s} "
              + ", ".join(f"{n} {v:+.1f}" for n, v in top))

    if not a.no_export:
        out = os.path.join(HERE, f"field_v{FIELD_VERSION}.json")
        L = gp.L_
        field = {
            "field_version": FIELD_VERSION,
            "space_version": sorted({s["space_version"] for s in manifest.values()}),
            "features": feat,
            "length_scale": ls.tolist(),
            "signal_variance": float(k.k1.k1.constant_value),
            "noise": float(noise),
            "y_mean": float(gp._y_train_mean), "y_std": float(gp._y_train_std),
            "X": gp.X_train_.tolist(),
            "alpha": np.ravel(gp.alpha_).tolist(),
            "L": L.tolist(),
            "badness": {"theta": THETA, "kappa": KAPPA, "beta": BETA},
            "ratings": int(real.sum()),
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(field, f)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
