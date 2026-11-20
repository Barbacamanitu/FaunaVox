"""Random creatures saying random phrases, joined into one WAV.

Each creature gets a random body (size, source, roughness, breathiness...), a random voice (pitch,
harshness, nonlinear phenomena) and a random speak phrase: syllables with vowels or glides,
consonants before and after them, and the occasional pause. Rendered with faunavox.dll.

    python random_chorus.py [--count 25] [--seed 7] [--out auditions/random/random_chorus.wav]

Also writes <out>.txt listing each creature's phrase and body, in the order they're heard.
"""
import argparse
import ctypes
import json
import os
import random
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "dist", "faunavox.dll")

VOWELS = ["a", "e", "i", "o", "u", "@", "iy", "ih", "eh", "ae", "ah", "aa", "oh", "ao", "uh", "uw", "er"]
CONSONANTS = ["h", "s", "sh", "f", "th", "ch", "j"]


def random_phrase(r):
    words = []
    for k in range(r.randint(3, 8)):
        syl = []
        if r.random() < 0.45:
            syl.append(r.choice(CONSONANTS))
        vowels = r.sample(VOWELS, 1 if r.random() < 0.7 else r.randint(2, 3))
        syl.append("~".join(vowels))
        if r.random() < 0.15:
            syl.append(r.choice(CONSONANTS))
        words.append(" ".join(syl))
        if k > 0 and r.random() < 0.15:
            words.append("|")
    while words[-1] == "|":
        words.pop()
    return " ".join(words)


def random_creature(r, name):
    mass = 10 ** r.uniform(-0.5, 2.6)  # 0.3 .. 400 kg
    source = r.choices(["larynx", "syrinx", "pulse"], weights=[0.7, 0.2, 0.1])[0]
    body = {
        "mass": round(mass, 2),
        "source": source,
        "roughness": round(r.uniform(0.0, 0.7), 2),
        "breathiness": round(r.uniform(0.0, 0.45), 2),
        "nasality": round(r.uniform(0.0, 0.4), 2),
        "open_quotient": round(r.uniform(0.4, 0.65), 2),
        "formant_gain": round(r.uniform(4.0, 12.0), 1),
    }
    voice = {
        "vowels": "a",
        "pitch": round(r.uniform(-4.0, 6.0), 1),
        "harshness": round(r.uniform(0.0, 0.4), 2),
        "vibrato": round(r.uniform(0.0, 0.4), 2),
        "tempo": round(r.uniform(0.8, 1.4), 2),
        "subharmonics": round(r.uniform(0.0, 0.4), 2),
        "chaos": round(r.uniform(0.0, 0.3), 2),
        "biphonation": round(r.uniform(0.0, 0.3), 2) if source == "larynx" else 0.0,
        "jumps": round(r.uniform(0.0, 0.2), 2),
        "calls": {"speak": {"vowels": random_phrase(r), "repeats": [1, 1]}},
    }
    return {"name": name, "body": body, "voice": voice}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gap", type=float, default=0.5, help="silence between creatures (s)")
    ap.add_argument("--distance", type=float, default=15.0)
    ap.add_argument("--reverb", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "auditions", "random", "random_chorus.wav"))
    args = ap.parse_args()

    d = ctypes.CDLL(os.path.abspath(DLL))
    d.cv_creature_from_json.restype = ctypes.c_void_p
    d.cv_creature_from_json.argtypes = [ctypes.c_char_p]
    d.cv_creature_free.argtypes = [ctypes.c_void_p]
    d.cv_render.restype = ctypes.c_void_p
    d.cv_render.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_float, ctypes.c_float]
    d.cv_buffer_len.restype = ctypes.c_int32
    d.cv_buffer_len.argtypes = [ctypes.c_void_p]
    d.cv_buffer_data.restype = ctypes.POINTER(ctypes.c_float)
    d.cv_buffer_data.argtypes = [ctypes.c_void_p]
    d.cv_buffer_free.argtypes = [ctypes.c_void_p]
    d.cv_last_error.restype = ctypes.c_char_p
    sr = d.cv_sample_rate()

    r = random.Random(args.seed)
    silence = np.zeros(int(args.gap * sr), dtype=np.float32)
    parts, log, t = [], [], 0.0
    i = 0
    while len(log) < args.count:
        i += 1
        spec = random_creature(r, f"creature_{len(log) + 1:02d}")
        c = d.cv_creature_from_json(json.dumps(spec).encode())
        if not c:
            print("skipped:", d.cv_last_error().decode())
            continue
        buf = d.cv_render(c, b"speak", r.randrange(1, 2**31), args.distance, args.reverb)
        d.cv_creature_free(c)
        if not buf:
            print("skipped:", d.cv_last_error().decode())
            continue
        n = d.cv_buffer_len(buf)
        audio = np.ctypeslib.as_array(d.cv_buffer_data(buf), shape=(n,)).copy()
        d.cv_buffer_free(buf)
        b = spec["body"]
        line = (f"{t:6.2f}s  {spec['name']}  {b['source']:6s} {b['mass']:7.2f} kg  "
                f"pitch {spec['voice']['pitch']:+.1f}  says: {spec['voice']['calls']['speak']['vowels']}")
        print(line)
        log.append(line)
        parts += [audio, silence]
        t += (n + len(silence)) / sr

    mix = np.concatenate(parts[:-1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    with open(os.path.splitext(args.out)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print(f"wrote {os.path.normpath(args.out)} ({len(mix) / sr:.1f} s, {len(log)} creatures)")


if __name__ == "__main__":
    main()
