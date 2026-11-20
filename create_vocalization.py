"""General-purpose procedural animal vocalizations (source-filter model).

Three inputs decide a sound:
    Body   - physical traits, normally taken from the creature's actual size
             (mass, vocal tract length, sound source type, roughness, tract stiffness, ...)
    Voice  - heritable traits decoded from the genome
             (vowel list, pitch offset, pitch range, harshness, vibrato, tempo, call length,
             falsetto, repetition, and per-intent overrides for a species' repertoire)
    intent - what the animal is trying to say: idle, warn, alarm, flee, mating,
             pain, beg, territorial

The intents follow Morton's motivation-structural rules: hostile calls are low and
harsh, fearful or appeasing calls are high and tonal. Voice.calls can reshape any intent
for a species (a pig's idle call is a grunt two octaves below its squeal) without
changing the model. The presets were tuned against real recordings of each animal.

Sound sources:
    larynx  mammals: glottal pulses with jitter, shimmer, subharmonics, chaos, fry and falsetto
    syrinx  birds: two tonal voices for song; `buzz` or harshness switches to pulsed
            vibration (caws, quacks), since the syrinx labia work like vocal folds
    pulse   frogs and insects: pulse trains ringing a vocal sac resonance and its octave
    hiss    reptiles: turbulent noise only

Usage:
    python create_vocalization.py --preset lion --intent warn
    python create_vocalization.py --preset wolf --demo             # every intent to ./vocalizations
    python create_vocalization.py --all-presets --demo
    python create_vocalization.py --preset cat --mass 12 --vowels i,a,u --intent mating
    python create_vocalization.py --json creature.json --intent alarm

JSON format: {"body": {"mass": 40, "tract_length": 0.2, ...}, "voice": {"vowels": ["u", "o"], ...}}
A vowel is a name (a e i o u @) or a list of F1-F3 ratios relative to a neutral tube.
"""
import argparse
import json
import os
import wave
from dataclasses import dataclass, field, fields, replace

import numpy as np
from numba import njit
from scipy.signal import butter, fftconvolve, iirnotch, lfilter, sosfilt

SR = 44100
C_SOUND = 350.0  # speed of sound in warm, humid air (m/s)
# Breath noise RMS relative to the voice: floor + scale * breathiness * (0.5 + mean harshness).
# Default body, clean call: about -24 dB; breathy, harsh growl: about -16 dB.
BREATH_FLOOR, BREATH_SCALE = 0.02, 0.26
GROWL_NOISE = 0.6  # extra noise (relative to voice) at full harshness; NLP now carry most of the roughness
NOISE_CORNER_MAX = 2000.0  # turbulence noise rolls off above this even in small tracts (before harshness)
INHALE_LEVEL = 0.08  # inhale RMS relative to the loudest call (about -22 dB)
# Clean-voice irregularity, at healthy-voice levels; harshness adds chaos on top.
BASE_JITTER = 0.004    # cycle-to-cycle period variation
BASE_SHIMMER = 0.03    # cycle-to-cycle amplitude variation
DRIFT_DEPTH = 0.012    # slow pitch wander (fraction of f0)
DRIFT_TIME = 0.15      # seconds the wander takes to change
SHAPE_JITTER = 0.015   # cycle-to-cycle variation of the glottal pulse shape
CHAOS_JITTER = 0.15    # extra period jitter at full harshness (chaotic phonation)
FRY_JITTER, FRY_SHIMMER = 0.25, 0.4  # extra irregularity deep in the fry register
GLOTTAL_DAMPING = 0.5     # extra F1 bandwidth, as a fraction of f0, when f0 sits on F1
TRACT_LOSS_LOWPASS = 0.5  # loop lowpass in the tube: higher = broader, weaker upper formants
CHEST_GAIN = 0.02  # radiation strips the lows; the chest/body resonance puts weight back
# Formant resonance on top of the tube. Real calls show formant peaks about twice as prominent
# as the bare tube gives, and a flat envelope with any noise in it reads as breathy.
FORMANT_GAIN = 8.0
FORMANT_LEVELS = (1.0, 1.0, 0.9, 0.7, 0.5)  # relative boost of F1..F5
# Breath noise mostly happens while the glottis is open. Pulsed with each cycle it fuses into a
# rough voice; a steady floor between pulses is heard as a separate hiss layer.
BREATH_PULSE = 0.9
# Fraction of each glottal cycle the folds are open (clean voice). High = smooth, sine-like pulses
# with a dominant fundamental (soft, breathy); low = sharp pulses rich in upper harmonics (pressed).
OPEN_QUOTIENT = 0.5
# Nonlinear phenomena (NLP): subharmonics, deterministic chaos, biphonation and frequency jumps.
# Real voices hold a steady regime and switch abruptly between regimes in episodes; random
# cycle-to-cycle wobble instead reads as noise and breath.
NLP_RATE = 2.5              # regime onsets per second at full trait and full loudness (more = stuttering)
NLP_EPISODE = (0.15, 0.6)   # mean episode length (s) at low .. full trait
SUB_DEPTH = 0.6             # strength drop of the weak cycles in a subharmonic episode
SUB_TIMING = 0.1            # period alternation in a subharmonic episode
CHAOS_PERIOD = 0.12         # period swing driven by the chaotic map at full chaos
CHAOS_AMP = 0.6             # amplitude swing driven by the chaotic map at full chaos
BIPHONATION_COUPLING = 0.3  # how strongly the second source modulates the first (sidebands); high = ring-mod robot

# F1-F3 of each vowel relative to a uniform tube's formants (adult human averages). Scaling them
# by the creature's own tube formants keeps vowels recognisable while tract length still sets size.
VOWELS = {
    "a": (1.46, 0.73, 0.98),
    "e": (1.06, 1.23, 0.99),
    "i": (0.54, 1.53, 1.20),
    "o": (1.14, 0.56, 0.96),
    "u": (0.60, 0.58, 0.90),
    "@": (1.00, 1.00, 1.00),  # schwa, neutral tube
    # The eleven SSBE (Standard Southern British English) monophthongs: mean F1/F2 of three L1
    # speakers, divided by the neutral tube's 500/1500 Hz. F3 is not in that table; it is estimated
    # from typical SSBE values (about 2.9 kHz for front close vowels down to 2.25 kHz for rounded back).
    "i:": (296 / 500, 2241 / 1500, 1.16),   # fleece
    "ɪ": (396 / 500, 1839 / 1500, 1.04),    # kit
    "ɛ": (532 / 500, 1656 / 1500, 1.02),    # dress (the table's "e")
    "æ": (667 / 500, 1565 / 1500, 1.00),    # trap
    "ʌ": (661 / 500, 1296 / 1500, 0.98),    # strut
    "ɑ:": (680 / 500, 1193 / 1500, 0.98),   # palm / start
    "ɒ": (643 / 500, 1019 / 1500, 0.96),    # lot
    "ɔ:": (480 / 500, 857 / 1500, 0.96),    # thought
    "ʊ": (395 / 500, 1408 / 1500, 0.92),    # foot
    "u:": (386 / 500, 1587 / 1500, 0.90),   # goose (fronted, as in modern SSBE)
    "ɜ:": (519 / 500, 1408 / 1500, 0.98),   # nurse (non-rhotic)
}
# ASCII names for the SSBE vowels (ARPAbet where it has one; "oh" for the British lot vowel).
VOWEL_ALIASES = {"iy": "i:", "ih": "ɪ", "eh": "ɛ", "ae": "æ", "ah": "ʌ", "aa": "ɑ:", "oh": "ɒ",
                 "ao": "ɔ:", "uh": "ʊ", "uw": "u:", "er": "ɜ:"}
# Lip rounding per vowel (0 = spread, 1 = fully rounded). A custom vowel may give it as a 4th value.
VOWEL_ROUNDING = {"u": 1.0, "o": 0.6, "@": 0.2,
                  "ɒ": 0.5, "ɔ:": 0.8, "ʊ": 0.7, "u:": 0.8, "ɜ:": 0.1}
LIP_CUTOFF_HZ = (1700.0, 9000.0)  # lip lowpass for a human-size tract: closed/rounded .. wide open


@dataclass
class Body:
    """Physical traits, normally taken from the creature's actual size."""
    mass: float = 50.0             # kg
    tract_length: float = None     # larynx to lips (m); defaults from mass
    source: str = "larynx"         # larynx | syrinx | pulse | hiss
    fold_scale: float = 1.0        # > 1 = longer, heavier vocal folds -> lower pitch
    larynx_drop: float = 0.15      # how far the larynx can pull down, as a fraction of tract length
    roughness: float = 0.3         # tissue irregularity: baseline harshness
    breathiness: float = 0.25
    nasality: float = 0.1
    lung_capacity: float = 1.0     # scales call length
    resonator_freq: float = None   # vocal sac / body resonance for pulse sources (Hz)
    resonator_harmonic: float = 0.0  # level of a second band at the resonance's octave (pulse sources)
    buzz: float = 0.0              # syrinx only: 0 = tonal song, 1 = pulsed buzz (quacks, caws) even when not harsh
    chest: float = 1.0             # chest/body resonance strength (big cats' thump; low for pigs)
    chest_width: float = 70.0      # chest resonance bandwidth (Hz); wide = weight without a ringing beat against the voice
    noise_color: float = 1.0       # turbulence brightness: > 1 for snorts through narrow nostrils
    tract_damping: float = None    # formant bandwidth multiplier; default 4 for syrinx (birds barely filter), 1 otherwise
    hf_loss: float = 1.0           # tract wall losses on upper formants: < 1 for stiff, bright tracts (pig snout)
    low_cut: float = 0.0           # Hz below which a small mouth/nostril radiates poorly; 0 = off
    formant_gain: float = FORMANT_GAIN  # dB of extra resonance at each formant (tone and character)
    formant_width: float = 1.0     # formant bandwidth multiplier for that resonance: < 1 = sharper, ringier
    breath_pulse: float = BREATH_PULSE  # 0..1: how much breath noise pulses with each glottal cycle
    open_quotient: float = OPEN_QUOTIENT  # fraction of each glottal cycle the folds are open: high = soft and breathy, low = pressed and bright

    @classmethod
    def from_dimensions(cls, mass, head_length, neck_length, **kw):
        """Vocal tract runs through the neck and most of the head."""
        return cls(mass=mass, tract_length=0.8 * head_length + 0.5 * neck_length, **kw)

    @property
    def tract(self):
        return self.tract_length or 0.06 * self.mass ** (1.0 / 3.0)

    @property
    def base_f0(self):
        """Natural pitch. Loosely allometric: bigger animals have bigger, slower folds."""
        if self.source == "syrinx":
            return 2500.0 * self.mass ** -0.2 / self.fold_scale
        if self.source == "pulse":
            return 40.0 / self.fold_scale  # pulse repetition rate
        return 600.0 * self.mass ** -0.3 / self.fold_scale

    @property
    def carrier(self):
        return self.resonator_freq or 400.0 * self.mass ** -0.3

    @property
    def call_scale(self):
        """Bigger lungs sustain longer calls."""
        return (self.mass / 50.0) ** 0.12 * self.lung_capacity


@dataclass
class Voice:
    """Heritable vocal traits, normally decoded from the genome."""
    vowels: list = field(default_factory=lambda: ["a"])
    pitch: float = 0.0         # semitones on top of the body's natural pitch
    pitch_range: float = 1.0   # scales how far pitch contours swing
    harshness: float = 0.0     # added to body roughness
    vibrato: float = 0.0       # semitones of vibrato on every call
    tempo: float = 1.0         # > 1 = shorter, quicker calls
    sustain: float = 1.0       # call length multiplier (howlers hold calls for seconds)
    melody: float = 1.0        # 1 = intent's full up-down contour, 0 = one smooth swell
    falsetto: float = 0.0      # 0..1: high notes flip to a thin, near-sinusoidal register (howls)
    repetition: float = 1.0    # scales how many calls an intent repeats (frogs, ducks, crows call in series)
    # Nonlinear phenomena, 0..1 each: how prone this voice is to them. Added to the intent's own amounts.
    subharmonics: float = 0.0  # episodes of period doubling/tripling: lower, rougher, bigger-sounding
    chaos: float = 0.0         # episodes of deterministic chaos: irregular but voiced (screams, snarls)
    biphonation: float = 0.0   # a second, independent tone over the voice, with sidebands (canids, deer)
    biphonation_ratio: float = 1.45  # its pitch relative to the voice's (not a whole number: independent)
    jumps: float = 0.0         # abrupt register breaks: the pitch leaps up or down and back (yelps, meows)
    # Species repertoire: per-intent overrides of Intent fields, e.g. {"idle": {"pitch": -18}} for a
    # pig whose contact grunt sits far below its squeal. Same model, different parameters.
    calls: dict = field(default_factory=dict)


def shape_contour(contour, melody):
    """Blend an intent's pitch contour toward a single swell: rise to its peak, hold with a gentle
    sag, fall to its end. A melodic up-down contour on a pure tone reads as an instrument."""
    u = np.linspace(0.0, 1.0, 17)
    full = curve(contour, u)
    first, last, peak = full[0], full[-1], full.max()
    swell = curve([(0, first), (0.12, peak), (0.85, peak - 1.0), (1, last)], u)
    return list(zip(u, melody * full + (1.0 - melody) * swell))


@dataclass
class Intent:
    """Call template for one behaviour. Curves are (u, value) over normalized call time u."""
    pitch: float           # semitone offset for the whole call
    contour: list          # (u, semitones) pitch shape, scaled by Voice.pitch_range
    amp: list
    harsh: float           # extra harshness, applied in proportion to loudness
    dur: float             # seconds for a 50 kg animal
    repeats: tuple = (1, 1)
    gap: float = 0.3       # silence between repeats (s)
    mouth: list = ((0, 0.3), (0.3, 0.8), (1, 0.4))
    larynx: float = 0.0    # larynx lowering at full loudness (sounds bigger)
    breath: float = 0.0
    vibrato: float = 0.0
    fry: float = 0.5       # creaky, pitch-dropping tail
    nasal: float = 0.0
    inhale: bool = False   # audible breath before each call
    glide: bool = False    # glide through the whole vowel list in one call, else one vowel per call
    # Sequential: one syllable per vowel in the list, in order ("~" joins vowels into one gliding
    # syllable, "|" adds a pause), and repeats says how many times the whole phrase is said.
    sequence: bool = False
    spread: float = 0.0    # lips pulled back (a snarl), cancelling the vowel's rounding
    vowels: list = None    # vowels for this call instead of the voice's own list
    vibrato_rate: float = 5.5  # Hz; screams waver faster than songs
    rumble: float = 0.0    # irregular 20-50 Hz amplitude roughness (gravelly grunts); not periodic, so no propeller
    # Nonlinear phenomena this behaviour brings out, added to the voice's own (see Voice). High
    # arousal (threat, fear, pain) drives the folds into subharmonics, chaos and register breaks.
    subharmonics: float = 0.0
    chaos: float = 0.0
    biphonation: float = 0.0
    jumps: float = 0.0
    voiced: float = 1.0    # 0 = unvoiced (a hiss) even from a voiced body
    drift: float = 0.0     # pitch change per repeat, semitones
    accel: float = 1.0     # gap multiplier per repeat (below 1 speeds up)
    # Spacing over the series: (position 0..1 from first repeat to last, multiplier of the gap). Notes
    # shorten with the square root of it, the way a trill's notes do. Empty = steady.
    rhythm: list = field(default_factory=list)


INTENTS = {
    # Soft contact call.
    "idle": Intent(pitch=0, contour=[(0, -1), (0.4, 1.5), (1, -2)],
                   amp=[(0, 0), (0.15, 0.6), (0.7, 0.5), (1, 0)], harsh=0.0, dur=0.5,
                   repeats=(1, 2), gap=0.35, mouth=[(0, 0.2), (0.4, 0.6), (1, 0.3)], nasal=0.2, fry=0.3,
                   subharmonics=0.1),
    # Threat growl: low, harsh, mouth barely open, larynx down to sound bigger.
    "warn": Intent(pitch=-6, contour=[(0, -1), (0.3, 1), (0.8, 0.5), (1, -2)],
                   amp=[(0, 0), (0.2, 0.8), (0.8, 0.9), (1, 0)], harsh=0.8, dur=1.5,
                   repeats=(1, 2), gap=0.4, mouth=[(0, 0.3), (0.5, 0.5), (1, 0.35)],
                   larynx=1.0, breath=0.1, inhale=True, fry=0.4, spread=1.0,
                   subharmonics=0.5, chaos=0.3),
    # Alarm: short, sharp, high, repeated barks.
    "alarm": Intent(pitch=7, contour=[(0, -3), (0.2, 3), (1, -4)],
                    amp=[(0, 0), (0.05, 1), (0.4, 0.7), (1, 0)], harsh=0.35, dur=0.18,
                    repeats=(3, 5), gap=0.18, mouth=[(0, 0.5), (0.2, 1), (1, 0.5)], fry=0.2,
                    subharmonics=0.3, jumps=0.2),
    # Distress scream while fleeing: very high and chaotic.
    "flee": Intent(pitch=11, contour=[(0, 0), (0.15, 3), (0.4, 1), (0.6, 2.5), (1, -5)],
                   amp=[(0, 0), (0.08, 1), (0.7, 0.85), (1, 0)], harsh=0.7, dur=0.8,
                   repeats=(1, 3), gap=0.25, mouth=[(0, 0.6), (0.15, 1), (1, 0.6)],
                   breath=0.2, inhale=True, fry=0.3, subharmonics=0.4, chaos=0.5, jumps=0.4),
    # Advertisement call: long, melodic, tonal, gliding through the voice's vowels.
    "mating": Intent(pitch=2, contour=[(0, -2), (0.25, 3), (0.5, 1), (0.75, 4), (1, -3)],
                     amp=[(0, 0), (0.2, 0.8), (0.8, 0.85), (1, 0)], harsh=0.1, dur=1.6,
                     repeats=(2, 3), gap=0.5, mouth=[(0, 0.3), (0.3, 0.9), (0.7, 0.7), (1, 0.3)],
                     vibrato=0.4, inhale=True, glide=True, fry=0.4, subharmonics=0.1),
    # Yelp: instant onset, high, steep fall.
    "pain": Intent(pitch=11, contour=[(0, -2), (0.1, 4), (1, -9)],
                   amp=[(0, 0), (0.03, 1), (0.3, 0.7), (1, 0)], harsh=0.45, dur=0.35,
                   mouth=[(0, 0.6), (0.1, 1), (1, 0.3)], fry=0.6, subharmonics=0.3, chaos=0.3, jumps=0.5),
    # Submissive whine: high, rising, nasal, clean.
    "beg": Intent(pitch=8, contour=[(0, -2), (0.6, 3), (1, 4)],
                  amp=[(0, 0), (0.3, 0.5), (0.85, 0.55), (1, 0)], harsh=-0.3, dur=0.8,
                  repeats=(1, 3), gap=0.3, mouth=[(0, 0.1), (1, 0.3)], nasal=0.5,
                  vibrato=0.3, glide=True, fry=0.1, jumps=0.2),
    # Vocalizing: every vowel in the list in turn, one syllable each ("~" glides, "|" pauses),
    # with a small rise-fall per syllable and the pitch sinking over the phrase.
    "speak": Intent(pitch=3, contour=[(0, -1), (0.3, 1), (1, -1.5)],
                    amp=[(0, 0), (0.12, 1), (0.75, 0.85), (1, 0)], harsh=0.0, dur=0.3,
                    repeats=(1, 1), gap=0.08, mouth=[(0, 0.3), (0.25, 0.95), (0.8, 0.85), (1, 0.4)],
                    vibrato=0.1, fry=0.3, drift=-0.3, sequence=True),
    # "territorial" is a full roar bout, built by territorial_bout().
}
INTENT_NAMES = [n for n in INTENTS if n != "speak"] + ["territorial", "speak"]
# How far sequential syllables' loudness is evened out (0..1): rounded vowels radiate much less.
SEQUENCE_EVEN = 0.7


@dataclass
class Call:
    start: float
    dur: float
    pitch: list            # (u, semitones relative to the body's base f0)
    amp: list
    harsh: list
    mouth: list
    vowels: list
    larynx: list = None
    voiced: float = 1.0
    breath: float = 0.0
    fry: float = 0.5
    vibrato: float = 0.0
    nasal: float = 0.0
    falsetto: float = 0.0
    spread: float = 0.0
    vibrato_rate: float = 5.5
    rumble: float = 0.0
    subharmonics: float = 0.0
    chaos: float = 0.0
    biphonation: float = 0.0
    biphonation_ratio: float = 1.45
    jumps: float = 0.0
    # Sequential syllables: how far their loudness is evened out after rendering (0 = as rendered,
    # 1 = all equal), so quiet vowels (rounded u, o) aren't lost; level is the loudness the
    # syllable is meant to have relative to the others.
    even: float = 0.0
    level: float = 1.0
    consonant: str = None  # a consonant in a sequential call: rendered as noise
    consonant_db: float = 0.0  # its level relative to the loudest syllable
    # A syllable next to a consonant: F1 to F3 move from the consonant's locus (ratios, and the
    # vowel's pull on each) to the vowel's own over the first (from) or last (to) fraction of the call.
    formants_from: tuple = None  # (loci, pull, fraction of the call)
    formants_to: tuple = None
    voice_bar: bool = False  # the voicing heard through a voiced stop's closure


def nlp_levels(voice, it=None, sub=0.0, chaos=0.0, biph=0.0, jumps=0.0):
    """A call's nonlinear-phenomena amounts: the voice's proneness plus the behaviour's, 0..1."""
    if it is not None:
        sub, chaos, biph, jumps = it.subharmonics, it.chaos, it.biphonation, it.jumps
    c = lambda v: float(np.clip(v, 0.0, 1.0))
    return dict(subharmonics=c(voice.subharmonics + sub), chaos=c(voice.chaos + chaos),
                biphonation=c(voice.biphonation + biph), jumps=c(voice.jumps + jumps),
                biphonation_ratio=voice.biphonation_ratio)


def rms(x):
    return np.sqrt(np.mean(np.square(x))) + 1e-12


def curve(points, u):
    xs, ys = zip(*points)
    return np.interp(u, xs, ys)


def vowel_name(v):
    name = VOWEL_ALIASES.get(v, v)
    if name not in VOWELS:
        raise ValueError(f"unknown vowel {v!r}; choose from {list(VOWELS) + list(VOWEL_ALIASES)} or give [F1, F2, F3] ratios")
    return name


# Vowel-list separators. In a sequential call (the "speak" intent, or any intent with
# "sequence": true) every vowel is its own syllable unless joined: "~" glides from the vowel before
# it into the one after, inside one syllable; "|" is a pause. Any string in the list may hold
# several tokens ("a i~u | e"); spaces and commas separate them. Other calls ignore the separators.
GLIDE, PAUSE = "~", "|"
# Consonants for sequential calls (name or IPA alias -> name). h is breath through the tract shaped
# like the vowel next to it; s, sh, f and th are turbulence at the teeth or lips (fricatives); p, t,
# k, b, d and g close the tract and release it with a burst (stops).
CONSONANTS = {"h": "h", "s": "s", "sh": "sh", "ʃ": "sh", "f": "f", "th": "th", "θ": "th",
              "ch": "ch", "tʃ": "ch", "j": "j", "dʒ": "j",
              "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "g"}
# Stops: the lips (p, b), the tongue tip (t, d) or the back of the tongue (k, g) close the tract,
# then release it with a burst. p, t and k are followed by breath before the voice starts; b, d and
# g are voiced through the closure between vowels.
STOPS = ("p", "t", "k", "b", "d", "g")
VOICED_STOPS = ("b", "d", "g")
# Affricates: a short sh that bursts out and dies away ("itch", "judge").
AFFRICATES = ("ch", "j")
# A consonant's length (s, for a 50 kg animal; it grows only with the square root of size) and its
# loudness relative to the loudest syllable (dB, RMS over the consonant). Kept well under measured
# speech power: noise stands out from a creature's voice far more than from speech, and a faint
# consonant is better than a loud one. The order follows Wilde (1995, table 4.1): sh 5-10 dB above s,
# and f and th about equal, far below both. Listeners reject an s even 4 dB too loud but accept f
# anywhere within 4 dB.
CONSONANT_SHAPE = {"h": (0.08, -20.0), "s": (0.13, -27.0), "sh": (0.13, -22.0),
                   "f": (0.11, -29.0), "th": (0.11, -29.0),
                   # affricates: a short burst, quieter than a full sh
                   "ch": (0.07, -24.0), "j": (0.065, -27.0),
                   # Stops: the closure (silence), and the burst's level. The burst is a click of a
                   # few ms, so its RMS can sit near a fricative's and still be heard. k's is the
                   # strongest and longest, p's the weakest; voiced ones about 5 dB weaker (Klatt 1980).
                   "p": (0.065, -27.0), "t": (0.05, -23.0), "k": (0.055, -22.0),
                   "b": (0.055, -31.0), "d": (0.04, -28.0), "g": (0.045, -27.0)}
# A stop's burst length and voice onset time: how long after the release the voice starts, with
# breath through the next vowel's formants in between (s, for a 50 kg animal; grows with the square
# root of size). Klatt (1979, 1980): bursts of 6, 8 and 18 ms for p, t and k; voiceless stops 45-60 ms
# before the voice, voiced ones at once.
STOP_TIMING = {"p": (0.006, 0.04), "t": (0.008, 0.05), "k": (0.016, 0.055),
               "b": (0.005, 0.0), "d": (0.007, 0.0), "g": (0.014, 0.0)}
# The breath after a voiceless stop's burst, before the voice starts (dB re the loudest syllable).
# After a vowel (at the end of a syllable) the release is weaker and shorter: this much quieter, and
# this share of the onset's breath.
STOP_ASPIRATION = -24.0
CODA_RELEASE = (-6.0, 0.4)
# Voicing heard through the closure of b, d or g between vowels: the folds keep vibrating while the
# walls of the closed tract pass only the lowest harmonics. Its level (dB re the loudest syllable;
# Klattara's voice bar is 16 dB under a stressed vowel) and its lowpass (Hz, human size).
VOICE_BAR = (-22.0, 450.0)
# s, sh, ch and j: noise from the airstream hitting the teeth, much stronger than f and th.
STRIDENTS = ("s", "sh", "ch", "j")
# How much weaker a strident is next to an open back vowel ("sa" against "se"), dB at full openness
# and backness (Wilde 1995: s 8-13 dB weaker before /a/, about 3 dB of it the vowel's F1).
STRIDENT_OPEN_BACK = -7.0
# The turbulence source falls off at 6 dB/octave above a corner that rises with the airflow, so a
# quiet fricative is darker as well as softer (Stevens; Wilde 1995, fig. 2.6). The corner (Hz, human
# size) for a fricative at the given level (dB re the loudest syllable), and the dB per octave it
# moves. Gentler than the source alone: lip radiation (+6 dB/octave) undoes most of the source's slope,
# which is why f and th measure nearly flat.
SOURCE_CORNER = (4500.0, -10.0, 16.0)
# A fricative's noise rises and falls in about this long (s, for a 50 kg animal; it grows with the
# square root of size): a flat plateau with short ramps, nothing extra at the edges, which listeners
# hear as unnatural (Wilde 1995). An affricate's release bursts out faster.
FRICATIVE_RAMP = (0.025, 0.012)
# The near-silent moment between a fricative and the vowel it releases into: the noise stops before
# the voice starts (s, for a 50 kg animal; grows with the square root of size).
RELEASE_DIP = 0.012
# How fast a vowel after a fricative starts: the attack of its loudness curve is squeezed by this
# much. The voice switches on abruptly after a voiceless consonant.
ONSET_SHARPEN = 0.2
# Where F1 to F3 point at each fricative's place of articulation (as ratios of the neutral tube's
# 500, 1500 and 2500 Hz, so they scale with size), and how far the vowel pulls each one at the
# boundary: 0 keeps the locus, 1 takes the vowel's own value. F1 drops toward 300 Hz at every
# constriction, the plainest cue that the tract closed in. These loci are wider than natural locus
# equations: synthetic speech needs its place cues exaggerated (Klattara, after DECtalk). Next to a
# front vowel, k and g close further forward and their F2 and F3 meet near 2300 Hz ("key" against
# "coo"): FRONT_VELAR.
_POSTALVEOLAR = ((0.56, 2000.0 / 1500.0, 2650.0 / 2500.0), (0.3, 0.55, 0.55))
_LABIAL = ((0.5, 900.0 / 1500.0, 2100.0 / 2500.0), (0.3, 0.6, 0.6))
_ALVEOLAR = ((0.5, 1.2, 2700.0 / 2500.0), (0.3, 0.5, 0.5))
_VELAR = ((0.5, 2000.0 / 1500.0, 2100.0 / 2500.0), (0.3, 0.5, 0.3))
FRONT_VELAR = ((0.5, 2250.0 / 1500.0, 2350.0 / 2500.0), (0.3, 0.3, 0.3))
LOCUS = {"f": ((0.6, 1100.0 / 1500.0, 2200.0 / 2500.0), (0.35, 0.7, 0.7)),
         "th": ((0.6, 1.0, 2600.0 / 2500.0), (0.35, 0.6, 0.6)),
         "s": _ALVEOLAR,
         "sh": _POSTALVEOLAR, "ch": _POSTALVEOLAR, "j": _POSTALVEOLAR,
         "p": _LABIAL, "b": _LABIAL, "t": _ALVEOLAR, "d": _ALVEOLAR, "k": _VELAR, "g": _VELAR}
# How much of each formant's pull toward the vowel is kept at the boundary. Half leaves the formants
# farther from the vowel, so they bend more sharply into each consonant and mark its place more
# clearly (Klattara: final consonants told apart 95% -> 97% of the time).
LOCUS_PULL = 0.5
# How long the vowel's F2 takes to move from the consonant's locus to its own target (s, for a 50 kg
# animal; grows with the square root of size).
FORMANT_TRANSITION = 0.06
# The breath that joins a fricative to its vowel: as the tongue or lips release, the airstream
# passes through the opening tract and takes on the vowel's formants. Its level against the
# fricative's (dB), and its length as a fraction of the fricative's into a vowel and out of one: the
# glottis closes slowly after a fricative, so the breath runs on into the next vowel, but opens
# quickly before one (Wilde 1995).
ASPIRATION = (-12.0, 0.6, 0.3)
# Breath from the open glottis has a weak F1: the glottis damps it heavily, about 8-10 dB under F2 and
# F3 (Wilde 1995). An h is highpassed at this multiple of its vowel's F1.
ASPIRATION_F1_CUT = 1.5
# How much of a consonant overlaps the vowel it leads into or trails from.
CONSONANT_OVERLAP = 0.25


def vowel_tokens(vowels):
    """A vowel list as vowels (names or ratio lists) and the separators GLIDE and PAUSE. The list
    may also be one string holding the whole phrase ("a i~u | e")."""
    if isinstance(vowels, str):
        vowels = [vowels]
    out = []
    for v in vowels:
        if not isinstance(v, str):
            out.append(v)
            continue
        word = ""
        for ch in v:
            if ch in (GLIDE, PAUSE) or ch.isspace() or ch == ",":
                if word:
                    out.append(word)
                    word = ""
                if ch in (GLIDE, PAUSE):
                    out.append(ch)
            else:
                word += ch
        if word:
            out.append(word)
    return [CONSONANTS.get(t, t) if isinstance(t, str) else t for t in out]


def _separator(t):
    return isinstance(t, str) and (t in (GLIDE, PAUSE) or t in CONSONANT_SHAPE)


def plain_vowels(vowels):
    """The vowels of a list in order, without separators or consonants (checked)."""
    out = [v for v in vowel_tokens(vowels) if not _separator(v)]
    if not out:
        raise ValueError(f"vowel list {vowels!r} must hold at least one vowel")
    for v in out:
        if isinstance(v, str):
            vowel_name(v)
        elif len(v) < 3:
            raise ValueError("a custom vowel needs at least 3 ratios [F1, F2, F3(, rounding)]")
    return out


def syllables(vowels):
    """A vowel list as [(onset, vowels, coda, pause_after)]: every vowel its own syllable, except
    runs joined by ~. Consonants start the next syllable, except those before a pause or at the
    end, which close the one before."""
    plain_vowels(vowels)  # check it
    out, pending, join = [], [], False
    for t in vowel_tokens(vowels):
        glide = isinstance(t, str) and t == GLIDE
        if isinstance(t, str) and t == PAUSE:
            if out:
                out[-1][2] += pending
                pending = []
                out[-1][3] = True
        elif isinstance(t, str) and t in CONSONANT_SHAPE:
            pending.append(t)
        elif not glide:
            if join and out and not out[-1][3] and not pending:
                out[-1][1].append(t)
            else:
                out.append([pending, [t], [], False])
                pending = []
        join = glide
    if out:
        out[-1][2] += pending
    return [tuple(x) for x in out]


def vowel_ratios(v):
    return np.asarray(VOWELS[vowel_name(v)] if isinstance(v, str) else v[:3], dtype=float)


def vowel_rounding(v):
    if isinstance(v, str):
        return VOWEL_ROUNDING.get(vowel_name(v), 0.0)
    return float(v[3]) if len(v) > 3 else 0.0


# ---------------------------------------------------------------- call structure

def inhale_call(start, dur):
    return Call(start=start, dur=dur, pitch=[(0, 0), (1, 0)], amp=[(0, 0), (0.5, 0.18), (1, 0)],
                harsh=[(0, 0.4), (1, 0.4)], mouth=[(0, 0.5), (1, 0.6)], vowels=["@"],
                voiced=0.0, breath=0.5)


def build_calls(body, voice, intent_name, rng):
    # A species override may base an intent on another one ("like"), e.g. a frog's territorial
    # call is its croak chorus rather than a roar bout.
    overrides = dict(voice.calls.get(intent_name, {}))
    base_name = overrides.pop("like", intent_name)
    if base_name == "territorial":
        return territorial_bout(body, voice, rng)
    it = replace(INTENTS[base_name], **overrides)
    if it.sequence:
        return sequence_calls(body, voice, it, rng)
    scale = body.call_scale / voice.tempo
    base_harsh = 0.5 * body.roughness + voice.harshness
    calls, t = [], 0.0
    lo, hi = (max(1, int(round(r * voice.repetition))) for r in it.repeats)
    vowels = plain_vowels(it.vowels or voice.vowels)
    gap = it.gap
    count = rng.integers(lo, hi + 1)
    for i in range(count):
        spacing = rhythm_at(it.rhythm, i, count)
        if it.inhale:
            calls.append(inhale_call(t, 0.25 * scale))
            t += 0.3 * scale
        # Sustain and melody shape only the long tonal calls (howls, songs, whines), not barks.
        tonal = it.glide
        dur = it.dur * scale * (voice.sustain if tonal else 1.0) * (1.0 + 0.1 * rng.standard_normal()) \
            * np.sqrt(spacing)
        contour = shape_contour(it.contour, voice.melody) if tonal else it.contour
        shift = it.pitch + voice.pitch + it.drift * i + rng.normal(0.0, 0.7)
        level = 0.85 ** i
        calls.append(Call(
            start=t, dur=dur,
            pitch=[(u, shift + s * voice.pitch_range) for u, s in contour],
            amp=[(u, a * level) for u, a in it.amp],
            harsh=[(u, base_harsh + it.harsh * a) for u, a in it.amp],  # harshest when loudest
            mouth=list(it.mouth),
            vowels=list(vowels) if it.glide else [vowels[i % len(vowels)]],
            larynx=[(u, it.larynx * a) for u, a in it.amp],
            breath=it.breath, fry=it.fry, vibrato=it.vibrato + voice.vibrato, nasal=it.nasal,
            falsetto=voice.falsetto, spread=it.spread, vibrato_rate=it.vibrato_rate,
            rumble=it.rumble, voiced=it.voiced, **nlp_levels(voice, it)))
        t += dur + gap * spacing * scale * rng.uniform(0.8, 1.2)
        gap *= it.accel
    return calls


def rhythm_at(rhythm, i, n):
    """The rhythm curve's spacing for repeat i of n (1 with no curve)."""
    if not rhythm:
        return 1.0
    return max(0.01, float(curve(rhythm, np.array([i / (n - 1) if n > 1 else 0.0]))[0]))


def sequence_calls(body, voice, it, rng):
    """A sequential call: one syllable per vowel in the list, in order, repeats included. Vowels
    joined by "~" make one longer syllable gliding through them; "|" adds a pause. it.repeats is how
    many times the whole phrase is said. Syllables keep their level (no fading as in a bark series)
    and the pitch moves by it.drift per syllable."""
    scale = body.call_scale / voice.tempo
    base_harsh = 0.5 * body.roughness + voice.harshness
    syls = syllables(it.vowels or voice.vowels)
    n = len(syls)
    lo = max(1, int(round(it.repeats[0])))
    hi = max(lo, int(round(it.repeats[1])))
    pause = max(0.25, 3.0 * it.gap) * scale
    calls, t = [], 0.0
    for p in range(rng.integers(lo, hi + 1)):
        if p > 0:
            t += pause
        if it.inhale:
            calls.append(inhale_call(t, 0.25 * scale))
            t += 0.3 * scale
        gap = it.gap
        for j, (onset, vowels, coda, pause_after) in enumerate(syls):
            tonal = len(vowels) > 1
            spacing = rhythm_at(it.rhythm, j, n)
            dur = it.dur * scale * (1.0 + 0.7 * (len(vowels) - 1)) * (voice.sustain if tonal else 1.0) \
                * (1.0 + 0.1 * rng.standard_normal()) * np.sqrt(spacing)
            dur = max(dur, 0.01)
            contour = shape_contour(it.contour, voice.melody) if tonal else it.contour
            shift = it.pitch + voice.pitch + it.drift * j + rng.normal(0.0, 0.4)
            level = 1.0 - 0.2 * j / (max(n, 2) - 1)
            # Consonants before the vowel run into it.
            for m, cons in enumerate(onset):
                if cons in STOPS:
                    # Voiced through the closure when the voice runs straight into it.
                    voice_in = m == 0 and j > 0 and not syls[j - 1][3] and not syls[j - 1][2]
                    released = not (m + 1 < len(onset) and onset[m + 1] in STOPS)
                    t = push_stop(calls, cons, t, scale, shift, vowels[0], voice_in, onset=True, released=released)
                    continue
                c = consonant_call(cons, t, scale, shift, vowels[0], leads=True)
                last = m + 1 == len(onset)
                if last and cons != "h":
                    # Released into the vowel: a brief dip, then the voice starts, with the breath
                    # running on into it through its formants.
                    calls.append(aspiration_call(c, t + c.dur - 0.1 * ASPIRATION[1] * c.dur, vowels[0], leads=True))
                    t += c.dur + RELEASE_DIP * np.sqrt(scale)
                else:
                    t += c.dur * (1.0 - CONSONANT_OVERLAP if last else 1.0)
                calls.append(c)
            v = Call(
                start=t, dur=dur,
                pitch=[(u, shift + s * voice.pitch_range) for u, s in contour],
                amp=[(u, a * level) for u, a in it.amp],
                harsh=[(u, base_harsh + it.harsh * a) for u, a in it.amp],
                mouth=list(it.mouth), vowels=list(vowels),
                larynx=[(u, it.larynx * a) for u, a in it.amp],
                breath=it.breath, fry=it.fry, vibrato=it.vibrato + voice.vibrato, nasal=it.nasal,
                falsetto=voice.falsetto, spread=it.spread, vibrato_rate=it.vibrato_rate,
                rumble=it.rumble, voiced=it.voiced, even=SEQUENCE_EVEN, level=level,
                **nlp_levels(voice, it))
            # Next to a consonant the voice starts abruptly, and the formants move from or to the
            # place of the consonant: the one closing this syllable, or else the one opening the next
            # when the voice runs straight into it.
            transition = min(FORMANT_TRANSITION * np.sqrt(scale) / dur, 0.45)
            if onset and onset[-1] != "h":
                v.amp = sharpen_onset(v.amp)
                v.formants_from = transition_to(onset[-1], vowels[0], transition)
            next_onset = syls[j + 1][0] if not pause_after and j + 1 < len(syls) else []
            after = coda[0] if coda else (next_onset[0] if next_onset else None)
            if after and after != "h":
                v.formants_to = transition_to(after, vowels[-1], transition)
            calls.append(v)
            t += dur
            # Consonants after it start as it fades.
            for m, cons in enumerate(coda):
                if cons in STOPS:
                    released = not (m + 1 < len(coda) and coda[m + 1] in STOPS)
                    t = push_stop(calls, cons, t, scale, shift, vowels[-1], m == 0, onset=False, released=released)
                    continue
                if cons in AFFRICATES:
                    start = t  # bursts out as the vowel ends
                elif m == 0:
                    start = t - CONSONANT_OVERLAP * CONSONANT_SHAPE[cons][0] * np.sqrt(scale)
                else:
                    start = t
                c = consonant_call(cons, start, scale, shift, vowels[-1], leads=False)
                if m == 0 and cons != "h" and cons not in AFFRICATES:
                    # the vowel's breath narrowing into the fricative
                    calls.append(aspiration_call(c, start + CONSONANT_OVERLAP * c.dur - 0.5 * ASPIRATION[2] * c.dur,
                                                 vowels[-1], leads=False))
                t = start + c.dur
                calls.append(c)
            # No gap before a syllable that starts with a consonant: the consonant runs on from this one.
            run_on = not pause_after and j + 1 < len(syls) and len(syls[j + 1][0]) > 0
            t += (0.0 if run_on else 1.0) * gap * spacing * scale * rng.uniform(0.8, 1.2) + (pause if pause_after else 0.0)
            gap *= it.accel
    return calls


def consonant_call(cons, start, scale, pitch, vowel, leads):
    """A consonant of a sequential call. An h is breath through the tract shaped like the vowel
    next to it; it swells into a vowel it leads, and fades after one it trails."""
    if cons == "h":
        amp = [(0, 0), (0.5, 0.8), (1, 1)] if leads else [(0, 1), (0.4, 0.7), (1, 0)]
    else:
        amp = [(0, 0), (0.15, 1), (0.85, 1), (1, 0)]
    return Call(start=start, dur=CONSONANT_SHAPE[cons][0] * np.sqrt(scale), pitch=[(0, pitch), (1, pitch)],
                amp=amp, harsh=[(0, 0), (1, 0)], mouth=[(0, 0.7), (1, 0.8)], vowels=[vowel],
                voiced=0.0, breath=1.0, consonant=cons,
                consonant_db=CONSONANT_SHAPE[cons][1] + (STRIDENT_OPEN_BACK * open_back(vowel) if cons in STRIDENTS else 0.0))


def push_stop(calls, cons, start, scale, pitch, vowel, voice_in, onset, released=True):
    """A stop from `start`: the closure (silent, or voiced when voice_in and the stop is b, d or g),
    the burst, and for p, t and k the breath through the vowel's formants until the voice starts.
    onset: released into the vowel after it, rather than closing the one before (weaker and
    shorter). A stop right before another is not released: its closure runs on into the next one's,
    as in "big gate". Returns when the next sound starts."""
    k = np.sqrt(scale)
    closure, burst_db = CONSONANT_SHAPE[cons]
    burst, vot = STOP_TIMING[cons]
    closure *= k
    if voice_in and cons in VOICED_STOPS:
        calls.append(Call(start=start, dur=closure, pitch=[(0, pitch), (1, pitch - 1.0)],
                          amp=[(0, 1.0), (0.6, 0.8), (1, 0.5)], harsh=[(0, 0), (1, 0)],
                          mouth=[(0, 0.1), (1, 0.1)], vowels=[vowel], consonant=cons,
                          consonant_db=VOICE_BAR[0], voice_bar=True))
    t = start + closure
    if not released:
        return t
    release = 0.0 if onset else CODA_RELEASE[0]
    b = consonant_call(cons, t, scale, pitch, vowel, onset)
    b.dur = burst * k
    b.consonant_db = burst_db + release
    t += b.dur
    calls.append(b)
    if vot > 0:
        a = consonant_call("h", t, 1.0, pitch, vowel, onset)
        a.dur = max((vot - burst) * k, 0.005) * (1.0 if onset else CODA_RELEASE[1])
        a.amp = [(0, 0.6), (0.15, 1.0), (0.7, 0.8), (1, 0.25)] if onset else [(0, 0.6), (0.15, 1.0), (1, 0)]
        a.consonant_db = STOP_ASPIRATION + release
        t += a.dur
        calls.append(a)
    return t


def front_vowel(v):
    """A front vowel (i, e, ae, and the fronted SSBE u:): F2 above the neutral tube's."""
    return vowel_ratios(v)[1] > 1.03


def transition_to(cons, vowel, length):
    """A consonant's locus next to the vowel, with the transition's length, for formants_from/_to."""
    if cons not in LOCUS:
        return None
    loci, pull = FRONT_VELAR if cons in ("k", "g") and front_vowel(vowel) else LOCUS[cons]
    return (loci, pull, length)


def open_back(v):
    r = vowel_ratios(v)
    return float(np.clip((r[0] - 1.0) / 0.36, 0.0, 1.0) * np.clip((1.15 - r[1]) / 0.4, 0.0, 1.0))


def sharpen_onset(amp):
    """A loudness curve whose attack (up to its first peak) is squeezed by ONSET_SHARPEN."""
    peak = max(a for _, a in amp)
    k = next(i for i, (_, a) in enumerate(amp) if a >= peak)
    return [(u * ONSET_SHARPEN if i <= k else u, a) for i, (u, a) in enumerate(amp)]


def formant_transition(call, k, u, r):
    """Formant k (as a ratio of the neutral tube's) at positions u of a call, moved towards a
    neighbouring fricative's locus at the call's start or end."""
    def toward(loci, pull, r, w):
        edge = loci[k] + LOCUS_PULL * pull[k] * (r - loci[k])
        return edge * (1.0 - w) + r * w
    if call.formants_from:
        loci, pull, length = call.formants_from
        r = toward(loci, pull, r, smoothstep(u / length))
    if call.formants_to:
        loci, pull, length = call.formants_to
        r = toward(loci, pull, r, smoothstep((1.0 - u) / length))
    return r


def aspiration_call(fricative, start, vowel, leads):
    """The breath between a fricative and its vowel (see ASPIRATION): an h through the vowel's
    tract, loudest at the boundary."""
    c = consonant_call("h", start, 1.0, fricative.pitch[0][1], vowel, leads)
    c.dur = (ASPIRATION[1] if leads else ASPIRATION[2]) * fricative.dur
    c.amp = [(0, 0), (0.3, 1), (1, 0)] if leads else [(0, 0), (0.5, 1), (1, 0)]
    c.consonant_db = fricative.consonant_db + ASPIRATION[0]
    return c


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def render_fricative(call, body, rng):
    """A fricative: turbulence where the airstream hits the teeth or lips, coloured by the small
    cavity in front of the constriction rather than by the whole tract. So its colour depends on the
    size of the mouth (smaller hisses higher) and on lip rounding from the vowel beside it, which
    lengthens that cavity and lowers the hiss ("sue" against "see").
    - s: a peak near 4.5 kHz (human size, the front cavity's quarter wave) and a weaker one near
      6.5 kHz, steeply cut below 3 kHz;
    - sh: a lower peak near 2.7 kHz with a weaker second one, cut below 2 kHz;
    - f, th: flat and faint (no cavity in front of the lips or teeth); th a little brighter;
    - ch, j: the sh hiss, short, bursting out and dying away.
    The quieter it is, the darker (SOURCE_CORNER). It holds steady between short ramps, and the
    turbulence flutters a little."""
    n = max(16, int(call.dur * SR))
    size = np.sqrt(0.17 / body.tract)
    lower = 1.0 - 0.15 * (vowel_rounding(call.vowels[0]) if call.vowels else 0.0)
    hz = lambda f: min(f * size, 0.45 * SR)
    unit = lambda x: x / rms(x)
    noise = rng.standard_normal(n)
    # (cutoff, [(peak Hz, width Hz, weight)], broadband floor weight, floor lowpass)
    cut, peaks, floor, top = {
        "s": (3000 * lower, [(4500, 1000, 1.0), (6500, 1500, 0.3)], 0.1, 9000),
        "sh": (2000 * lower, [(2700, 900, 1.0), (4800, 1800, 0.3)], 0.1, 6500),
        "ch": (2000 * lower, [(2700, 900, 1.0), (4800, 1800, 0.3)], 0.1, 6500),
        "j": (2000 * lower, [(2700, 900, 1.0), (4800, 1800, 0.3)], 0.1, 6500),
        "th": (1300, [], 1.0, 9000),
    }.get(call.consonant, (1000, [], 1.0, 8000))  # f
    top = hz(top)
    cut = min(hz(cut), 0.5 * top)
    corner = hz(SOURCE_CORNER[0] * 2.0 ** ((call.consonant_db - SOURCE_CORNER[1]) / SOURCE_CORNER[2]))
    source = sosfilt(butter(1, corner, "lowpass", fs=SR, output="sos"), noise)
    high = butter(2, cut, "highpass", fs=SR, output="sos")
    base = sosfilt(high, sosfilt(high, source))  # 24 dB/octave below the cutoff
    # The top is rolled off at 12 dB/octave: the air above it is what makes a hiss harsh.
    soft = butter(2, top, "lowpass", fs=SR, output="sos")
    base = sosfilt(soft, base)
    y = floor * unit(base)
    # The peaks are added in phase. Klatt (1980) flips every other parallel resonator to fill the dip
    # between peaks, but with a weak second peak that cancels it and everything above it.
    for f, w, weight in peaks:
        y = y + weight * unit(lfilter(*resonator(min(hz(f * lower), 0.9 * top), hz(w)), base))
    # Short ramps in and out (an affricate bursts out and dies away), and a slow flutter of the
    # turbulence (about 30 Hz and below).
    shape = CONSONANT_SHAPE[call.consonant][0]
    if call.consonant in AFFRICATES:
        rise, fall = FRICATIVE_RAMP[1] / shape, 0.6
    else:
        rise, fall = FRICATIVE_RAMP[0] / shape, FRICATIVE_RAMP[0] / shape
    rise, fall = min(rise, 0.5), min(fall, 0.6)
    flutter = sosfilt(butter(2, 30.0, "lowpass", fs=SR, output="sos"), rng.standard_normal(n))
    u = np.arange(n) / n
    env = smoothstep(u / rise) * smoothstep((1.0 - u) / fall)
    return y * env * np.exp(0.15 * flutter / (np.std(flutter) + 1e-12))


def render_burst(call, body, rng):
    """A stop's burst: the click of air as the closure opens, a few ms long, rising at once and dying
    away. Its spectrum marks the place (Klatt 1980; Stevens & Blumstein 1978):
    - p, b: diffuse and falling, nothing in front of the lips to shape it;
    - t, d: high and rising, from the small cavity behind the teeth (a peak near 4.5 kHz);
    - k, g: one compact peak near the next vowel's F2, where the tongue meets the palate (higher
      before front vowels).
    Sizes scale like a fricative's, and a quieter burst is darker (SOURCE_CORNER)."""
    n = max(16, int(call.dur * SR))
    size = np.sqrt(0.17 / body.tract)
    hz = lambda f: min(f * size, 0.45 * SR)
    unit = lambda x: x / (rms(x) + 1e-12)
    sos = lambda order, f, kind: butter(order, f, kind, fs=SR, output="sos")
    corner = hz(SOURCE_CORNER[0] * 2.0 ** ((call.consonant_db - SOURCE_CORNER[1]) / SOURCE_CORNER[2]))
    source = sosfilt(sos(1, corner, "lowpass"), rng.standard_normal(n))
    if call.consonant in ("p", "b"):
        y = sosfilt(sos(2, hz(3000), "lowpass"), sosfilt(sos(2, hz(300), "highpass"), source))
    elif call.consonant in ("t", "d"):
        high = sos(2, hz(2500), "highpass")
        base = sosfilt(sos(2, hz(8000), "lowpass"), sosfilt(high, sosfilt(high, source)))
        y = 0.3 * unit(base) + unit(lfilter(*resonator(hz(4500), hz(1500)), base))
    else:
        f2 = (vowel_ratios(call.vowels[0])[1] if call.vowels else 1.0) * 1500.0
        peak = float(np.clip(1.1 * f2, 1500.0, 2600.0))
        base = sosfilt(sos(2, hz(5000), "lowpass"), sosfilt(sos(2, hz(1000), "highpass"), source))
        y = 0.15 * unit(base) + unit(lfilter(*resonator(hz(peak), hz(500)), base))
    i = np.arange(n)
    u = i / n
    return y * np.minimum(i / (0.0004 * SR), 1.0) * np.exp(-3.0 * u) * smoothstep((1.0 - u) / 0.15)


def render_voice_bar(call, body, rng):
    """Voicing through a voiced stop's closure: the call as rendered, with only its lowest
    harmonics left (the closed tract's walls pass nothing higher)."""
    y = render_call(call, body, rng)
    low = butter(2, min(VOICE_BAR[1] * 0.17 / body.tract, 0.45 * SR), "lowpass", fs=SR, output="sos")
    return sosfilt(low, sosfilt(low, y))


def render_any(call, body, rng):
    """A call by its kind: voice bar, stop burst, fricative, or voice and breath."""
    if call.voice_bar:
        return render_voice_bar(call, body, rng)
    if call.consonant in STOPS:
        return render_burst(call, body, rng)
    if call.consonant not in (None, "h"):
        return render_fricative(call, body, rng)
    return render_call(call, body, rng)


def territorial_bout(body, voice, rng):
    """Intro moan, full roar, then accelerating grunts with inhales (from lion_roar.py)."""
    s = body.call_scale / voice.tempo / 1.17  # lion_roar.py timings are for a 190 kg lion
    j = lambda v, sd=0.06: v * (1.0 + sd * rng.standard_normal())
    p = lambda pts: [(u, voice.pitch + st * voice.pitch_range) for u, st in pts]
    base = 0.5 * body.roughness + voice.harshness
    h = lambda pts: [(u, base + dh) for u, dh in pts]
    vowels = plain_vowels(voice.vowels)
    first, last = vowels[0], vowels[-1]
    calls = [
        Call(start=0.0, dur=j(0.9) * s, breath=0.1, fry=0.4, vowels=[first], nasal=0.3,
             pitch=p([(0, -9.9), (0.5, j(-4.6, 0.1)), (1, -7.6)]),
             amp=[(0, 0), (0.3, 0.3), (0.8, 0.28), (1, 0)], harsh=h([(0, -0.3), (1, -0.15)]),
             mouth=[(0, 0.05), (0.6, 0.25), (1, 0.1)], larynx=[(0, 0), (1, 0.2)],
             **nlp_levels(voice, sub=0.1)),
        inhale_call(1.0 * s, 0.35 * s),
        Call(start=1.4 * s, dur=j(2.3) * s, breath=0.1, fry=0.8, vowels=list(vowels),
             pitch=p([(0, -6.5), (0.15, j(2.1, 0.3)), (0.4, j(7.9, 0.1)), (0.6, 6.5), (0.85, -1.3), (1, -8.7)]),
             amp=[(0, 0), (0.08, 0.6), (0.35, 1.0), (0.7, 0.9), (0.9, 0.4), (1, 0)],
             harsh=h([(0, -0.1), (0.3, 0.3), (0.5, 0.5), (0.75, 0.2), (1, 0.0)]),
             mouth=[(0, 0.3), (0.3, 0.9), (0.7, 1.0), (1, 0.4)],
             larynx=[(0, 0.2), (0.3, 1.0), (0.8, 1.0), (1, 0.5)],
             **nlp_levels(voice, sub=0.5, chaos=0.3)),
    ]
    t = 4.0 * s
    for i, gap in enumerate([0.95, 0.85, 0.75, 0.7, 0.65]):
        level = 1.4 - 0.15 * i  # low f0 carries less energy after radiation, so push grunts harder
        calls.append(inhale_call(t - 0.3 * s, 0.28 * s))
        calls.append(Call(start=t, dur=j(0.42, 0.1) * s, breath=0.1, fry=0.9, vowels=[last],
                          pitch=p([(0, -7.6), (0.25, j(-3.7, 0.15)), (1, -13.1)]),
                          amp=[(0, 0), (0.12, level), (0.5, level * 0.8), (1, 0)],
                          harsh=h([(0, 0.25), (0.4, 0.45), (1, 0.15)]),
                          mouth=[(0, 0.4), (0.3, 0.85), (1, 0.3)],
                          larynx=[(0, 0.5), (0.3, 0.9), (1, 0.6)],
                          **nlp_levels(voice, sub=0.4, chaos=0.2)))
        t += gap * s
    return calls


# ---------------------------------------------------------------- sources

def rosenberg_pulse(p, open_quotient):
    """Rosenberg glottal flow pulse over phase p in [0, 1)."""
    tp = 0.72 * open_quotient   # opening phase
    tn = 0.28 * open_quotient   # closing phase (shorter = brighter; abrupt closure is what gives a voice bite)
    g = np.zeros_like(p)
    opening = p < tp
    g[opening] = 0.5 * (1.0 - np.cos(np.pi * p[opening] / tp))
    closing = (p >= tp) & (p < tp + tn)
    g[closing] = np.cos(0.5 * np.pi * (p[closing] - tp) / tn)
    return g


def larynx_source(call, f0, amp, harsh, rng, base_f0, open_quotient=OPEN_QUOTIENT):
    """Period-by-period pulse train so jitter/shimmer/period-doubling are per cycle.
    Returns (flow, open): `open` is the unscaled glottal opening (0..1), used to pulse breath noise."""
    flow = np.zeros(len(f0))
    opening = np.zeros(len(f0))
    t, k, drift = 0.0, 0, 0.0
    tremor_rate = 5.0 + rng.uniform(0.0, 2.0)
    tremor_phase = rng.uniform(0.0, 2.0 * np.pi)
    amp_peak = float(np.max(amp)) + 1e-9
    # NLP regime: 0 = normal, 2/3 = subharmonic period, -1 = chaos. `cyc` counts cycles in it.
    regime, cyc, chaos_x = 0, 0, 0.5
    while t < call.dur:
        i = min(int(t * SR), len(f0) - 1)
        u = t / call.dur
        a, h = amp[i], harsh[i]

        # Creaky tail: as air runs out the folds slacken, pitch sags and cycles turn irregular.
        tail = np.clip((u - 0.8) / 0.2, 0.0, 1.0) * call.fry
        freq = f0[i] * (1.0 - 0.45 * tail)
        h_voice = h  # harshness before the creaky tail
        h = min(1.0, h + 0.4 * tail)

        # Slow wander plus cycle-to-cycle jitter; jitter explodes when harsh (chaos). The wander is
        # paced in seconds, not cycles: per-cycle pacing made high voices wobble fast and sound hoarse.
        keep = np.exp(-1.0 / (freq * DRIFT_TIME))
        drift = keep * drift + np.sqrt(1.0 - keep * keep) * rng.standard_normal()
        # Fry pulses are notoriously irregular in timing and size; evenly spaced ones motorboat.
        fry = float(np.clip(1.0 - freq / (0.6 * base_f0), 0.0, 1.0))
        jitter = BASE_JITTER + CHAOS_JITTER * h ** 3 + 0.05 * tail + FRY_JITTER * fry
        period = 1.0 / (freq * (1.0 + DRIFT_DEPTH * drift + jitter * rng.standard_normal()))
        # Harshness terms are quadratic: mild harshness should barely roughen a clean voice.
        shimmer = 1.0 + (BASE_SHIMMER + 0.25 * h * h + FRY_SHIMMER * fry) * rng.standard_normal()
        a *= 1.0 + 0.07 * np.sin(2.0 * np.pi * tremor_rate * t + tremor_phase)  # respiratory tremor

        # NLP episodes. Loud moments (high subglottal pressure) push the folds over a bifurcation:
        # the voice switches abruptly into a subharmonic or chaotic regime, holds it, then drops back.
        drive = (amp[i] / amp_peak) ** 2
        dt = 1.0 / freq
        if regime == 0:
            p_sub = 1.0 - np.exp(-NLP_RATE * call.subharmonics * drive * dt)
            p_chaos = 1.0 - np.exp(-NLP_RATE * call.chaos * drive * dt)
            r = rng.random()
            if r < p_sub:
                regime, cyc = (3 if rng.random() < 0.25 else 2), 0
            elif r < p_sub + p_chaos:
                regime, cyc, chaos_x = -1, 0, rng.uniform(0.2, 0.8)
        else:
            level = call.chaos if regime < 0 else call.subharmonics
            mean_len = NLP_EPISODE[0] + (NLP_EPISODE[1] - NLP_EPISODE[0]) * level
            if rng.random() < 1.0 - np.exp(-dt / mean_len):
                regime = 0
        cyc += 1

        # Period doubling from harshness alone: alternate cycles differ in length and strength.
        sub = np.clip((h - 0.35) / 0.5, 0.0, 1.0) * 0.55
        if regime > 0:
            # Subharmonic episode: one strong cycle then weaker ones, every 2 or 3 cycles -> f0/2, f0/3.
            depth = max(sub, SUB_DEPTH * (0.5 + 0.5 * call.subharmonics))
            if cyc % regime:
                period *= 1.0 + SUB_TIMING * depth / (regime - 1)
                a *= 1.0 - depth * (1.0 if cyc % regime == 1 else 0.5)
            else:
                period *= 1.0 - SUB_TIMING * depth
        elif k % 2:
            period *= 1.0 + 0.12 * sub
            a *= 1.0 - sub
        else:
            period *= 1.0 - 0.12 * sub
        if regime < 0:
            # Deterministic chaos: a logistic map just past the onset of chaos drives period and
            # strength. Irregular but correlated cycle to cycle (banded), so it stays a voice, not hiss.
            chaos_x = (3.6 + 0.4 * call.chaos) * chaos_x * (1.0 - chaos_x)
            swing = 2.0 * chaos_x - 1.0
            period *= 1.0 + CHAOS_PERIOD * call.chaos * swing
            a *= max(0.0, 1.0 + CHAOS_AMP * call.chaos * swing)

        # Pressed (harsh) phonation closes faster -> more high harmonics; shape varies per cycle.
        oq = open_quotient * (1.0 - 0.385 * h) * (1.0 + (SHAPE_JITTER + 0.05 * h * h) * rng.standard_normal())
        # Fry register: far below its natural pitch the glottis still opens only briefly, so
        # each cycle is a sharp click that rings every formant (pig grunts, creaky growls).
        oq = float(np.clip(oq * min(1.0, freq / (0.6 * base_f0)) ** 3, 0.03, 0.9))
        i0 = int(np.ceil(t * SR))
        i1 = min(int(np.ceil((t + period) * SR)), len(flow))
        idx = np.arange(i0, i1)
        # Falsetto: well above its natural pitch the folds thin out and vibrate almost
        # sinusoidally, so the voice keeps only a few harmonics (a howl, not a buzz).
        thin = call.falsetto * np.clip((np.log2(freq / base_f0) - 0.3) / 0.7, 0.0, 1.0) * (1.0 - h_voice ** 2)
        p = (idx / SR - t) / period
        pulse = (1.0 - thin) * rosenberg_pulse(p, oq) + thin * 0.5 * (1.0 - np.cos(2.0 * np.pi * p))
        flow[idx] += max(a * shimmer, 0.0) * pulse
        opening[idx] = np.maximum(opening[idx], pulse)
        t += period
        k += 1
    return flow, opening


def episodes(n, rate, mean_len, rng):
    """0/1 mask of random on-episodes (onsets per second, mean length in s), with 5 ms ramps so
    abrupt switches don't click."""
    mask = np.zeros(n)
    t = rng.exponential(1.0 / rate) if rate > 0 else n
    while rate > 0 and t * SR < n:
        length = rng.exponential(mean_len)
        mask[int(t * SR):int((t + length) * SR)] = 1.0
        t += length + rng.exponential(1.0 / rate)
    ramp = max(1, int(0.005 * SR))
    return np.convolve(mask, np.ones(ramp) / ramp, mode="same")


def register_jumps(call, n, rng):
    """Frequency jumps: the folds snap into another vibration mode, so the pitch leaps abruptly
    (several semitones, mostly upward when aroused) and later snaps back. Returns semitone steps."""
    steps = np.zeros(n)
    # A few breaks per call, not one per half second: a long howl with a break every beat reads as
    # a robot. Breaks are short and never overlap, so they can't stack into a staircase.
    count = rng.poisson(call.jumps * (0.5 + 0.25 * call.dur))
    t = rng.uniform(0.1, 0.4) * call.dur
    for _ in range(count):
        length = min(0.04 + rng.exponential(0.12), 0.3, 0.4 * call.dur)
        if t + length > 0.9 * call.dur:
            break
        size = rng.uniform(3.0, 7.0) * (1.0 if rng.random() < 0.7 else -1.0)
        steps[int(t * SR):min(int((t + length) * SR), n)] = size
        t += length + rng.uniform(0.15, 0.4) * call.dur
    ramp = max(1, int(0.004 * SR))
    return np.convolve(steps, np.ones(ramp) / ramp, mode="same")


def biphonate(flow, call, f0_contour, amp, rng):
    """Biphonation: a second, independent oscillation (a vocal membrane, or the two folds
    desynchronising) at a non-harmonic pitch ratio. It adds its own tone and modulates the main
    flow, giving sidebands at f0 +- g. Comes and goes in long episodes.

    The second tone follows only the call's slow contour (`f0_contour`: no vibrato, no register
    jumps) and wanders on its own. Tracking every move of the main pitch would make it a parallel
    harmony, which sounds like a robot."""
    n = len(flow)
    level = call.biphonation
    mask = episodes(n, 0.5 + 1.0 * level, 0.6 + 1.4 * level, rng)
    g = call.biphonation_ratio * f0_contour * (1.0 + 0.04 * smooth_noise(n, 0.5, rng))  # wanders, never locks
    phase = 2.0 * np.pi * np.cumsum(g) / SR
    b = level * mask
    return flow * (1.0 + BIPHONATION_COUPLING * b * np.sin(phase)) + b * amp * 0.5 * (1.0 - np.cos(phase))


def smooth_noise(n, rate, rng):
    """Band-limited random wander, roughly `rate` changes per second."""
    sos = butter(1, rate, "lowpass", fs=SR, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    return x / (np.std(x) + 1e-9)


def syrinx_source(call, f0, amp, harsh, rng, base_f0, buzz=0.0, open_quotient=OPEN_QUOTIENT):
    """Two independently controlled tonal voices (left and right syrinx) for song. The syrinx
    labia vibrate like vocal folds, so harsh calls (caws, quacks) cross over into the same
    pulsed, chaotic oscillation as a harsh larynx rather than a brighter tone."""
    n = len(f0)
    h = harsh
    jitter = (0.003 + 0.03 * h ** 2) * smooth_noise(n, 300.0, rng)
    phase1 = 2.0 * np.pi * np.cumsum(f0 * (1.0 + jitter)) / SR
    ratio = 2.0 ** (rng.uniform(3.0, 7.0) / 12.0)
    phase2 = 2.0 * np.pi * np.cumsum(f0 * ratio * (1.0 + 0.01 * smooth_noise(n, 20.0, rng))) / SR
    drive = 1.0 + 2.0 * h
    tonal = np.tanh(drive * np.sin(phase1)) / np.tanh(drive) + np.sin(phase2) * (0.05 + 0.4 * h)
    pulsed = np.maximum(buzz, np.clip((h - 0.3) / 0.4, 0.0, 1.0))
    tonal_open = 0.5 * (1.0 + np.sin(phase1))
    if not pulsed.any():
        return amp * tonal, tonal_open
    # Fry is judged against the call's own pitch here: the songbird-scaled natural pitch would
    # turn every low quack into razor-sharp clicks.
    pulses, opening = larynx_source(call, f0, amp, harsh, rng, min(base_f0, float(np.median(f0))), open_quotient)
    return ((1.0 - pulsed) * amp * tonal + pulsed * 2.0 * (pulses - np.mean(pulses)),
            (1.0 - pulsed) * tonal_open + pulsed * opening)


def pulse_source(call, rate, amp, harsh, carrier, rng, harmonic=0.0):
    """Pulse train (frog trill, insect chirp): every pulse is a short burst ringing at the carrier,
    plus `harmonic` of its octave (frog croaks show two bands)."""
    out = np.zeros(len(rate))
    t = 0.0
    while t < call.dur:
        i = min(int(t * SR), len(rate) - 1)
        h = harsh[i]
        period = 1.0 / (rate[i] * (1.0 + (0.02 + 0.2 * h) * rng.standard_normal()))
        length = min(period * 0.8, 12.0 / carrier[i])
        tt = np.arange(int(length * SR)) / SR
        ring = np.exp(-tt * carrier[i] / 3.0)
        burst = (np.sin(2.0 * np.pi * carrier[i] * tt) + harmonic * np.sin(4.0 * np.pi * carrier[i] * tt)) * ring
        burst += h * 0.5 * rng.standard_normal(len(tt)) * np.exp(-tt * carrier[i] / 2.0)
        i0 = int(t * SR)
        i1 = min(i0 + len(tt), len(out))
        out[i0:i1] += amp[i] * burst[: i1 - i0]
        t += max(period, 1.0 / SR)
    return out


# ---------------------------------------------------------------- filters

def resonator(freq, bw):
    """Klatt-style two-pole resonator with unity gain at DC."""
    r = np.exp(-np.pi * bw / SR)
    b1 = 2.0 * r * np.cos(2.0 * np.pi * freq / SR)
    b2 = -r * r
    return [1.0 - b1 - b2], [1.0, -b1, -b2]


@njit(cache=True)
def _tube(x, delay, loss, lp_a, fn, fv):
    """Per-sample lossy tube: a delay line with negative feedback (closed at the glottis, open
    at the lips) resonates at every odd quarter wavelength, so the formant series never ends and
    the spectral envelope stays flat like a real tract. A one-pole lowpass in the loop makes the
    upper formants broader (wall and radiation losses). Three pole-zero sections then move the
    neutral F1-F3 (fn) to the vowel's targets (fv)."""
    n = len(x)
    size = 1
    while size < int(delay.max()) + 4:
        size *= 2
    mask = size - 1
    buf = np.zeros(size)
    state = np.zeros((3, 4))
    y = np.empty(n)
    lp = 0.0
    two_pi = 2.0 * np.pi / SR
    for i in range(n):
        d = delay[i]
        di = int(d)
        frac = d - di
        r0 = buf[(i - di) & mask]
        r1 = buf[(i - di - 1) & mask]
        lp = (1.0 - lp_a) * (r0 + frac * (r1 - r0)) + lp_a * lp
        out = x[i] - loss[i] * lp
        buf[i & mask] = out

        v = out
        for k in range(3):
            # Match the comb's own pole radius at this formant so the zero cancels it cleanly.
            w = two_pi * fn[k, i]
            lp_mag = (1.0 - lp_a) / np.sqrt(1.0 - 2.0 * lp_a * np.cos(w) + lp_a * lp_a)
            r = (loss[i] * lp_mag) ** (2.0 / (d + lp_a / (1.0 - lp_a)))
            cz = 2.0 * r * np.cos(w)
            cp = 2.0 * r * np.cos(two_pi * fv[k, i])
            r2 = r * r
            g = (1.0 - cp + r2) / (1.0 - cz + r2)  # unity gain at DC
            o = g * (v - cz * state[k, 0] + r2 * state[k, 1]) + cp * state[k, 2] - r2 * state[k, 3]
            state[k, 1] = state[k, 0]
            state[k, 0] = v
            state[k, 3] = state[k, 2]
            state[k, 2] = o
            v = o
        y[i] = v
    return y


@njit(cache=True)
def _formant_peaks(x, freqs, bws, gains_db):
    """Cascade of peaking (bell) filters that follow the formants: adds resonance at each formant
    without changing the level between them. Coefficients are refreshed every 8 samples."""
    n = len(x)
    k_count = freqs.shape[0]
    y = x.copy()
    for k in range(k_count):
        if gains_db[k] == 0.0:
            continue
        a_lin = 10.0 ** (gains_db[k] / 40.0)
        z1 = 0.0
        z2 = 0.0
        b0 = b1 = b2 = a1 = a2 = 0.0
        for i in range(n):
            if i % 8 == 0:
                w0 = 2.0 * np.pi * freqs[k, i] / SR
                alpha = np.sin(w0) * bws[k, i] / (2.0 * freqs[k, i])  # sin(w0) / (2Q), Q = f / bandwidth
                norm = 1.0 / (1.0 + alpha / a_lin)
                b0 = (1.0 + alpha * a_lin) * norm
                b1 = -2.0 * np.cos(w0) * norm
                b2 = (1.0 - alpha * a_lin) * norm
                a1 = b1
                a2 = (1.0 - alpha / a_lin) * norm
            v = y[i]
            out = b0 * v + z1
            z1 = b1 * v - a1 * out + z2
            z2 = b2 * v - a2 * out
            y[i] = out
    return y


def vocal_tract(x, u, call, body, rng, bw_scale=1.0, f0=None):
    """Lossy tube for this tract length, shaped by the current vowel. Lowering the larynx
    lengthens the tube (all formants drop, sounding bigger), jaw opening raises F1/F2, and the
    whole tract drifts slowly the way a real jaw and tongue never hold still.

    When the pitch nears F1 the open glottis damps the tract (source-tract interaction). Without
    that, a high voice sitting on a sharp F1 rings out as a near-pure tone, like a flute."""
    n = len(x)
    mouth = curve(call.mouth, u)
    larynx = curve(call.larynx, u) if call.larynx else np.zeros_like(u)
    length = body.tract * (1.0 + body.larynx_drop * larynx) * (1.0 + 0.01 * smooth_noise(n, 3.0, rng))

    lp_a = float(np.clip(TRACT_LOSS_LOWPASS * body.hf_loss, 0.05, 0.9))
    round_trip = 2.0 * length / C_SOUND * SR
    delay = np.maximum(round_trip - lp_a / (1.0 - lp_a), 2.0)  # the loop lowpass adds its own delay
    f1 = SR / (2.0 * round_trip)

    vowel_table = np.array([vowel_ratios(v) for v in call.vowels])
    vowel_pos = np.linspace(0.0, 1.0, len(call.vowels)) if len(call.vowels) > 1 else np.zeros(1)
    fn = np.array([(2 * k + 1) * f1 for k in range(3)])
    ratios = np.array([np.interp(u, vowel_pos, vowel_table[:, k]) for k in range(3)])
    ratios = np.array([formant_transition(call, k, u, ratios[k]) for k in range(3)])
    fv = fn * ratios
    fv[0] *= 0.75 + 0.5 * mouth
    fv[1] *= 0.9 + 0.2 * mouth
    fv *= 1.0 + 0.03 * np.array([smooth_noise(n, 2.0, rng) for _ in range(3)])
    fv = np.minimum(fv, 0.45 * SR)
    fn = np.minimum(fn, 0.45 * SR)

    bw1 = (60.0 + 0.07 * f1) * bw_scale
    nearness = 0.0
    if f0 is not None:
        nearness = np.exp(-(((f0 - fv[0]) / (0.5 * fv[0])) ** 2))
        bw1 = bw1 + GLOTTAL_DAMPING * f0 * nearness
    loss = np.exp(-np.pi * bw1 * round_trip / SR)  # round-trip loss giving F1 that bandwidth
    y = _tube(x, delay, loss, lp_a, fn, fv)

    if body.formant_gain:
        # Formant resonance: F1-F3 at the vowel's targets, F4-F5 on the neutral tube. F1 keeps the
        # glottal damping above, so a high voice sitting on F1 still doesn't ring like a flute.
        peaks = np.vstack([fv, np.minimum([7.0 * f1, 9.0 * f1], 0.42 * SR)])
        peaks = np.minimum(peaks, 0.42 * SR)
        widths = body.formant_width * bw_scale * (40.0 + 0.06 * peaks)
        if f0 is not None:
            widths[0] = widths[0] + GLOTTAL_DAMPING * f0 * nearness
        y = _formant_peaks(y, peaks, widths, body.formant_gain * np.asarray(FORMANT_LEVELS))

    # A small or rounded lip opening radiates high frequencies poorly: "oo" howls and closed-mouth
    # growls are dark, a wide-open bark is bright. Cutoffs scale with size like the formants do.
    rounding = np.interp(u, vowel_pos, [vowel_rounding(v) for v in call.vowels])
    aperture = np.clip(mouth * (1.0 - 0.8 * rounding * (1.0 - call.spread)), 0.0, 1.0)
    closed, wide = LIP_CUTOFF_HZ
    cutoff = (closed + (wide - closed) * aperture ** 1.5) * (0.17 / body.tract) ** 0.7
    return _lips(y, np.minimum(cutoff, 0.45 * SR)), mouth


@njit(cache=True)
def _lips(x, cutoff):
    """Three cascaded one-pole lowpasses with a per-sample cutoff."""
    y = np.empty_like(x)
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    for i in range(len(x)):
        a = np.exp(-2.0 * np.pi * cutoff[i] / SR)
        s1 = (1.0 - a) * x[i] + a * s1
        s2 = (1.0 - a) * s1 + a * s2
        s3 = (1.0 - a) * s2 + a * s3
        y[i] = s3
    return y


def nasalize(y, mouth, call, body):
    """Sound leaking through the nose: an antiresonance (scaled to the tract) and a softer top end.
    Kept partial; a fully nasal mix muffles everything to mud."""
    notch = min(800.0 * 0.38 / body.tract, 0.35 * SR)
    b, a = iirnotch(notch, 1.5, fs=SR)
    nasal = sosfilt(butter(1, min(4.0 * notch, 0.45 * SR), "lowpass", fs=SR, output="sos"), lfilter(b, a, y))
    mix = np.clip(body.nasality + call.nasal + 0.8 * (0.3 - mouth), 0.0, 0.5)
    return (1.0 - mix) * y + mix * nasal


# ---------------------------------------------------------------- rendering

def render_call(call, body, rng):
    n = int(call.dur * SR) + SR // 10
    u = np.clip(np.arange(n) / (call.dur * SR), 0.0, 1.0)
    amp = curve(call.amp, u)
    if call.rumble > 0:
        # Gravel: secondary tissue flutters irregularly at 20-50 Hz. Band-limited noise, never a
        # steady beat, which would read as a propeller. Applied in log amplitude so it never gates
        # the call to silence; linear modulation clipped at zero chops it into staccato puffs.
        wobble = sosfilt(butter(2, [20.0, 50.0], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
        amp = amp * np.exp(0.5 * call.rumble * wobble / (np.std(wobble) + 1e-9))
    harsh = np.clip(curve(call.harsh, u), 0.0, 1.0)
    # Animal vibrato wanders in rate and depth; a perfectly regular one is a musician's.
    rate = call.vibrato_rate * (1.0 + 0.15 * smooth_noise(n, 1.5, rng))
    depth = np.maximum(call.vibrato * (1.0 + 0.35 * smooth_noise(n, 2.0, rng)), 0.0)
    vibrato = depth * np.sin(2.0 * np.pi * np.cumsum(rate) / SR) * np.clip(u * 2.0 - 0.3, 0.0, 1.0)
    contour = curve(call.pitch, u)
    semitones = contour + vibrato
    if call.voiced > 0 and call.jumps > 0:
        semitones = semitones + register_jumps(call, n, rng)
    f0 = body.base_f0 * 2.0 ** (semitones / 12.0)

    if body.source == "pulse":
        carrier = body.carrier * 2.0 ** (semitones / 24.0)
        y = pulse_source(call, f0, amp, harsh, carrier, rng, body.resonator_harmonic) if call.voiced > 0 else np.zeros(n)
        y += 0.05 * call.breath * amp * rng.standard_normal(n)
        # Vocal sac / body resonance, and its octave when the call has a second band.
        b, a = resonator(body.carrier, 0.15 * body.carrier)
        out = lfilter(b, a, y)
        if body.resonator_harmonic > 0 and 2.0 * body.carrier < 0.45 * SR:
            b, a = resonator(2.0 * body.carrier, 0.15 * body.carrier)
            out += body.resonator_harmonic * lfilter(b, a, y)
        return out

    voiced = call.voiced > 0 and body.source != "hiss"
    flow, opening = np.zeros(n), np.ones(n)
    if voiced and body.source == "larynx":
        flow, opening = larynx_source(call, f0, amp, harsh, rng, body.base_f0, body.open_quotient)
    elif voiced and body.source == "syrinx":
        flow, opening = syrinx_source(call, f0, amp, harsh, rng, body.base_f0, body.buzz, body.open_quotient)
    if voiced and body.source == "larynx" and call.biphonation > 0:
        flow = biphonate(flow, call, body.base_f0 * 2.0 ** (contour / 12.0), amp, rng)

    # Turbulence noise, pulsing with the glottal flow and louder when harsh. It falls off with
    # frequency (white noise through a flat tract plus radiation is hiss); the corner tracks the
    # tract's own F1, except for hissing animals.
    # Growls are mostly turbulence and tissue flutter: harsher calls get brighter, broadband noise.
    harsh_mean = np.sum(harsh * amp) / (np.sum(amp) + 1e-9)
    corner = 4000.0 if body.source == "hiss" else min(3.0 * C_SOUND / (4.0 * body.tract), NOISE_CORNER_MAX) * (1.0 + harsh_mean ** 2) * body.noise_color
    noise = sosfilt(butter(2, 150, "highpass", fs=SR, output="sos"), rng.standard_normal(n))
    noise = sosfilt(butter(1, min(corner, 0.4 * SR), "lowpass", fs=SR, output="sos"), noise)
    # Pulsed with the glottal opening of every cycle (not the call's loudness), so quiet stretches
    # pulse as deeply as loud ones and the noise fuses into the voice instead of hissing beside it.
    gate = (1.0 - body.breath_pulse) + body.breath_pulse * opening if voiced else 1.0
    turbulence = noise * gate * amp * (0.4 + 0.8 * harsh)
    breath = body.breathiness + call.breath

    # Birds barely filter their song; mammal and reptile tracts shape it strongly. Voice and breath
    # go through the same tract (same drift) separately so they can be mixed at a known ratio.
    bw_scale = body.tract_damping or (4.0 if body.source == "syrinx" else 1.0)
    tract_seed = int(rng.integers(2 ** 31))

    def radiate(x):
        y, mouth = vocal_tract(x, u, call, body, np.random.default_rng(tract_seed), bw_scale,
                               f0 if voiced else None)
        return lfilter([1.0, -0.97], [1.0], y), mouth  # lip radiation (+6 dB/oct)

    if voiced:
        y, mouth = radiate(flow)
        if breath > 0:
            # Set breath relative to the voice itself: absolute noise levels came out anywhere from
            # inaudible to louder than the voice, depending on body size and pitch.
            y_noise, _ = radiate(turbulence)
            # The cubic growl term keeps clean calls clean while real growls come out roughly
            # as much noise as tone, as recordings show.
            ratio = BREATH_FLOOR + BREATH_SCALE * breath * (0.5 + harsh_mean) + GROWL_NOISE * harsh_mean ** 3
            y = y + ratio * rms(y) / rms(y_noise) * y_noise
    else:
        y, mouth = radiate(turbulence * breath)

    if body.source in ("larynx", "syrinx"):
        y = nasalize(y, mouth, call, body)
    if body.source == "larynx":
        # Chest/body resonance driven by the raw flow: the low thump of big animals.
        b, a = resonator(95.0 * (190.0 / body.mass) ** (1.0 / 3.0), body.chest_width)
        y = y + CHEST_GAIN * body.chest * min(1.0, body.mass / 20.0) * lfilter(b, a, flow)
    if call.consonant == "h":
        # Breath from the open glottis: its F1 is heavily damped.
        r0 = vowel_ratios(call.vowels[0])[0] if call.vowels else 1.0
        f1 = C_SOUND / (4.0 * body.tract) * r0 * 1.125
        y = sosfilt(butter(2, min(ASPIRATION_F1_CUT * f1, 0.45 * SR), "highpass", fs=SR, output="sos"), y)
    if body.low_cut:
        # A small mouth or nostrils radiate low frequencies poorly (a pig's snout).
        y = sosfilt(butter(2, body.low_cut, "highpass", fs=SR, output="sos"), y)
    return y


def outdoor_reverb(x, rng, wet):
    """Short, diffuse tail. `wet` is the tail's RMS relative to the dry sound (0.1 = -20 dB):
    open terrain reflects little, and the game's own mixer can add environment reverb."""
    tail_len = int(0.8 * SR)
    tt = np.arange(tail_len) / SR
    ir = rng.standard_normal(tail_len) * np.exp(-tt / 0.1)
    ir = sosfilt(butter(2, 4000, "lowpass", fs=SR, output="sos"), ir)
    ir[: int(0.015 * SR)] = 0.0
    ir *= wet / np.sqrt(np.sum(ir ** 2))  # normalize by energy, not peak
    ir[0] = 1.0
    return fftconvolve(x, ir)[: len(x) + tail_len // 2]


def create_vocalization(body, voice, intent="idle", seed=0, distance=20.0, reverb=1.0):
    """Render one vocalization. Returns mono float32 audio at SR, peak-normalized to 0.9."""
    if intent not in INTENT_NAMES:
        raise ValueError(f"unknown intent {intent!r}; choose from {INTENT_NAMES}")
    rng = np.random.default_rng(seed)
    calls = build_calls(body, voice, intent, rng)
    rendered = [(c, render_any(c, body, rng)) for c in calls]

    # Sequential syllables: quiet vowels brought part way up to the others.
    even_ref = max([rms(y) / c.level for c, y in rendered if c.even > 0], default=0.0)
    if even_ref > 0:
        rendered = [(c, y * (even_ref * c.level / rms(y)) ** c.even if c.even > 0 else y) for c, y in rendered]

    # Consonants are set against the loudest syllable.
    syllable_peak = max([rms(y) for c, y in rendered if c.consonant is None and c.voiced > 0], default=0.0)
    if syllable_peak > 0:
        rendered = [(c, y * syllable_peak * 10 ** (c.consonant_db / 20) / rms(y)
                     if c.consonant else y) for c, y in rendered]

    # Inhales are set against the calls they precede, not in absolute terms.
    voiced_levels = [rms(y) for c, y in rendered if c.voiced > 0]
    if voiced_levels and body.source != "hiss":
        loudest = max(voiced_levels)
        rendered = [(c, y if c.voiced > 0 or c.consonant else y * INHALE_LEVEL * loudest / rms(y))
                    for c, y in rendered]

    total = max(int(c.start * SR) + len(y) for c, y in rendered)
    mix = np.zeros(total)
    for c, y in rendered:
        s = int(c.start * SR)
        mix[s:s + len(y)] += y

    # Nothing useful sits well below the voice's own pitch; there it only adds rumble and mud.
    mix = sosfilt(butter(2, max(30.0, 0.35 * body.base_f0), "highpass", fs=SR, output="sos"), mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    peak_harsh = max(np.clip(max(v for _, v in c.harsh), 0.0, 1.0) for c in calls)
    drive = 1.0 + 0.8 * peak_harsh
    mix = np.tanh(drive * mix) / np.tanh(drive)  # tissue/airflow saturation grit

    # Air absorbs highs over distance; farther sources are also more reverberant.
    cutoff = max(1500.0, 16000.0 * np.exp(-distance / 150.0))
    mix = sosfilt(butter(2, min(cutoff, 0.45 * SR), "lowpass", fs=SR, output="sos"), mix)
    mix = outdoor_reverb(mix, rng, wet=reverb * min(0.35, 0.06 + 0.002 * distance))
    return (0.9 * mix / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- presets and CLI

PRESETS = {
    "lion": (Body(mass=190, tract_length=0.38, roughness=0.5),
             Voice(vowels=["a", "o"], harshness=0.2)),
    "tiger": (Body(mass=220, tract_length=0.42, roughness=0.4, larynx_drop=0.2, breathiness=0.15, open_quotient=0.4),
              Voice(vowels=["o", "a"], pitch=-1, calls={
                  "warn": {"dur": 1.4, "repeats": (1, 1), "harsh": 0.75, "inhale": False,
                           "mouth": [(0, 0.3), (0.3, 0.6), (1, 0.3)]},
                  "territorial": {"like": "warn", "dur": 4.0, "harsh": 0.45, "inhale": False,
                                  "mouth": [(0, 0.2), (0.3, 0.5), (1, 0.3)]}})),
    "wolf": (Body.from_dimensions(40, head_length=0.25, neck_length=0.12, roughness=0.25, nasality=0.15, breathiness=0.1),
             Voice(vowels=["u", "o"], pitch=8, pitch_range=1.6, vibrato=-0.3, sustain=3.0, melody=0.1,
                   falsetto=0.35)),  # vibrato cancels most of the mating intent's: howls are steady
    "cow": (Body(mass=600, roughness=0.3, nasality=0.3, fold_scale=0.8),
            Voice(vowels=["o", "a"], pitch_range=0.7)),
    "deer": (Body(mass=150, tract_length=0.35, larynx_drop=0.5, roughness=0.55),
             Voice(vowels=["o", "a"], pitch=-2, harshness=0.15)),
    "pig": (Body(mass=100, tract_length=0.22, roughness=0.6, nasality=0.0, chest=1.0, chest_width=180.0,
                 noise_color=2.0, hf_loss=0.3, breathiness=0.1, tract_damping=1.0, low_cut=50.0, open_quotient=0.4),
            Voice(vowels=["o", "a"], calls={
                # Grunt: a low (~92 Hz), heavy, gravelly voiced swell. The gravel is smooth, irregular
                # 20-50 Hz roughness; breath noise and deep fry pulses both turned it into puffs or a
                # propeller-like pulse train. Weight comes from the long tract and a broad chest resonance.
                "idle": {"pitch": -8.5, "contour": [(0, -1), (0.3, 1), (1, -2)], "harsh": 0.2, "dur": 0.3,
                         "amp": [(0, 0), (0.15, 1), (0.6, 0.8), (1, 0)], "breath": -0.05, "rumble": 0.4,
                         "repeats": (1, 2), "gap": 0.3, "nasal": 0.0, "spread": 1.0, "vowels": ["e"],
                         "mouth": [(0, 0.6), (0.4, 0.9), (1, 0.6)]},
                # Oink: the grunt recipe, longer and shaped, gliding "o" to "i" with a rise and fall.
                # No creaky tail and only light gravel: either one makes it choppy.
                "beg": {"pitch": -7.5, "contour": [(0, -1), (0.35, 2), (1, -2.5)], "harsh": 0.2, "dur": 0.45,
                        "nasal": 0.0, "glide": True, "spread": 1.0, "breath": -0.05, "rumble": 0.15, "fry": 0.0,
                        "amp": [(0, 0), (0.12, 1), (0.7, 0.85), (1, 0)], "repeats": (1, 2), "gap": 0.35,
                        "vowels": ["o", "i"], "mouth": [(0, 0.5), (0.4, 0.9), (1, 0.5)], "inhale": False},
                # Squeal: roughness from the voice itself (pressed, chaotic) rather than added breath
                # noise, which reads as air escaping a balloon. The pitch and loudness break up and down
                # through the call ("wee-ee") instead of one long sag; the vowel sits between "e" and "i".
                "flee": {"pitch": 29.5, "contour": [(0, -3), (0.08, 2), (0.3, 0.5), (0.45, 2), (0.7, 0), (0.85, 1), (1, -3)],
                         "harsh": 0.6, "breath": -1.0, "vibrato": 0.3, "vibrato_rate": 9.0,
                         "amp": [(0, 0), (0.04, 1), (0.3, 0.8), (0.45, 1), (0.7, 0.8), (0.85, 0.95), (1, 0)],
                         "dur": 1.9, "repeats": (1, 1), "vowels": [(0.8, 1.38, 1.1)], "inhale": False, "fry": 0.0,
                         "mouth": [(0, 0.8), (0.15, 1), (1, 0.8)]},
                "territorial": {"like": "warn"}})),
    "cat": (Body(mass=4, roughness=0.15, open_quotient=0.65),
            Voice(vowels=["i", "a", "u"], pitch=3, pitch_range=1.5, subharmonics=0.3, calls={
                "idle": {"pitch": 7, "dur": 1.0, "repeats": (1, 1), "vowels": ["a"],
                         "contour": [(0, -1), (0.2, 0.5), (0.7, -0.5), (1, 1)]},
                "mating": {"pitch": -0.5, "dur": 1.4, "repeats": (1, 1), "vowels": ["e", "a", "a"], "harsh": 0.2,
                           "inhale": False, "mouth": [(0, 0.5), (0.3, 1.0), (0.7, 0.9), (1, 0.5)]}})),
    "mouse": (Body(mass=0.03, roughness=0.1, breathiness=0.1),
              Voice(vowels=["i"], pitch_range=1.3, tempo=1.4)),
    "songbird": (Body(mass=0.03, source="syrinx", roughness=0.05),
                 Voice(vowels=["i", "e"], pitch_range=2.0, vibrato=0.5, tempo=1.3)),
    "crow": (Body(mass=0.5, source="syrinx", roughness=0.5, tract_length=0.08, buzz=1.0, open_quotient=0.65, formant_gain=12.0),
             Voice(vowels=["a"], pitch=-30, calls={
                 "idle": {"dur": 0.65, "repeats": (3, 6), "gap": 0.4, "harsh": 0.45, "vowels": ["o"], "spread": 1.0,
                          "mouth": [(0, 0.3), (0.3, 0.6), (1, 0.3)]},
                 "alarm": {"dur": 0.65, "repeats": (3, 5), "gap": 0.5, "harsh": 0.6, "vowels": ["o"], "spread": 1.0,
                           "mouth": [(0, 0.3), (0.3, 0.6), (1, 0.3)]},
                 "territorial": {"like": "idle", "repeats": (5, 9)}})),
    "duck": (Body(mass=1.2, source="syrinx", tract_length=0.075, roughness=0.3, nasality=0.45, buzz=1.0, tract_damping=0.6,
                 open_quotient=0.65, formant_gain=0.0),  # the bill already rings sharply
             # Quacks: a flat-topped burst that stops dead, with the bill opening "o" to "a" so the
             # formant sweeps up into one sharp peak near 2 kHz.
             Voice(vowels=["o", "a"], pitch=-45, calls={
                 "idle": {"dur": 0.28, "contour": [(0, -2), (0.6, 1), (1, 1.5)], "repeats": (1, 3), "gap": 0.35,
                          "harsh": 0.35, "glide": True, "fry": 0.0, "amp": [(0, 0), (0.06, 1), (0.85, 0.9), (1, 0)],
                          "mouth": [(0, 0.3), (0.4, 0.7), (1, 0.6)]},
                 "alarm": {"pitch": 5, "dur": 0.28, "contour": [(0, -2), (0.6, 1), (1, 1.5)], "repeats": (8, 14),
                           "gap": 0.25, "glide": True, "fry": 0.0, "amp": [(0, 0), (0.06, 1), (0.85, 0.9), (1, 0)],
                           "mouth": [(0, 0.3), (0.4, 0.7), (1, 0.6)]},
                 "territorial": {"like": "alarm"}})),
    "frog": (Body(mass=0.3, source="pulse", resonator_freq=1100, resonator_harmonic=1.3, fold_scale=0.67, roughness=0.2),
             Voice(vowels=["@"], pitch_range=0.5, calls={
                 "mating": {"dur": 0.7, "repeats": (8, 11), "gap": 0.65, "inhale": False},
                 "idle": {"dur": 0.7, "repeats": (3, 6), "gap": 0.45},
                 "territorial": {"like": "mating"}})),
    "cricket": (Body(mass=0.001, source="pulse", resonator_freq=4500, fold_scale=0.4),
                Voice(vowels=["@"], pitch_range=0.3, tempo=1.5)),
    "snake": (Body(mass=3, source="hiss", tract_length=0.12, breathiness=1.0),
              Voice(vowels=["i", "@"])),
}


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
        sd.play(audio, SR)
        sd.wait()
    except Exception:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Body(**data.get("body", {})), Voice(**data.get("voice", {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="lion", choices=sorted(PRESETS))
    ap.add_argument("--json", help="body/voice JSON file (overrides --preset)")
    ap.add_argument("--intent", default=None, choices=INTENT_NAMES, help="default: idle, or speak with --say")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--distance", type=float, default=20.0, help="listener distance in metres")
    ap.add_argument("--reverb", type=float, default=1.0, help="reverb amount multiplier (0 = dry)")
    ap.add_argument("--demo", action="store_true", help="render every intent into --out-dir")
    ap.add_argument("--all-presets", action="store_true", help="with --demo: every preset")
    ap.add_argument("--out", default="vocalization.wav")
    ap.add_argument("--out-dir", default="vocalizations")
    ap.add_argument("--no-play", action="store_true")
    for f in fields(Body):
        if f.type is float:
            ap.add_argument("--" + f.name.replace("_", "-"), type=float, help=f"Body.{f.name}")
    ap.add_argument("--source", choices=["larynx", "syrinx", "pulse", "hiss"])
    ap.add_argument("--vowels", help="comma-separated vowel list, e.g. i,a,u")
    ap.add_argument("--say", help="English text for the speak intent (english_phrase.py), e.g. \"Hey, Heath!\"")
    args = ap.parse_args()
    if args.intent is None:
        args.intent = "speak" if args.say else "idle"

    def resolve(preset):
        body, voice = load_json(args.json) if args.json else PRESETS[preset]
        overrides = {f.name: getattr(args, f.name) for f in fields(Body) if getattr(args, f.name, None) is not None}
        body = Body(**{**body.__dict__, **overrides})
        if args.vowels:
            voice = Voice(**{**voice.__dict__, "vowels": args.vowels.split(",")})
        if args.say:
            from english_phrase import english_to_phrase
            phrase = english_to_phrase(args.say)
            if not phrase:
                raise SystemExit(f"--say: nothing in {args.say!r} can be voiced")
            print(f"say: {phrase}")
            calls = {**voice.calls, "speak": {**voice.calls.get("speak", {}), "vowels": phrase}}
            voice = Voice(**{**voice.__dict__, "calls": calls})
        return body, voice

    if args.demo:
        os.makedirs(args.out_dir, exist_ok=True)
        for preset in (sorted(PRESETS) if args.all_presets else [args.preset]):
            body, voice = resolve(preset)
            for intent in INTENT_NAMES:
                audio = create_vocalization(body, voice, intent, args.seed, args.distance, args.reverb)
                path = os.path.join(args.out_dir, f"{preset}_{intent}.wav")
                write_wav(path, audio)
                print(f"{path} ({len(audio) / SR:.1f}s)")
        return

    body, voice = resolve(args.preset)
    audio = create_vocalization(body, voice, args.intent, args.seed, args.distance, args.reverb)
    write_wav(args.out, audio)
    print(f"Wrote {args.out} ({len(audio) / SR:.1f}s): {body.source}, {body.mass:g} kg, "
          f"tract {body.tract:.3f} m, base f0 {body.base_f0:.0f} Hz")
    if not args.no_play:
        play(args.out, audio)


if __name__ == "__main__":
    main()
