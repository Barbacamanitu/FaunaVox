"""English text to a phrase for a sequential call ("hey heath" -> "h eh~ih h iy th").

Same as the native crate's english.rs, from the same data:
- words are looked up in the CMU Pronouncing Dictionary (native/data/cmudict.dict, from
  https://github.com/cmusphinx/cmudict, BSD licence in native/data/cmudict.LICENSE);
- words it doesn't know (made-up names) go through letter-to-sound rules
  (native/data/english_rules.txt).

The ARPAbet phonemes are then mapped onto what the synth can say:
- vowels go to the SSBE vowel names, and diphthongs to glides (EY -> "eh~ih");
- w and y become a glide into the next vowel (W EH -> "uw~eh");
- the voiceless consonants h s sh f th are kept, and voiced or affricate ones go to the nearest
  (z -> s, v -> f, dh -> th, zh -> sh), and the affricates ch and j;
- the stops p t k b d g are kept;
- nasals, l and r are dropped;
- punctuation becomes a pause, "|".

    python english_phrase.py "Hey, Heath!"      # prints: h eh~ih | h iy th
"""
import os
import sys

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "data")
_dict = None
_rules = None

VOWELS = "AEIOU"
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
VOWEL_MAP = {"IY": "iy", "IH": "ih", "EH": "eh", "AE": "ae", "AA": "aa", "AO": "ao", "AH": "ah",
             "UH": "uh", "UW": "uw", "ER": "er", "EY": "eh~ih", "AY": "a~ih", "AW": "a~uh",
             "OY": "ao~ih", "OW": "o~uh"}
CONSONANT_MAP = {"HH": "h", "S": "s", "Z": "s", "SH": "sh", "ZH": "sh", "CH": "ch", "JH": "j",
                 "F": "f", "V": "f", "TH": "th", "DH": "th",
                 "P": "p", "T": "t", "K": "k", "B": "b", "D": "d", "G": "g"}
PAUSE_CHARS = ",;:.!?()—–"


def dictionary():
    """word (lower case) -> ARPAbet phonemes with stress digits (first pronunciation)."""
    global _dict
    if _dict is None:
        _dict = {}
        with open(os.path.join(_DATA, "cmudict.dict"), encoding="utf-8") as f:
            for line in f:
                parts = line.split("#")[0].split()
                if parts and "(" not in parts[0]:
                    _dict.setdefault(parts[0], parts[1:])
    return _dict


def rules():
    global _rules
    if _rules is None:
        _rules = {}
        with open(os.path.join(_DATA, "english_rules.txt"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                open_, close, eq = line.find("["), line.find("]"), line.find("=")
                if min(open_, close, eq) < 0 or close == open_ + 1:
                    continue
                pattern = line[open_ + 1:close]
                _rules.setdefault(pattern[0], []).append(
                    (line[:open_], pattern, line[close + 1:eq], line[eq + 1:].split()))
    return _rules


def _consonant(c):
    return c.isalpha() and c.isascii() and c not in VOWELS


def _fits(w, i, ctx, step):
    """Does context ctx fit word w from position i, reading forwards (step 1) or backwards (-1)?"""
    at = lambda k: w[k] if 0 <= k < len(w) else " "
    for c in (ctx if step > 0 else reversed(ctx)):
        if c == "_":
            if at(i).isalpha():
                return False
            i += step
        elif c == "#":
            s = i
            while at(i) in VOWELS and at(i) != " ":
                i += step
            if i == s:
                return False
        elif c == ":":
            while _consonant(at(i)):
                i += step
        elif c == "^":
            if not _consonant(at(i)):
                return False
            i += step
        elif c == ".":
            if at(i) not in "BDVGJLMNRWZ" or at(i) == " ":
                return False
            i += step
        elif c == "+":
            if at(i) not in "EIY" or at(i) == " ":
                return False
            i += step
        elif c == "%":
            if step < 0:
                return False
            if at(i) == "E":
                i += 1
                if at(i) == "L" and at(i + 1) == "Y":
                    i += 2
                elif at(i) in ("R", "S", "D"):
                    i += 1
            elif at(i) == "I" and at(i + 1) == "N" and at(i + 2) == "G":
                i += 3
            else:
                return False
        elif c in "&@":
            pairs = ("C", "S") if c == "&" else ("T", "C", "S")
            singles = "SCGZXJ" if c == "&" else "TSRDLZNJ"
            if step > 0 and at(i) in pairs and at(i + 1) == "H":
                i += 2
            elif step < 0 and at(i) == "H" and at(i - 1) in pairs:
                i -= 2
            elif at(i) in singles and at(i) != " ":
                i += step
            else:
                return False
        else:
            if at(i) != c:
                return False
            i += step
    return True


def rule_phonemes(word):
    """A word's phonemes by the letter-to-sound rules."""
    w = word.upper()
    out, i = [], 0
    table = rules()
    while i < len(w):
        for left, pattern, right, phonemes in table.get(w[i], []):
            if w.startswith(pattern, i) and _fits(w, i - 1, left, -1) and _fits(w, i + len(pattern), right, 1):
                out += phonemes
                i += len(pattern)
                break
        else:
            i += 1
    return out


def word_phonemes(word):
    """A word's ARPAbet phonemes: from the dictionary, else by rule."""
    lower = word.lower()
    d = dictionary()
    found = d.get(lower) or d.get(lower.strip("'"))
    return list(found) if found is not None else rule_phonemes(lower.strip("'"))


def english_to_phonemes(text):
    """English text as ARPAbet phonemes, with "|" where punctuation calls for a pause."""
    out, word = [], ""

    def flush():
        nonlocal word
        if word:
            out.extend(word_phonemes(word))
            word = ""

    for k, c in enumerate(text):
        if c.isalpha() or (c in "'’" and word):
            word += "'" if c == "’" else c
            continue
        flush()
        if c.isdigit() and c.isascii():
            out.extend(word_phonemes(DIGITS[int(c)]))
        elif c in PAUSE_CHARS or (c == "-" and not (0 < k < len(text) - 1 and text[k - 1].isalpha()
                                                     and text[k + 1].isalpha())):
            if out and out[-1] != "|":
                out.append("|")
    flush()
    while out and out[-1] == "|":
        out.pop()
    return out


def phonemes_to_phrase(phonemes):
    """ARPAbet phonemes as a phrase for a sequential call."""
    out, glide = [], None
    for p in phonemes:
        if p == "|":
            glide = None
            if out and out[-1] != "|":
                out.append("|")
            continue
        base = p.rstrip("0123456789")
        stress = p[len(base):]
        if base in VOWEL_MAP:
            v = "@" if base == "AH" and stress == "0" else VOWEL_MAP[base]
            out.append(f"{glide}~{v}" if glide else v)
            glide = None
            continue
        glide = {"W": "uw", "Y": "iy"}.get(base)
        if base in CONSONANT_MAP:
            out.append(CONSONANT_MAP[base])
    while out and out[-1] == "|":
        out.pop()
    if out and out[0] == "|":
        out.pop(0)
    return " ".join(out)


def english_to_phrase(text):
    """English text as a phrase for the "speak" intent (or any intent with "sequence": true), e.g.
    "Hey, Heath!" -> "h eh~ih | h iy th". Empty if nothing in it can be voiced."""
    return phonemes_to_phrase(english_to_phonemes(text))


if __name__ == "__main__":
    print(english_to_phrase(" ".join(sys.argv[1:])))
