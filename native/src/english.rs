//! English text to a sequential-call phrase ("hey heath" -> "h eh~ih h iy th").
//!
//! Words are looked up in the CMU Pronouncing Dictionary (data/cmudict.dict, from
//! https://github.com/cmusphinx/cmudict, BSD licence in data/cmudict.LICENSE). Words it doesn't
//! know (made-up names) go through letter-to-sound rules (data/english_rules.txt). The ARPAbet
//! phonemes are then mapped onto what the synth can say:
//!
//! - vowels to the SSBE vowel names, diphthongs to glides (EY -> "eh~ih");
//! - w and y to a glide into the next vowel (W EH -> "uw~eh");
//! - the voiceless consonants h s sh f th, with voiced and affricate ones mapped to the nearest
//!   (z -> s, v -> f, dh -> th, zh -> sh), and the affricates ch and j;
//! - the stops p t k b d g;
//! - nasals, l and r dropped (the synth can't make them);
//! - punctuation (, ; : . ! ? and dashes) to a pause, "|".

use std::collections::HashMap;
use std::sync::OnceLock;

const DICT: &str = include_str!("../data/cmudict.dict");
const RULES: &str = include_str!("../data/english_rules.txt");

/// word (lower case) -> ARPAbet phonemes with stress digits; the first pronunciation of each word.
fn dictionary() -> &'static HashMap<&'static str, Vec<&'static str>> {
    static D: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();
    D.get_or_init(|| {
        let mut map = HashMap::with_capacity(140_000);
        for line in DICT.lines() {
            let line = line.split('#').next().unwrap_or("");
            let mut parts = line.split_whitespace();
            let Some(word) = parts.next() else { continue };
            if word.contains('(') {
                continue; // alternative pronunciation
            }
            map.entry(word).or_insert_with(|| parts.collect());
        }
        map
    })
}

struct Rule {
    left: Vec<char>,
    pattern: Vec<char>,
    right: Vec<char>,
    phonemes: Vec<&'static str>,
}

fn rules() -> &'static HashMap<char, Vec<Rule>> {
    static R: OnceLock<HashMap<char, Vec<Rule>>> = OnceLock::new();
    R.get_or_init(|| {
        let mut map: HashMap<char, Vec<Rule>> = HashMap::new();
        for line in RULES.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let (Some(open), Some(close), Some(eq)) = (line.find('['), line.find(']'), line.find('=')) else { continue };
            let pattern: Vec<char> = line[open + 1..close].chars().collect();
            let Some(&first) = pattern.first() else { continue };
            map.entry(first).or_default().push(Rule {
                left: line[..open].chars().collect(),
                pattern,
                right: line[close + 1..eq].chars().collect(),
                phonemes: line[eq + 1..].split_whitespace().collect(),
            });
        }
        map
    })
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'A' | 'E' | 'I' | 'O' | 'U')
}

fn is_consonant(c: char) -> bool {
    c.is_ascii_alphabetic() && !is_vowel(c)
}

/// Does `ctx` fit the word read forwards from `i` (right context)?
fn fits_right(w: &[char], mut i: usize, ctx: &[char]) -> bool {
    let at = |i: usize| w.get(i).copied().unwrap_or(' ');
    for &c in ctx {
        match c {
            '_' => {
                if at(i).is_ascii_alphabetic() {
                    return false;
                }
                i += 1;
            }
            '#' => {
                let s = i;
                while is_vowel(at(i)) {
                    i += 1;
                }
                if i == s {
                    return false;
                }
            }
            ':' => {
                while is_consonant(at(i)) {
                    i += 1;
                }
            }
            '^' => {
                if !is_consonant(at(i)) {
                    return false;
                }
                i += 1;
            }
            '.' => {
                if !"BDVGJLMNRWZ".contains(at(i)) {
                    return false;
                }
                i += 1;
            }
            '+' => {
                if !"EIY".contains(at(i)) {
                    return false;
                }
                i += 1;
            }
            '%' => {
                // ER E ES ED ING ELY
                if at(i) == 'E' {
                    i += 1;
                    if at(i) == 'L' && at(i + 1) == 'Y' {
                        i += 2;
                    } else if matches!(at(i), 'R' | 'S' | 'D') {
                        i += 1;
                    }
                } else if at(i) == 'I' && at(i + 1) == 'N' && at(i + 2) == 'G' {
                    i += 3;
                } else {
                    return false;
                }
            }
            '&' => {
                if (at(i) == 'C' || at(i) == 'S') && at(i + 1) == 'H' {
                    i += 2;
                } else if "SCGZXJ".contains(at(i)) {
                    i += 1;
                } else {
                    return false;
                }
            }
            '@' => {
                if matches!(at(i), 'T' | 'C' | 'S') && at(i + 1) == 'H' {
                    i += 2;
                } else if "TSRDLZNJ".contains(at(i)) {
                    i += 1;
                } else {
                    return false;
                }
            }
            letter => {
                if at(i) != letter {
                    return false;
                }
                i += 1;
            }
        }
    }
    true
}

/// Does `ctx` fit the word read backwards from just before `end` (left context)?
fn fits_left(w: &[char], end: usize, ctx: &[char]) -> bool {
    let mut i = end as isize - 1;
    let at = |i: isize| if i >= 0 { w.get(i as usize).copied().unwrap_or(' ') } else { ' ' };
    for &c in ctx.iter().rev() {
        match c {
            '_' => {
                if at(i).is_ascii_alphabetic() {
                    return false;
                }
                i -= 1;
            }
            '#' => {
                let s = i;
                while is_vowel(at(i)) {
                    i -= 1;
                }
                if i == s {
                    return false;
                }
            }
            ':' => {
                while is_consonant(at(i)) {
                    i -= 1;
                }
            }
            '^' => {
                if !is_consonant(at(i)) {
                    return false;
                }
                i -= 1;
            }
            '.' => {
                if !"BDVGJLMNRWZ".contains(at(i)) {
                    return false;
                }
                i -= 1;
            }
            '+' => {
                if !"EIY".contains(at(i)) {
                    return false;
                }
                i -= 1;
            }
            '&' => {
                if at(i) == 'H' && (at(i - 1) == 'C' || at(i - 1) == 'S') {
                    i -= 2;
                } else if "SCGZXJ".contains(at(i)) {
                    i -= 1;
                } else {
                    return false;
                }
            }
            '@' => {
                if at(i) == 'H' && matches!(at(i - 1), 'T' | 'C' | 'S') {
                    i -= 2;
                } else if "TSRDLZNJ".contains(at(i)) {
                    i -= 1;
                } else {
                    return false;
                }
            }
            letter => {
                if at(i) != letter {
                    return false;
                }
                i -= 1;
            }
        }
    }
    true
}

/// A word's phonemes by the letter-to-sound rules.
fn rule_phonemes(word: &str) -> Vec<String> {
    let w: Vec<char> = word.to_ascii_uppercase().chars().collect();
    let rules = rules();
    let mut out = Vec::new();
    let mut i = 0;
    while i < w.len() {
        let applied = rules.get(&w[i]).and_then(|list| {
            list.iter().find(|r| {
                w[i..].starts_with(&r.pattern) && fits_left(&w, i, &r.left) && fits_right(&w, i + r.pattern.len(), &r.right)
            })
        });
        match applied {
            Some(r) => {
                out.extend(r.phonemes.iter().map(|p| p.to_string()));
                i += r.pattern.len();
            }
            None => i += 1,
        }
    }
    out
}

/// A word's ARPAbet phonemes (with stress digits on vowels where the dictionary gives them):
/// from the dictionary, else by rule.
pub fn word_phonemes(word: &str) -> Vec<String> {
    let lower = word.to_lowercase();
    let dict = dictionary();
    let found = dict.get(lower.as_str()).or_else(|| dict.get(lower.trim_matches('\'')));
    match found {
        Some(p) => p.iter().map(|s| s.to_string()).collect(),
        None => rule_phonemes(lower.trim_matches('\'')),
    }
}

const DIGITS: [&str; 10] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];

/// English text as ARPAbet phonemes, with "|" where punctuation calls for a pause.
pub fn english_to_phonemes(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut Vec<String>| {
        if !word.is_empty() {
            out.extend(word_phonemes(word));
            word.clear();
        }
    };
    let chars: Vec<char> = text.chars().collect();
    for (k, &c) in chars.iter().enumerate() {
        if c.is_alphabetic() || (c == '\'' && !word.is_empty()) || (c == '\u{2019}' && !word.is_empty()) {
            word.push(if c == '\u{2019}' { '\'' } else { c });
        } else {
            flush(&mut word, &mut out);
            if let Some(d) = c.to_digit(10) {
                out.extend(word_phonemes(DIGITS[d as usize]));
            } else if matches!(c, ',' | ';' | ':' | '.' | '!' | '?' | '(' | ')' | '\u{2014}' | '\u{2013}')
                || (c == '-' && !(k > 0 && chars[k - 1].is_alphabetic() && chars.get(k + 1).is_some_and(|n| n.is_alphabetic())))
            {
                if out.last().is_some_and(|l| l != "|") {
                    out.push("|".into());
                }
            }
        }
    }
    flush(&mut word, &mut out);
    while out.last().is_some_and(|l| l == "|") {
        out.pop();
    }
    out
}

/// One ARPAbet vowel as the synth's vowel (or a "~" glide for a diphthong). Unstressed AH is a
/// schwa.
fn vowel(base: &str, stress: &str) -> Option<&'static str> {
    Some(match base {
        "IY" => "iy",
        "IH" => "ih",
        "EH" => "eh",
        "AE" => "ae",
        "AA" => "aa",
        "AO" => "ao",
        "AH" if stress == "0" => "@",
        "AH" => "ah",
        "UH" => "uh",
        "UW" => "uw",
        "ER" => "er",
        "EY" => "eh~ih",
        "AY" => "a~ih",
        "AW" => "a~uh",
        "OY" => "ao~ih",
        "OW" => "o~uh",
        _ => return None,
    })
}

/// ARPAbet phonemes as a phrase for a sequential call.
pub fn phonemes_to_phrase(phonemes: &[String]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut glide: Option<&str> = None;
    for p in phonemes {
        if p == "|" {
            glide = None;
            if out.last().is_some_and(|l| l != "|") {
                out.push("|".into());
            }
            continue;
        }
        let base = p.trim_end_matches(|c: char| c.is_ascii_digit());
        let stress = &p[base.len()..];
        if let Some(v) = vowel(base, stress) {
            out.push(match glide.take() {
                Some(g) => format!("{g}~{v}"),
                None => v.to_string(),
            });
            continue;
        }
        glide = None;
        let token = match base {
            "W" => {
                glide = Some("uw");
                continue;
            }
            "Y" => {
                glide = Some("iy");
                continue;
            }
            "HH" => "h",
            "S" | "Z" => "s",
            "SH" | "ZH" => "sh",
            "CH" => "ch",
            "JH" => "j",
            "F" | "V" => "f",
            "TH" | "DH" => "th",
            "P" => "p",
            "T" => "t",
            "K" => "k",
            "B" => "b",
            "D" => "d",
            "G" => "g",
            _ => continue, // nasals, l, r: the synth can't make them
        };
        out.push(token.to_string());
    }
    while out.last().is_some_and(|l| l == "|") {
        out.pop();
    }
    if out.first().is_some_and(|l| l == "|") {
        out.remove(0);
    }
    out.join(" ")
}

/// English text as a phrase for the "speak" intent (or any intent with "sequence": true), e.g.
/// "Hey, Heath!" -> "h eh~ih | h iy th". Empty if nothing in it can be voiced.
pub fn english_to_phrase(text: &str) -> String {
    phonemes_to_phrase(&english_to_phonemes(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hey_heath() {
        assert_eq!(english_to_phrase("hey heath"), "h eh~ih h iy th");
        assert_eq!(english_to_phrase("Hey, Heath!"), "h eh~ih | h iy th");
    }

    #[test]
    fn glides_and_unknown_words() {
        assert_eq!(english_to_phrase("why"), "uw~a~ih");
        assert_eq!(english_to_phrase("you"), "iy~uw");
        // Not in the dictionary: by rule
        assert!(!english_to_phrase("Grashnak").is_empty());
    }

    #[test]
    fn stops_are_kept() {
        assert_eq!(english_to_phrase("get back"), "g eh t b ae k");
    }

    #[test]
    fn stops_render() {
        use crate::{create_vocalization, Body, Voice, VowelSpec};
        let body = Body { mass: 25.0, ..Body::default() };
        let mut voice = Voice::default();
        voice.vowels = vec![VowelSpec::Name("g eh t b ae k | b ih g g eh~ih t | aa d aa p".into())];
        let y = create_vocalization(&body, &voice, "speak", 1, 0.0, 0.0).unwrap();
        assert!(y.len() > 1000 && y.iter().all(|v| v.is_finite()));
    }
}
