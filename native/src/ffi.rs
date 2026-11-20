//! C ABI for Unity (P/Invoke) or any native host.
//!
//! Handles are opaque pointers. Every call is panic-safe: on failure it returns null / a negative
//! value and `cv_last_error()` describes why (per thread). A creature handle is read-only while
//! rendering, so several threads may render from the same handle at once; `cv_creature_set*`
//! must not run concurrently with renders of that handle.
//!
//! Typical use:
//!   CvCreature* c = cv_creature_from_json(json);          // or cv_creature_from_preset("wolf")
//!   // or from a genome: cv_creature_from_genome(body_json, voice_dna, NULL)
//!   cv_creature_set(c, "mass", 42.0);                     // tweaks
//!   CvBuffer* b = cv_render(c, "mating", seed, 20.0f, 0.0f);
//!   copy cv_buffer_len(b) floats from cv_buffer_data(b); cv_buffer_free(b);
//!   cv_creature_free(c);

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::model::Creature;
use crate::{presets, synth};

/// A creature, and the decoder's report when it came from a genome.
pub struct CvCreature(Creature, Option<String>);
pub struct CvBuffer(Vec<f32>);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl Into<String>) {
    let s = msg.into().replace('\0', " ");
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(s).unwrap_or_default());
}

fn guard<T>(fail: T, f: impl FnOnce() -> Result<T, String>) -> T {
    set_error("");
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(e);
            fail
        }
        Err(p) => {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
            set_error(format!("internal error: {}", msg.unwrap_or_else(|| "panic".into())));
            fail
        }
    }
}

unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what} is null"));
    }
    CStr::from_ptr(p).to_str().map_err(|_| format!("{what} is not valid UTF-8"))
}

fn static_cstr(cell: &'static std::sync::OnceLock<CString>, make: impl FnOnce() -> String) -> *const c_char {
    cell.get_or_init(|| CString::new(make()).unwrap_or_default()).as_ptr()
}

/// Last error on this thread ("" if the last call succeeded). Valid until the next cv_* call
/// on this thread.
#[no_mangle]
pub extern "C" fn cv_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn cv_version() -> *const c_char {
    static V: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    static_cstr(&V, || env!("CARGO_PKG_VERSION").to_string())
}

#[no_mangle]
pub extern "C" fn cv_sample_rate() -> i32 {
    synth::SR as i32
}

/// Comma-separated intent names.
#[no_mangle]
pub extern "C" fn cv_intent_names() -> *const c_char {
    static V: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    static_cstr(&V, || synth::INTENT_NAMES.join(","))
}

/// Comma-separated preset names.
#[no_mangle]
pub extern "C" fn cv_preset_names() -> *const c_char {
    static V: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    static_cstr(&V, || presets::names().join(","))
}

/// Parse a creature from JSON: {"body": {...}, "voice": {...}} (same format as the Python tool).
#[no_mangle]
pub unsafe extern "C" fn cv_creature_from_json(json: *const c_char) -> *mut CvCreature {
    guard(ptr::null_mut(), || {
        let c = Creature::from_json(str_arg(json, "json")?)?;
        Ok(Box::into_raw(Box::new(CvCreature(c, None))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn cv_creature_from_preset(name: *const c_char) -> *mut CvCreature {
    guard(ptr::null_mut(), || {
        let name = str_arg(name, "name")?;
        let c = presets::get(name).ok_or_else(|| format!("unknown preset {name:?}; choose from {}", presets::names().join(", ")))?;
        Ok(Box::into_raw(Box::new(CvCreature(c, None))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn cv_creature_clone(c: *const CvCreature) -> *mut CvCreature {
    guard(ptr::null_mut(), || {
        let c = c.as_ref().ok_or("creature is null")?;
        Ok(Box::into_raw(Box::new(CvCreature(c.0.clone(), c.1.clone()))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn cv_creature_free(c: *mut CvCreature) {
    if !c.is_null() {
        drop(Box::from_raw(c));
    }
}

/// Set a numeric Body or Voice field by name ("mass", "tract_length", "pitch", "harshness",
/// ...; "body."/"voice." prefixes are optional). Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn cv_creature_set(c: *mut CvCreature, field: *const c_char, value: f64) -> i32 {
    guard(-1, || {
        let c = c.as_mut().ok_or("creature is null")?;
        let num = serde_json::Number::from_f64(value).ok_or("value must be finite")?;
        c.0.set_field(str_arg(field, "field")?, serde_json::Value::Number(num))?;
        Ok(0)
    })
}

/// Set any field to a JSON value, e.g. ("vowels", "[\"u\",\"o\"]"), ("source", "\"syrinx\""),
/// ("calls", "{\"idle\": {\"dur\": 0.6}}"). Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn cv_creature_set_json(c: *mut CvCreature, field: *const c_char, json_value: *const c_char) -> i32 {
    guard(-1, || {
        let c = c.as_mut().ok_or("creature is null")?;
        let v: serde_json::Value = serde_json::from_str(str_arg(json_value, "json_value")?).map_err(|e| format!("value JSON: {e}"))?;
        c.0.set_field(str_arg(field, "field")?, v)?;
        Ok(0)
    })
}

/// The creature as JSON. Free the string with cv_string_free.
#[no_mangle]
pub unsafe extern "C" fn cv_creature_to_json(c: *const CvCreature) -> *mut c_char {
    guard(ptr::null_mut(), || {
        let c = c.as_ref().ok_or("creature is null")?;
        Ok(CString::new(c.0.to_json()).map_err(|e| e.to_string())?.into_raw())
    })
}

/// Decode a creature from the voice section of its genome (GENOME_VOICE_DESIGN.md §4).
/// `body_json`: {"mass": 12.4, "tract_length" or "head_length" + "neck_length", "source":
/// "larynx"|"syrinx"|"pulse"|"hiss", "shape": "quadruped", "lips": true, "teeth": true}.
/// `dna`: the voice section (A, C, G, T). `versions_json` may be null (the versions built in) or
/// {"decoder": 2, "space": 1, "field": 1}; "field": false decodes without steering.
/// Returns a creature handle like cv_creature_from_json, or null on error.
#[no_mangle]
pub unsafe extern "C" fn cv_creature_from_genome(body_json: *const c_char, dna: *const c_char, versions_json: *const c_char) -> *mut CvCreature {
    guard(ptr::null_mut(), || {
        let body: crate::space::BodyInput =
            serde_json::from_str(str_arg(body_json, "body_json")?).map_err(|e| format!("body JSON: {e}"))?;
        let versions = if versions_json.is_null() {
            crate::genome::Versions::default()
        } else {
            crate::genome::Versions::from_json(str_arg(versions_json, "versions_json")?)?
        };
        let (c, report) = crate::genome::decode(&body, str_arg(dna, "dna")?, &versions)?;
        let report = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(CvCreature(c, Some(report)))))
    })
}

/// What the decoder did for a creature made by cv_creature_from_genome, as JSON: the trunk's and
/// each call's position on the axes, badness at each, the expansion size and the versions used.
/// Free the string with cv_string_free. Null (with an error) for a creature not from a genome.
#[no_mangle]
pub unsafe extern "C" fn cv_genome_report(c: *const CvCreature) -> *mut c_char {
    guard(ptr::null_mut(), || {
        let c = c.as_ref().ok_or("creature is null")?;
        let r = c.1.as_ref().ok_or("this creature wasn't decoded from a genome")?;
        Ok(CString::new(r.as_str()).map_err(|e| e.to_string())?.into_raw())
    })
}

/// The preference field built into this DLL (its version number).
#[no_mangle]
pub extern "C" fn cv_field_version() -> i32 {
    guard(-1, || Ok(crate::field::builtin().version as i32))
}

/// The genome decoder's version.
#[no_mangle]
pub extern "C" fn cv_decoder_version() -> i32 {
    crate::genome::DECODER_VERSION as i32
}

/// English text as a phrase for the "speak" intent (vowels, consonants and pauses), e.g.
/// "Hey, Heath!" -> "h eh~ih | h iy th". Set it as a call's vowels with cv_creature_set_json.
/// Free the string with cv_string_free. Returns null on error.
#[no_mangle]
pub unsafe extern "C" fn cv_english_to_phrase(text: *const c_char) -> *mut c_char {
    guard(ptr::null_mut(), || {
        let text = str_arg(text, "text")?;
        Ok(CString::new(crate::english::english_to_phrase(text)).map_err(|e| e.to_string())?.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "C" fn cv_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Render one vocalization: mono float samples at cv_sample_rate(), peak-normalized to 0.9.
/// `distance` in metres (air absorption, reverb mix); `reverb` 0 = dry. Returns null on error.
/// This is CPU work (tens of milliseconds); call it from a worker thread, not the audio thread.
#[no_mangle]
pub unsafe extern "C" fn cv_render(c: *const CvCreature, intent: *const c_char, seed: u64, distance: f32, reverb: f32) -> *mut CvBuffer {
    guard(ptr::null_mut(), || {
        let c = c.as_ref().ok_or("creature is null")?;
        let audio = synth::create_vocalization(&c.0.body, &c.0.voice, str_arg(intent, "intent")?, seed, distance as f64, reverb as f64)?;
        Ok(Box::into_raw(Box::new(CvBuffer(audio))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn cv_buffer_len(b: *const CvBuffer) -> i32 {
    b.as_ref().map_or(0, |b| b.0.len() as i32)
}

#[no_mangle]
pub unsafe extern "C" fn cv_buffer_data(b: *const CvBuffer) -> *const f32 {
    b.as_ref().map_or(ptr::null(), |b| b.0.as_ptr())
}

/// Copy up to `capacity` samples into `dst`; returns the number copied.
#[no_mangle]
pub unsafe extern "C" fn cv_buffer_copy(b: *const CvBuffer, dst: *mut f32, capacity: i32) -> i32 {
    match (b.as_ref(), dst.is_null()) {
        (Some(b), false) if capacity > 0 => {
            let n = b.0.len().min(capacity as usize);
            ptr::copy_nonoverlapping(b.0.as_ptr(), dst, n);
            n as i32
        }
        _ => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cv_buffer_free(b: *mut CvBuffer) {
    if !b.is_null() {
        drop(Box::from_raw(b));
    }
}
