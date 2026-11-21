//! Procedural creature vocalizations: a native port of `create_vocalization.py`.
//!
//! A sound is decided by a [`Body`] (physical traits), a [`Voice`] (heritable traits) and an
//! intent (what the call means). See the Python module and README for the model itself.
//!
//! ```no_run
//! use faunavox::{Creature, create_vocalization};
//! let creature = Creature::from_json(r#"{"body": {"mass": 40}, "voice": {"vowels": ["u", "o"]}}"#).unwrap();
//! let audio = create_vocalization(&creature.body, &creature.voice, "mating", 1, 20.0, 0.0).unwrap();
//! ```

pub mod dsp;
pub mod english;
pub mod field;
pub mod genome;
pub mod ffi;
pub mod model;
pub mod mouth;
pub mod presets;
pub mod rng;
pub mod space;
pub mod synth;

pub use english::english_to_phrase;
pub use model::{Body, Creature, Source, VowelSpec, Voice};
pub use mouth::{mouth_track, MouthTrack};
pub use synth::{create_vocalization, INTENT_NAMES, SR};
