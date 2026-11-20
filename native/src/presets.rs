//! Built-in creature presets, tuned against real recordings (same as PRESETS in the Python).
//! Stored as JSON so they double as examples of the creature file format.

use std::collections::BTreeMap;

use crate::model::Creature;

const PRESETS_JSON: &str = include_str!("presets.json");

pub fn all() -> BTreeMap<String, Creature> {
    let mut map: BTreeMap<String, Creature> = serde_json::from_str(PRESETS_JSON).expect("presets.json is valid");
    for (name, c) in map.iter_mut() {
        c.name = Some(name.clone());
    }
    map
}

pub fn get(name: &str) -> Option<Creature> {
    all().remove(name)
}

pub fn names() -> Vec<String> {
    all().into_keys().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn presets_valid() {
        for (name, c) in super::all() {
            c.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
