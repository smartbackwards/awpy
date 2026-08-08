//! Reusable technique demo + a specific (negative) finding.
//!
//! **Technique**: manually address a fixed-size C array's Nth element —
//! `resolve_field_key`'s dotted-path string parser has no case for fixed
//! arrays (only *dynamic*/networked-vector ones get numeric-index handling;
//! every element of a fixed array shares the identical `var_name`, so no
//! string path can select one). `FieldPath` is public in pbdems2 though, so
//! build one by hand the same way the library does internally for dynamic
//! arrays: `data[0] = field_idx`, `data[1] = array_idx`, `last = 1`, then
//! `.pack()`. Reuse this pattern for any other fixed-array field investigation.
//!
//! **This specific probe** was chasing whether `CFlashbang`'s `m_pReserveAmmo:
//! int32[2]` distinguishes holding 1 vs 2 flashbangs (CS2's one grenade type
//! you can carry two of; `PlayerState.flashbangs` currently can't, since only
//! one `CFlashbang` entity ever exists per player regardless of hold count —
//! see `fill_loadout`'s doc comment in `datasets.rs`). **Result: no.** Filtered
//! to player-owned (not dropped) entities, the pair reads a constant `(0, 1)`
//! in 100% of ~900k samples across a real match, whether holding 1 or 2 — this
//! field carries no information about hold count. Left here so the next
//! attempt doesn't have to re-derive the technique or re-check this field.
//!
//! `cargo run --release --example reserve_ammo_probe -- <demo>`

use std::collections::HashSet;
use std::path::PathBuf;

use awpy::Parser;
use awpy::entity::field_path::FieldPath;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("demo path"));

    let parser = Parser::from_file(&path).expect("open");
    let sers = parser.parse_send_tables().expect("send tables");
    let ser = sers.get("CFlashbang").expect("CFlashbang serializer");

    let field_idx = ser
        .fields
        .iter()
        .position(|f| f.var_name == "m_pReserveAmmo")
        .expect("m_pReserveAmmo field") as u8;
    println!("m_pReserveAmmo field_idx = {field_idx}");

    let key_of = |array_idx: u8| -> u64 {
        let mut fp = FieldPath::default();
        fp.data[0] = field_idx;
        fp.data[1] = array_idx;
        fp.last = 1;
        fp.pack()
    };
    let key0 = key_of(0);
    let key1 = key_of(1);
    println!("key[0] = {key0}, key[1] = {key1}");

    let owner_key = sers
        .get("CFlashbang")
        .and_then(|s| s.resolve_field_key("m_hOwnerEntity"));

    let filter: HashSet<&str> = HashSet::from(["CFlashbang"]);
    let mut owned_pairs: std::collections::HashMap<(i64, i64), u64> = std::collections::HashMap::new();
    // Track, per owner, the distinct entity indices seen holding a flashbang
    // across the whole demo -- if CS2 ever recycles a *second* CFlashbang
    // entity for the same owner within one continuous hold (rather than one
    // stable entity the whole time), that's a different signal worth knowing.
    let mut owner_entities: std::collections::HashMap<i64, HashSet<i32>> = std::collections::HashMap::new();

    parser
        .run_to_end_filtered(&filter, |ctx| {
            for (_, e) in ctx.entities().iter() {
                if !e.active || e.class_name.as_ref() != "CFlashbang" {
                    continue;
                }
                let owner = e.get_i64(owner_key);
                if owner == 16777215 {
                    continue; // unowned (dropped/world) -- skip
                }
                let v0 = e.get_i64(Some(key0));
                let v1 = e.get_i64(Some(key1));
                *owned_pairs.entry((v0, v1)).or_insert(0) += 1;
                owner_entities.entry(owner).or_default().insert(e.index);
            }
        })
        .expect("decode");

    println!("owned (v0,v1) distribution:");
    let mut pairs: Vec<_> = owned_pairs.into_iter().collect();
    pairs.sort();
    for ((v0, v1), n) in pairs {
        println!("  ({v0},{v1}): {n} tick-samples");
    }
    let max_entities_per_owner = owner_entities.values().map(|s| s.len()).max().unwrap_or(0);
    println!("max distinct CFlashbang entity indices ever seen for one owner: {max_entities_per_owner}");
}
